"""GSPMD-sharded ensemble training and ablation sweep runner in JAX/Optax."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from itg_surrogate.data.dataset import PreparedStellaratorDataset
from itg_surrogate.models.cnn import StellaratorPeriodicCNN
from itg_surrogate.models.symmetry import D96ParityTransform, random_cyclic_shift
from itg_surrogate.registry.leading import LEADING_EXPERIMENT
from itg_surrogate.registry.specs import ExperimentConfig
from itg_surrogate.training.evaluation import eval_batched_single, eval_ensemble_sharded
from itg_surrogate.training.metrics import (
    ValidationSelectedEnsembleResult,
    compute_per_class_r2,
    compute_r2,
    compute_rmse,
    select_validation_topk_ensemble,
)
from itg_surrogate.training.sharding import configure_jax_float32, create_ensemble_mesh


@dataclasses.dataclass(frozen=True)
class ExperimentRunResult:
    """Trained parameters, predictions, and OOD metrics for one `ExperimentConfig`."""

    experiment: ExperimentConfig
    n_params: int
    final_train_mse: float
    final_val_mse: float
    val_over_train_mse_ratio: float
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    indiv_test_r2_mean: float
    indiv_test_r2_std: float
    ens_val_r2: float
    ens_test_r2: float
    ens_unstable_test_r2: float
    ens_test_rmse: float
    per_class_test_r2: dict[str, float]
    best_params: Any
    best_batch_stats: Any
    wall_time_sec: float

    def to_summary_dict(self) -> dict[str, Any]:
        """Serializes scalar metrics and experiment config to a JSON-ready dict."""
        return {
            "experiment": self.experiment.to_dict(),
            "n_params": self.n_params,
            "final_train_mse": self.final_train_mse,
            "final_val_mse": self.final_val_mse,
            "val_over_train_mse_ratio": self.val_over_train_mse_ratio,
            "indiv_test_r2_mean": self.indiv_test_r2_mean,
            "indiv_test_r2_std": self.indiv_test_r2_std,
            "ens_val_r2": self.ens_val_r2,
            "ens_test_r2": self.ens_test_r2,
            "ens_unstable_test_r2": self.ens_unstable_test_r2,
            "ens_test_rmse": self.ens_test_rmse,
            "per_class_test_r2": dict(self.per_class_test_r2),
            "wall_time_sec": self.wall_time_sec,
        }


@dataclasses.dataclass(frozen=True)
class SweepRunSummary:
    """Aggregated results across a multi-experiment ablation sweep."""

    runs: tuple[ExperimentRunResult, ...]
    topk_ensemble: ValidationSelectedEnsembleResult

    def to_summary_dict(self) -> dict[str, Any]:
        """Serializes all sweep runs and validation top-K* metrics to JSON."""
        return {
            "runs": [r.to_summary_dict() for r in self.runs],
            "validation_selected_topk_ensemble": {
                "best_k": self.topk_ensemble.best_k,
                "total_candidates": self.topk_ensemble.total_candidates,
                "selected_indices": self.topk_ensemble.selected_indices.tolist(),
                "val_r2": self.topk_ensemble.val_r2,
                "test_r2": self.topk_ensemble.test_r2,
                "unstable_test_r2": self.topk_ensemble.unstable_test_r2,
                "unstable_lnq_r2": self.topk_ensemble.unstable_lnq_r2,
                "test_rmse": self.topk_ensemble.test_rmse,
                "per_class_test_r2": dict(self.topk_ensemble.per_class_test_r2),
            },
        }


def wd_kernel_mask(params: Any) -> Any:
    """Masks `AdamW` weight decay strictly to `Conv`/`Dense` kernels (`ndim >= 2`)."""
    return jax.tree_util.tree_map(lambda leaf: leaf.ndim >= 2, params)


def build_optimizer(
    experiment: ExperimentConfig, steps_per_epoch: int
) -> optax.GradientTransformation:
    """Constructs gradient clipping + warmup-cosine decoupled `AdamW`."""
    opt_cfg = experiment.optimizer
    total_steps = max(1, opt_cfg.num_epochs * steps_per_epoch)
    warmup_steps = min(opt_cfg.warmup_epochs * steps_per_epoch, total_steps // 5)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=opt_cfg.init_lr,
        peak_value=opt_cfg.peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=opt_cfg.end_lr,
    )
    mask = wd_kernel_mask if opt_cfg.mask_kernels_only else None
    return optax.chain(
        optax.clip_by_global_norm(opt_cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=opt_cfg.weight_decay,
            mask=mask,
        ),
    )


def train_experiment(
    experiment: ExperimentConfig,
    dataset: PreparedStellaratorDataset,
    log_epochs: tuple[int, ...] = (1, 55, 110),
    verbose: bool = True,
) -> ExperimentRunResult:
    """Trains `experiment.num_seeds` CNNs sharded across devices via `jax.sharding`."""
    if experiment.data_split != dataset.config:
        raise ValueError(
            f"Experiment {experiment.name!r} expects data_split={experiment.data_split}, "
            f"but dataset was prepared with config={dataset.config}."
        )
    configure_jax_float32()
    mesh_ctx = create_ensemble_mesh(experiment.num_seeds)
    model = StellaratorPeriodicCNN(config=experiment.arch)

    tr_idx = dataset.split.train_idx
    va_idx = dataset.split.val_idx
    te_idx = dataset.split.test_idx

    batch_size = min(experiment.optimizer.batch_size, len(tr_idx))
    steps_per_epoch = max(1, len(tr_idx) // batch_size)
    optimizer = build_optimizer(experiment, steps_per_epoch)

    aug = experiment.augmentation
    use_bn = experiment.arch.norm_type == "batchnorm"
    d96 = D96ParityTransform.from_channel_stats(
        num_z=dataset.z_norm.shape[1],
        channel_names=dataset.channel_names,
        z_mean=dataset.z_mean,
        z_std=dataset.z_std,
    )

    z_tr_dev = jax.device_put(dataset.z_norm[tr_idx], mesh_ctx.replicated_sharding)
    s_tr_dev = jax.device_put(dataset.s_norm[tr_idx], mesh_ctx.replicated_sharding)
    y_tr_dev = jax.device_put(dataset.targets[tr_idx], mesh_ctx.replicated_sharding)
    z_va_dev = jax.device_put(dataset.z_norm[va_idx], mesh_ctx.replicated_sharding)
    s_va_dev = jax.device_put(dataset.s_norm[va_idx], mesh_ctx.replicated_sharding)
    y_va_dev = jax.device_put(dataset.targets[va_idx], mesh_ctx.replicated_sharding)

    def init_single(rng: jax.Array) -> dict[str, Any]:
        r_p, r_dp = jax.random.split(rng)
        variables = model.init(
            {"params": r_p, "droppath": r_dp},
            jnp.ones(
                (1, dataset.z_norm.shape[1], dataset.z_norm.shape[2]),
                dtype=jnp.float32,
            ),
            jnp.ones((1, dataset.s_norm.shape[1]), dtype=jnp.float32),
            train=True,
        )
        params = variables["params"]
        batch_stats = variables["batch_stats"] if use_bn else {}
        return {
            "params": params,
            "batch_stats": batch_stats,
            "opt_state": optimizer.init(params),
            "best_params": params,
            "best_batch_stats": batch_stats,
            "best_val_mse": jnp.array(jnp.inf, dtype=jnp.float32),
        }

    def epoch_and_val_single(
        state: dict[str, Any],
        epoch_rng: jax.Array,
        z_tr: jnp.ndarray,
        s_tr: jnp.ndarray,
        y_tr: jnp.ndarray,
        z_va: jnp.ndarray,
        s_va: jnp.ndarray,
        y_va: jnp.ndarray,
    ) -> tuple[dict[str, Any], tuple[jnp.ndarray, jnp.ndarray]]:
        rng_perm, rng_steps = jax.random.split(epoch_rng)
        perm = jax.random.permutation(rng_perm, z_tr.shape[0])[: steps_per_epoch * batch_size]
        batch_idx = perm.reshape((steps_per_epoch, batch_size))
        step_rngs = jax.random.split(rng_steps, steps_per_epoch)

        def scan_step(
            carry: tuple[Any, Any, Any], step_in: tuple[jnp.ndarray, jax.Array]
        ) -> tuple[tuple[Any, Any, Any], jnp.ndarray]:
            params, batch_stats, opt_state = carry
            idx, s_rng = step_in
            r_flip, r_shift, r_noise, r_dp = jax.random.split(s_rng, 4)
            bz = z_tr[idx]
            if aug.use_d96_parity:
                bz = d96.random_reflect_jax(r_flip, bz)
            if aug.use_z96_shift:
                bz = random_cyclic_shift(r_shift, bz)
            if aug.input_noise_std > 0.0:
                bz = bz + aug.input_noise_std * jax.random.normal(
                    r_noise, shape=bz.shape, dtype=bz.dtype
                )
            bs = s_tr[idx]
            by = y_tr[idx]

            if use_bn:

                def loss_fn_bn(p: Any) -> tuple[jnp.ndarray, Any]:
                    preds, mutated = model.apply(
                        {"params": p, "batch_stats": batch_stats},
                        bz,
                        bs,
                        train=True,
                        mutable=["batch_stats"],
                        rngs={"droppath": r_dp},
                    )
                    return jnp.mean((preds - by) ** 2), mutated["batch_stats"]

                (step_mse, next_bs), grads = jax.value_and_grad(loss_fn_bn, has_aux=True)(params)
            else:

                def loss_fn_ln(p: Any) -> jnp.ndarray:
                    preds = model.apply({"params": p}, bz, bs, train=True, rngs={"droppath": r_dp})
                    return jnp.mean((preds - by) ** 2)

                step_mse, grads = jax.value_and_grad(loss_fn_ln)(params)
                next_bs = batch_stats

            updates, next_opt = optimizer.update(grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            return (next_params, next_bs, next_opt), step_mse

        (new_params, new_bs, new_opt), step_mses = jax.lax.scan(
            scan_step,
            (state["params"], state["batch_stats"], state["opt_state"]),
            (batch_idx, step_rngs),
        )
        train_mse = jnp.mean(step_mses)
        val_preds = eval_batched_single(
            model,
            new_params,
            new_bs,
            z_va,
            s_va,
            chunk_size=min(1024, max(1, z_va.shape[0])),
            num_shifts=1,
        )
        val_mse = jnp.mean((val_preds - y_va) ** 2)

        improved = val_mse < state["best_val_mse"]
        best_params = jax.tree_util.tree_map(
            lambda new_v, old_v: jnp.where(improved, new_v, old_v),
            new_params,
            state["best_params"],
        )
        best_bs = jax.tree_util.tree_map(
            lambda new_v, old_v: jnp.where(improved, new_v, old_v),
            new_bs,
            state["best_batch_stats"],
        )
        best_val_mse = jnp.where(improved, val_mse, state["best_val_mse"])

        return {
            "params": new_params,
            "batch_stats": new_bs,
            "opt_state": new_opt,
            "best_params": best_params,
            "best_batch_stats": best_bs,
            "best_val_mse": best_val_mse,
        }, (train_mse, val_mse)

    init_jit = jax.jit(
        jax.vmap(init_single),
        in_shardings=(mesh_ctx.ensemble_sharding,),
        out_shardings=mesh_ctx.ensemble_sharding,
    )
    epoch_jit = jax.jit(
        jax.vmap(
            epoch_and_val_single,
            in_axes=(0, 0, None, None, None, None, None, None),
        ),
        in_shardings=(
            mesh_ctx.ensemble_sharding,
            mesh_ctx.ensemble_sharding,
            mesh_ctx.replicated_sharding,
            mesh_ctx.replicated_sharding,
            mesh_ctx.replicated_sharding,
            mesh_ctx.replicated_sharding,
            mesh_ctx.replicated_sharding,
            mesh_ctx.replicated_sharding,
        ),
        out_shardings=(
            mesh_ctx.ensemble_sharding,
            (mesh_ctx.ensemble_sharding, mesh_ctx.ensemble_sharding),
        ),
    )

    master_rng = jax.random.PRNGKey(experiment.base_seed)
    state = init_jit(
        jax.device_put(
            jax.random.split(master_rng, experiment.num_seeds),
            mesh_ctx.ensemble_sharding,
        )
    )

    y_va_np = dataset.targets[va_idx]
    y_te_np = dataset.targets[te_idx]
    val_var = float(np.var(y_va_np))
    t0 = time.time()
    final_tr_mse = 0.0
    final_va_mse = 0.0

    for epoch in range(experiment.optimizer.num_epochs):
        ep_rngs = jax.device_put(
            jax.random.split(
                jax.random.fold_in(master_rng, epoch + 10_000),
                experiment.num_seeds,
            ),
            mesh_ctx.ensemble_sharding,
        )
        state, (tr_mse, va_mse) = epoch_jit(
            state,
            ep_rngs,
            z_tr_dev,
            s_tr_dev,
            y_tr_dev,
            z_va_dev,
            s_va_dev,
            y_va_dev,
        )
        final_tr_mse = float(np.mean(np.asarray(tr_mse)))
        final_va_mse = float(np.mean(np.asarray(va_mse)))
        if verbose and (
            (epoch + 1) in log_epochs or (epoch + 1) == experiment.optimizer.num_epochs
        ):
            best_r2 = 1.0 - float(np.mean(np.asarray(state["best_val_mse"]))) / val_var
            print(
                f"  [{experiment.name:32s}] Ep {epoch + 1:3d}/"
                f"{experiment.optimizer.num_epochs} | "
                f"tr_mse={final_tr_mse:.5f} | va_mse={final_va_mse:.5f} "
                f"(ratio={final_va_mse / max(final_tr_mse, 1e-12):.2f}x) | "
                f"best_val_R2={best_r2:.5f} ({time.time() - t0:.1f}s)"
            )

    bp = state["best_params"]
    bbs = state["best_batch_stats"] if use_bn else {}
    val_preds = eval_ensemble_sharded(
        model=model,
        sharded_params=bp,
        sharded_batch_stats=bbs,
        z_np=dataset.z_norm[va_idx],
        s_np=dataset.s_norm[va_idx],
        mesh_ctx=mesh_ctx,
        num_shifts=aug.eval_num_shifts,
        chunk_size=aug.eval_chunk_size,
        d96_transform=d96,
        eval_d96_reflection=aug.eval_d96_reflection,
    )
    test_preds = eval_ensemble_sharded(
        model=model,
        sharded_params=bp,
        sharded_batch_stats=bbs,
        z_np=dataset.z_norm[te_idx],
        s_np=dataset.s_norm[te_idx],
        mesh_ctx=mesh_ctx,
        num_shifts=aug.eval_num_shifts,
        chunk_size=aug.eval_chunk_size,
        d96_transform=d96,
        eval_d96_reflection=aug.eval_d96_reflection,
    )

    indiv_r2s = [compute_r2(y_te_np, test_preds[i]) for i in range(experiment.num_seeds)]
    ens_val_pred = np.mean(val_preds, axis=0)
    ens_test_pred = np.mean(test_preds, axis=0)
    uns_te = dataset.unstable_mask[te_idx]
    unstable_r2 = (
        compute_r2(y_te_np[uns_te], ens_test_pred[uns_te]) if np.any(uns_te) else float("nan")
    )
    per_cls = compute_per_class_r2(
        y_te_np,
        ens_test_pred,
        dataset.equilibrium_class[te_idx],
        dataset.equilibrium_descriptions,
    )
    n_params = sum(int(np.prod(leaf.shape[1:])) for leaf in jax.tree_util.tree_leaves(bp))
    wall_sec = time.time() - t0

    return ExperimentRunResult(
        experiment=experiment,
        n_params=n_params,
        final_train_mse=final_tr_mse,
        final_val_mse=final_va_mse,
        val_over_train_mse_ratio=final_va_mse / max(final_tr_mse, 1e-12),
        val_predictions=val_preds,
        test_predictions=test_preds,
        indiv_test_r2_mean=float(np.mean(indiv_r2s)),
        indiv_test_r2_std=float(np.std(indiv_r2s)),
        ens_val_r2=compute_r2(y_va_np, ens_val_pred),
        ens_test_r2=compute_r2(y_te_np, ens_test_pred),
        ens_unstable_test_r2=unstable_r2,
        ens_test_rmse=compute_rmse(y_te_np, ens_test_pred),
        per_class_test_r2=per_cls,
        best_params=jax.device_get(bp),
        best_batch_stats=jax.device_get(bbs) if use_bn else {},
        wall_time_sec=wall_sec,
    )


def run_sweep(
    experiments: Sequence[ExperimentConfig],
    dataset: PreparedStellaratorDataset,
    verbose: bool = True,
) -> SweepRunSummary:
    """Executes a sequence of `ExperimentConfig`s and compares to leading baseline."""
    runs: list[ExperimentRunResult] = []
    for i, exp in enumerate(experiments):
        if verbose:
            print(f"\n=== Sweep Run {i + 1}/{len(experiments)}: {exp.name} ===")
        runs.append(train_experiment(exp, dataset, verbose=verbose))

    all_val = np.concatenate([r.val_predictions for r in runs], axis=0)
    all_test = np.concatenate([r.test_predictions for r in runs], axis=0)
    te_idx = dataset.split.test_idx
    va_idx = dataset.split.val_idx

    topk = select_validation_topk_ensemble(
        val_member_preds=all_val,
        test_member_preds=all_test,
        y_val=dataset.targets[va_idx],
        y_test=dataset.targets[te_idx],
        q_test_true=dataset.q_varied[te_idx],
        unstable_test_mask=dataset.unstable_mask[te_idx],
        test_equilibrium_class=dataset.equilibrium_class[te_idx],
        equilibrium_descriptions=dataset.equilibrium_descriptions,
        target_mode=dataset.config.target_mode,
    )

    if verbose:
        ref_lead = (
            LEADING_EXPERIMENT.reference_benchmark.ens8_test_r2
            if LEADING_EXPERIMENT.reference_benchmark is not None
            else float("nan")
        )
        print(
            "\n=== Unseen-Stellarator Sweep Summary "
            f"(vs. Leading Reference R^2 = {ref_lead:.5f}) ==="
        )
        hdr = (
            f"{'Experiment':34s} | {'Params':>7s} | {'Val/Tr':>8s} | "
            f"{'Indiv R2':>10s} | {'Ens Test R2':>11s} | {'vs Leading':>10s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in runs:
            print(
                f"{r.experiment.name:34s} | {r.n_params:7,d} | "
                f"{r.val_over_train_mse_ratio:7.2f}x | "
                f"{r.indiv_test_r2_mean:10.5f} | {r.ens_test_r2:11.5f} | "
                f"{r.ens_test_r2 - ref_lead:+10.5f}"
            )
        print(
            f"Validation-Selected Top-{topk.best_k}/{topk.total_candidates} "
            f"Ensemble -> Test R^2 = {topk.test_r2:.5f} | "
            f"Unstable R^2 = {topk.unstable_test_r2:.5f} | "
            f"RMSE = {topk.test_rmse:.4f}"
        )

    return SweepRunSummary(runs=tuple(runs), topk_ensemble=topk)
