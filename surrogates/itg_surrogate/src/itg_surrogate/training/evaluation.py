"""Chunked GSPMD-sharded inference with exact Z_96 & D_96 coset averaging."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from itg_surrogate.models.cnn import StellaratorPeriodicCNN
from itg_surrogate.models.symmetry import D96ParityTransform
from itg_surrogate.training.sharding import EnsembleMeshContext


def eval_batched_single(
    model: StellaratorPeriodicCNN,
    params: Any,
    batch_stats: Any,
    z_data: jnp.ndarray,
    s_data: jnp.ndarray,
    chunk_size: int = 512,
    num_shifts: int = 32,
) -> jnp.ndarray:
    """Evaluates a single CNN in fixed-size chunks with `Z_96` coset averaging.

    Because five `MaxPool1D(stride=2)` stages downsample `96 -> 3` (`2^5 = 32`),
    shifting the input by `32` shifts the pooled `3`-bin representation by `1`
    before symmetric spatial pooling. Averaging over `num_shifts = 32` consecutive
    integer shifts `s in {0, ..., 31}` therefore yields exact invariance across
    all `96` elements of `Z_96`.
    """
    n_samples = z_data.shape[0]
    eff_chunk = min(chunk_size, max(1, n_samples))
    pad_len = (-n_samples) % eff_chunk
    if pad_len > 0:
        z_padded = jnp.pad(z_data, ((0, pad_len), (0, 0), (0, 0)))
        s_padded = jnp.pad(s_data, ((0, pad_len), (0, 0)))
    else:
        z_padded = z_data
        s_padded = s_data

    num_chunks = z_padded.shape[0] // eff_chunk
    z_chunks = z_padded.reshape((num_chunks, eff_chunk, z_data.shape[1], z_data.shape[2]))
    s_chunks = s_padded.reshape((num_chunks, eff_chunk, s_data.shape[1]))

    variables: dict[str, Any] = {"params": params}
    if batch_stats:
        variables["batch_stats"] = batch_stats

    def eval_chunk(chunk_inputs: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        zc, sc = chunk_inputs
        if num_shifts <= 1:
            return model.apply(variables, zc, sc, train=False)

        def pred_shift(shift: jnp.ndarray) -> jnp.ndarray:
            return model.apply(variables, jnp.roll(zc, shift=shift, axis=1), sc, train=False)

        return jnp.mean(jax.lax.map(pred_shift, jnp.arange(num_shifts)), axis=0)

    preds_chunks = jax.lax.map(eval_chunk, (z_chunks, s_chunks))
    return preds_chunks.reshape((-1,))[:n_samples]


def eval_ensemble_sharded(
    model: StellaratorPeriodicCNN,
    sharded_params: Any,
    sharded_batch_stats: Any,
    z_np: np.ndarray,
    s_np: np.ndarray,
    mesh_ctx: EnsembleMeshContext,
    num_shifts: int = 32,
    chunk_size: int = 512,
    d96_transform: D96ParityTransform | None = None,
    eval_d96_reflection: bool = False,
) -> np.ndarray:
    """Evaluates `M` sharded CNNs on `(z_np, s_np)` returning `(M, N)` predictions."""
    use_bn = bool(sharded_batch_stats)

    if use_bn:
        eval_jit = jax.jit(
            jax.vmap(
                lambda p, bs, zd, sd: eval_batched_single(
                    model,
                    p,
                    bs,
                    zd,
                    sd,
                    chunk_size=chunk_size,
                    num_shifts=num_shifts,
                ),
                in_axes=(0, 0, None, None),
            ),
            in_shardings=(
                mesh_ctx.ensemble_sharding,
                mesh_ctx.ensemble_sharding,
                mesh_ctx.replicated_sharding,
                mesh_ctx.replicated_sharding,
            ),
            out_shardings=mesh_ctx.ensemble_sharding,
        )
    else:
        eval_jit = jax.jit(
            jax.vmap(
                lambda p, zd, sd: eval_batched_single(
                    model,
                    p,
                    {},
                    zd,
                    sd,
                    chunk_size=chunk_size,
                    num_shifts=num_shifts,
                ),
                in_axes=(0, None, None),
            ),
            in_shardings=(
                mesh_ctx.ensemble_sharding,
                mesh_ctx.replicated_sharding,
                mesh_ctx.replicated_sharding,
            ),
            out_shardings=mesh_ctx.ensemble_sharding,
        )

    z_dev = jax.device_put(z_np, mesh_ctx.replicated_sharding)
    s_dev = jax.device_put(s_np, mesh_ctx.replicated_sharding)
    if use_bn:
        preds = np.asarray(eval_jit(sharded_params, sharded_batch_stats, z_dev, s_dev))
    else:
        preds = np.asarray(eval_jit(sharded_params, z_dev, s_dev))

    if eval_d96_reflection:
        if d96_transform is None:
            raise ValueError("d96_transform must be provided when eval_d96_reflection=True.")
        z_refl_dev = jax.device_put(d96_transform.reflect_numpy(z_np), mesh_ctx.replicated_sharding)
        if use_bn:
            preds_refl = np.asarray(
                eval_jit(sharded_params, sharded_batch_stats, z_refl_dev, s_dev)
            )
        else:
            preds_refl = np.asarray(eval_jit(sharded_params, z_refl_dev, s_dev))
        preds = 0.5 * (preds + preds_refl)

    return preds.astype(np.float32)
