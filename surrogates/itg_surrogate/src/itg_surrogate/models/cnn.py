"""Unified periodic 1D CNN / ResFiLM surrogate architecture in Flax Linen."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from itg_surrogate.models.layers import (
    apply_drop_path,
    apply_feature_norm,
    circular_conv1d,
    get_activation,
)
from itg_surrogate.registry.specs import ArchConfig


class StellaratorPeriodicCNN(nn.Module):
    """`Z_96`-equivariant periodic 1D CNN with optional Residual, FiLM & MultiPool.

    Supports the full ablation hierarchy via `ArchConfig`:
      - `Table1-GELU-32` & `Pyramid-*`: `use_residual=False, use_film=False,
        use_multistat_pool=False`
      - `+ MultiStatPool`: symmetric `[mean, max, std]` reduction over the 3
        coarse spatial bins remaining after 5 `MaxPool(2)` stages (`96 / 2^5 = 3`)
      - `+ PreLN-Residual`: zero-initialized periodic conv & FC residual branches
      - `+ EarlyFiLM`: identity-initialized `(1 + gamma) * x + beta` modulation of
        normalized conv activations conditioned on `(a/LT, a/Ln)` at all 5 scales
      - `ResFiLM-MultiPool`: all three structural enhancements combined, with
        optional depth-scheduled `drop_path_rate` on residual branches.
    """

    config: ArchConfig

    @nn.compact
    def __call__(
        self,
        z_profiles: jnp.ndarray,
        scalar_grads: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        cfg = self.config
        act_fn = get_activation(cfg.activation)
        total_blocks = float(len(cfg.conv_channels) + len(cfg.fc_dims))

        cond = None
        if cfg.use_film:
            cond = nn.Dense(cfg.film_hidden_dim, name="film_embed")(scalar_grads)
            cond = apply_feature_norm(cond, cfg.norm_type, train, name="film_norm")
            cond = act_fn(cond)

        x = z_profiles
        for i, (out_ch, ksize) in enumerate(zip(cfg.conv_channels, cfg.kernel_sizes, strict=True)):
            x = circular_conv1d(x, features=out_ch, kernel_size=ksize, name=f"conv_{i}")
            x = apply_feature_norm(x, cfg.norm_type, train, name=f"norm_conv_{i}")
            if cond is not None:
                gb = nn.Dense(
                    2 * out_ch,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                    name=f"film_{i}",
                )(cond)
                gamma, beta = jnp.split(gb[:, None, :], 2, axis=-1)
                x = x * (1.0 + gamma) + beta
            x = act_fn(x)

            if cfg.use_residual:
                r = circular_conv1d(
                    x,
                    features=out_ch,
                    kernel_size=max(3, ksize - 2),
                    name=f"res_c1_{i}",
                )
                r = apply_feature_norm(r, cfg.norm_type, train, name=f"res_norm_{i}")
                r = act_fn(r)
                r = circular_conv1d(
                    r,
                    features=out_ch,
                    kernel_size=3,
                    kernel_init=nn.initializers.zeros_init(),
                    name=f"res_c2_{i}",
                )
                rate_i = cfg.drop_path_rate * float(i + 1) / total_blocks
                x = x + apply_drop_path(r, rate_i, train, self.make_rng)

            x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding="VALID")

        # Symmetric pooling over the 3 coarse spatial bins (invariant to shifts of 32)
        if cfg.use_multistat_pool:
            x_mean = jnp.mean(x, axis=1)
            x_max = jnp.max(x, axis=1)
            x_std = jnp.sqrt(jnp.var(x, axis=1) + 1e-6)
            x = jnp.concatenate([x_mean, x_max, x_std], axis=-1)
        else:
            x = jnp.mean(x, axis=1)

        x = jnp.concatenate([x, scalar_grads], axis=-1)
        for j, fc_dim in enumerate(cfg.fc_dims):
            x = nn.Dense(fc_dim, name=f"fc_{j}")(x)
            x = apply_feature_norm(x, cfg.norm_type, train, name=f"norm_fc_{j}")
            x = act_fn(x)
            if cfg.use_residual:
                rf = nn.Dense(fc_dim, name=f"fc_res1_{j}")(x)
                rf = apply_feature_norm(rf, cfg.norm_type, train, name=f"fc_res_norm_{j}")
                rf = act_fn(rf)
                rf = nn.Dense(
                    fc_dim,
                    kernel_init=nn.initializers.zeros_init(),
                    name=f"fc_res2_{j}",
                )(rf)
                rate_j = cfg.drop_path_rate * float(len(cfg.conv_channels) + j + 1) / total_blocks
                x = x + apply_drop_path(rf, rate_j, train, self.make_rng)

        out = nn.Dense(1, name="head")(x)
        return jnp.squeeze(out, axis=-1)
