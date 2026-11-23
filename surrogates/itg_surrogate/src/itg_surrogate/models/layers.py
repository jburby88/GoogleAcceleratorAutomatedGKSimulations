"""Periodic 1D convolutional layers, normalization, and Stochastic Depth."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from itg_surrogate.registry.specs import ActivationName, NormType


def get_activation(name: ActivationName) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Resolves activation name to a JAX/Flax pointwise nonlinearity."""
    if name == "relu":
        return nn.relu
    if name == "gelu":
        return nn.gelu
    if name == "silu":
        return nn.silu
    raise ValueError(f"Unsupported activation: {name!r}")


def circular_conv1d(
    inputs: jnp.ndarray,
    features: int,
    kernel_size: int,
    name: str,
    kernel_init: Any = None,
) -> jnp.ndarray:
    """1D convolution with exact periodic wrap-around along `axis=1` (`z`)."""
    if kernel_init is None:
        kernel_init = nn.initializers.lecun_normal()
    length = inputs.shape[1]
    pad_total = kernel_size - 1
    if pad_total > 0:
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        wrap_idx = np.arange(-pad_left, length + pad_right) % length
        padded = inputs[:, wrap_idx, :]
    else:
        padded = inputs
    return nn.Conv(
        features=features,
        kernel_size=(kernel_size,),
        padding="VALID",
        kernel_init=kernel_init,
        name=name,
    )(padded)


def apply_feature_norm(
    inputs: jnp.ndarray,
    norm_type: NormType,
    train: bool,
    name: str,
) -> jnp.ndarray:
    """Applies per-sample `LayerNorm` (default) or cross-batch `BatchNorm`."""
    if norm_type == "layernorm":
        return nn.LayerNorm(epsilon=1e-5, name=name)(inputs)
    if norm_type == "batchnorm":
        return nn.BatchNorm(
            use_running_average=not train,
            momentum=0.95,
            epsilon=1e-5,
            name=name,
        )(inputs)
    raise ValueError(f"Unsupported norm_type: {norm_type!r}")


def apply_drop_path(
    residual: jnp.ndarray,
    drop_rate: float,
    train: bool,
    make_rng_fn: Callable[[str], jax.Array],
) -> jnp.ndarray:
    """Applies per-sample Stochastic Depth (`DropPath`) to a residual branch."""
    if drop_rate <= 0.0 or not train:
        return residual
    keep_prob = 1.0 - drop_rate
    rng = make_rng_fn("droppath")
    mask_shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
    keep_mask = jax.random.bernoulli(rng, p=keep_prob, shape=mask_shape)
    return residual * (keep_mask.astype(residual.dtype) / keep_prob)
