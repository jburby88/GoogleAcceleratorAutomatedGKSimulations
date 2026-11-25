"""Exact Z_96 cyclic-shift and Dihedral D_96 gyrokinetic parity symmetries."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

PSEUDOSCALAR_Z_CHANNELS: frozenset[str] = frozenset({"gbdrift0", "gds21"})


def random_cyclic_shift(rng: jax.Array, z_batch: jnp.ndarray) -> jnp.ndarray:
    """Applies an independent uniform random `Z_{N_z}` cyclic shift per sample."""
    batch_size, num_z, _ = z_batch.shape
    shifts = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=num_z)
    return jax.vmap(lambda sample, s: jnp.roll(sample, shift=s, axis=0))(z_batch, shifts)


@dataclasses.dataclass(frozen=True)
class D96ParityTransform:
    """Affine representation of exact gyrokinetic reflection in normalized space.

    Under `(x, y, z, v_parallel, phi, h) -> (-x, +y, -z, -v_parallel, -phi, -h)`,
    every term in the nonlinear electrostatic flux-tube gyrokinetic Vlasov-Poisson
    system flips sign while the turbulent heat flux `Q` is invariant. Along the
    periodic grid `z_k` (`k = 0..N_z-1`, `z_0 = -L/2`), `z_k -> -z_k` corresponds
    to index reversal `(-k) mod N_z` combined with `-1` parity on the radial-shear
    channels `gbdrift0` and `gds21` (which are linear in `grad x`).

    Because `z_norm = (z_raw - z_mean) / z_std`, negating a physical channel `c`
    with mean `mu_c` maps `z_norm[..., c] -> -z_norm[..., c] - 2 * mu_c / sigma_c`,
    making `reflect_jax` and `reflect_numpy` exact involutions to machine epsilon.
    """

    rev_z_idx: np.ndarray
    parity_signs: np.ndarray
    parity_offset: np.ndarray

    @classmethod
    def from_channel_stats(
        cls,
        num_z: int,
        channel_names: tuple[str, ...],
        z_mean: np.ndarray,
        z_std: np.ndarray,
    ) -> D96ParityTransform:
        rev_z_idx = ((-np.arange(num_z)) % num_z).astype(np.int32)
        signs = np.array(
            [-1.0 if ch in PSEUDOSCALAR_Z_CHANNELS else 1.0 for ch in channel_names],
            dtype=np.float32,
        ).reshape((1, 1, -1))
        offset = ((signs - 1.0) * (z_mean / z_std)).astype(np.float32)
        return cls(rev_z_idx=rev_z_idx, parity_signs=signs, parity_offset=offset)

    def reflect_numpy(self, z_norm: np.ndarray) -> np.ndarray:
        """Applies exact `D_96` reflection to a NumPy batch `(B, N_z, C)`."""
        return (z_norm[:, self.rev_z_idx, :] * self.parity_signs + self.parity_offset).astype(
            np.float32
        )

    def reflect_jax(self, z_batch: jnp.ndarray) -> jnp.ndarray:
        """Applies exact `D_96` reflection to a JAX batch `(B, N_z, C)`."""
        rev_idx = jnp.asarray(self.rev_z_idx, dtype=jnp.int32)
        signs = jnp.asarray(self.parity_signs, dtype=z_batch.dtype)
        offset = jnp.asarray(self.parity_offset, dtype=z_batch.dtype)
        return z_batch[:, rev_idx, :] * signs + offset

    def random_reflect_jax(self, rng: jax.Array, z_batch: jnp.ndarray) -> jnp.ndarray:
        """Applies exact `D_96` reflection independently per sample with `p = 0.5`."""
        reflected = self.reflect_jax(z_batch)
        do_flip = jax.random.bernoulli(rng, p=0.5, shape=(z_batch.shape[0], 1, 1))
        return jnp.where(do_flip, reflected, z_batch)
