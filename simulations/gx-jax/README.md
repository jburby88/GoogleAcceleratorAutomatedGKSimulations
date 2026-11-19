# Stellarator GX-JAX (`simulations/gx-jax`)

JAX/XLA port of the **stellarator flux-tube gyrokinetic core** of [GX](https://bitbucket.org/gyrokinetics/gx.git) designed for **Cloud TPUs (`v5p` / `v6e` Trillium)**, NVIDIA GPUs, and CPUs, integrated with `simulations/gcp/` (`aurora-steer` / `gcp-gx-babysitter`).

## TPU Numerical Design & Capabilities

1. **TPU FP32 MXU Precision (`bfloat16` avoidance):**
   - Sets `jax.config.update("jax_default_matmul_precision", "highest")` so `jnp.einsum` and matrix multiplications run at full IEEE FP32 accuracy on the TPU MXU rather than `bfloat16`.
   - Enables `jax_enable_x64 = True` on the host during setup to precompute 1D geometry arrays and quasineutrality/Ampere field matrices in `float64` before casting runtime state `G` to `complex64`.
   - Verified at container build time via `python3 -m gx_jax._verify_fp32_hlo` (zero `f64`/`c128` device ops in FP32 stepping/diagnostics).
2. **TPU `(8, 128)` HBM Tile Layout & Sharding:**
   - Keeps spatial spectral axes `(nz, ky, kx)` in the trailing dimensions of `G[nspecies, nhermite, nlaguerre, nz, ky, kx]` and shards across `("sp", "m")` via `jax.shard_map` / `NamedSharding`.
3. **Stellarator Parallel Boundary Conditions:**
   - Supports both `GradParallelLinked` (`boundary = "linked"`, including periodic) and `GradParallelNTFT` (`nonTwist` for low-shear $\hat{s} \approx 0$ stellarators such as W7-X and QI/QA configurations), grouping variable-length $(k_x, z)$ chains into static padded buckets for XLA.
4. **Zonal-Flow Flux-Surface Average & Electromagnetic Physics:**
   - Weights $\langle \phi \rangle_\psi$ along $z$ by `jacobian(z)` on $k_y = 0$ for adiabatic-electron runs (`BOLTZMANN_ELECTRONS`), and supports kinetic electrons (`all_kinetic`) with $\delta A_\parallel$ (`fapar`) and $\delta B_\parallel$ (`fbpar`).
5. **Babysitter & NetCDF Compatibility:**
   - Reads standard C++ GX `.in` TOML decks and VMEC `wout_*.nc` / `.eik.nc` inputs, writes compatible `.out.nc` and `.restart.nc` files, and normalizes fluxes by $\langle |\nabla x| \rangle$ to match `gx-diagnose`.

## CLI Entry Points

- `gx-jax` / `gx`: single-deck solver (`gx_jax.cli:main`), supporting `--fft-backend {auto,fft,mxu}`, `--async-io`, `--use-scan`, and multi-host `jax.distributed` (`--distributed`, `--mesh`).
- `gx-jax-ensemble`: batched `jax.vmap` ensemble runner (`gx_jax.ensemble:main`) across a `("batch",)` TPU device mesh.
- `gx-jax-campaign`: Spot campaign orchestrator (`gx_jax.campaign:main`) with local scratch staging, atomic promotion to GCS FUSE, and preemption resume.

## Installation & Testing

```bash
pip install -e "./simulations/gx-jax[dev]"
# Fast core unit tests (also run as part of `make test` in CI):
pytest simulations/gx-jax/tests/test_grids_geo_laguerre.py simulations/gx-jax/tests/test_fields.py
# Full physics, multi-device sharding, golden NetCDF, and preemption suite:
make test-gx-jax
```

For Cloud TPU runtime dependencies:

```bash
pip install "jax[tpu]>=0.7.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Attribution

Derived from the mathematical formulation and MIT-licensed reference implementation of **GX** (Copyright (c) 2011-2022 Noah R. Mandell, William D. Dorland, and the GX team; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)):

- N. R. Mandell, W. D. Dorland, and M. Landreman (2018), *J. Plasma Phys.* **84**, 905840108.
- N. R. Mandell et al. (2024), *J. Plasma Phys.* **90**, 905900402.
