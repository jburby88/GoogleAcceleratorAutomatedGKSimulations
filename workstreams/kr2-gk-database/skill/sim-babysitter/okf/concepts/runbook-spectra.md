---
id: runbook-spectra
triggers: ["SPECTRAL_PILEUP"]
---

# Runbook: High-$k$ Spectral Pile-Up (`SPECTRAL_PILEUP`)

<!-- TODO(david): confirm runbook procedures -->

**Allowed actions (enforced by aurora-steer, `manifest.yaml`):** `cold_restart` or `escalate_to_human`.

## Symptoms
- Electrostatic or electromagnetic fluctuation energy piles up at the highest resolved $k_x$ or $k_y$ modes (`high_k_tail_ratio > 0.25` computed from NetCDF `.out.nc` spectra (`[Diagnostics] fluxes = true`)) instead of decaying smoothly across the dissipation range.

## Recovery Action (`cold_restart`)
1. Set `action = "cold_restart"` and `restart_from_checkpoint = false` (changing `nx`/`ny`/`nz` invalidates existing checkpoints).
2. Increase `nx` and/or `ny` by ~1.5× (staying strictly within `nx <= 512`, `ny <= 256`).
3. Optionally increase `hyperdiffusion_coeff` by 1.25–1.5× (staying `<= 2.0`) to damp unresolved sub-grid cascade. A dissipation change is only ever applied on a cold restart, and every patch is recorded in the run's `parameter_patches` provenance.
4. Note: On GCP Cloud Batch (`backend = "gcp_batch"`), increasing `nx`/`ny`/`nz` automatically triggers the executor's VRAM estimator (`gpu_selector.py`) to switch from Spot NVIDIA L4 (24 GB) to Spot A100 (40 GB) if estimated memory exceeds 20 GB.

---

## Questions for Prof. David Hatch to Confirm in This Runbook

1. **Spectral Diagnostic Metrics (`scalar_metrics`):**
   - Which specific numbers from your Python diagnostic script quantify spectral pile-up in $k_x$, $k_y$, $z$, or velocity space $(v_\parallel, \mu)$? (e.g., ratio of fluctuation amplitude in the top 10% of $k_x$ or $k_y$ modes vs. peak mode, currently placeholder `high_k_tail_ratio > 0.25` computed from NetCDF `.out.nc` spectra (`[Diagnostics] fluxes = true`)).
2. **Directional Resolution vs. Hyper-Diffusivity Decision Tree:**
   - How should the agent distinguish between:
     - **Radial ($k_x$) pile-up** -> increase `nx` (`box.nx0` / `Dimensions.nx`) or `general.hyp_x`?
     - **Binormal ($k_y$) pile-up** -> increase `ny` (`box.nky0` / `Dimensions.ny`) or `general.hyp_y`?
     - **Parallel ($z / \theta$) structure under-resolution** -> increase `nz` (`box.nz0` / `Dimensions.ntheta`) or `general.hyp_z` (`hyperdiffusion_coeff`; note that in GX this parameter maps to the perpendicular `Dissipation.D_hyper`)?
3. **FFT Grid Multiples:**
   - Do GENE and GX require `nx`, `ny`, `nz` to be multiples of powers of 2 / 2-3-5 radix sizes for cuFFT/FFTW efficiency (e.g., stepping `64 -> 96 -> 128 -> 192 -> 256`)?
