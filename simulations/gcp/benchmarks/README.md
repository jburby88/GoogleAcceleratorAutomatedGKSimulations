# GX GCP Benchmarks

These directories contain real nonlinear adiabatic-electron ITG Cyclone Base Case input decks for testing GX on GCP GPUs.

## Contents

- `cyclone_itg_unstabilized`: Configured with an initially coarse grid (`nx = 32`, `ny = 16`) and low hyperviscosity (`D_hyper = 0.01`). This will exhibit high-$k$ spectral pile-up or timestep collapse. `aurora-steer` + `gcp-gx-babysitter` will autonomously patch it and attempt to converge it on Strike 2.
- `cyclone_itg_converged`: A well-resolved baseline input deck (`nx = 64`, `ny = 32`, `ntheta = 32`, `D_hyper = 0.05`). This runs stably on GCP.

## How to Run

Copy these benchmark folders into the `gs://<bucket>/queue/pending/` directory of your GCS bucket. The GCP Cloud Batch executor will pick them up, process `gx.in` and `run_meta.json`, and run them on GCP GPUs. Output NetCDF files (`.out.nc` and `.restart.nc`) will be uploaded back to GCS.
