# Cloud & Compute Cost Estimate (`COST_ESTIMATE.md`)

## 1. Control Plane & Steering (`aurora-steer` + Dashboard + BigQuery)

| Service | Sizing | Estimated Campaign Cost |
| :--- | :--- | :--- |
| **Cloud Run (`aurora-steer`)** | `min_instances=0`, `max_instances=1`, ~4,000 `/triage` + ~80,000 `/v1/events` batches | **< \$8** |
| **Vertex AI (`MODEL_ID`, default `gemini-4-argon`, `global`)** | `<= 1` call/failure (hard cap: 2 automated strikes and 6 LLM calls per run per 24 h) × ~800 failures × ~3k input tokens (runbooks + `<2 KB` summary) + a short JSON answer plus the model's reasoning tokens (billed as output) | **< \$80** at the announced standard rate of \$4 / \$20 per 1M input / output tokens (introductory \$2 / \$10 until 2026-11-30); setting `MODEL_ID=gemini-3.8-flash` costs considerably less. Re-check the Vertex AI pricing page before each campaign. |
| **Cloud Run (`dashboard` + Direct IAP)** | `min_instances=0`, native Cloud Run IAP (no \$18/mo Classic LB) | **< \$5** |
| **Cloud Run Job + Scheduler (`gcp-gx-babysitter`)** | 2-min tick during active GCP Batch campaigns (~3s per tick) | **< \$6** |
| **BigQuery + Cloud Logging** | < 500 MB total telemetry storage & view queries | **\$0 (Free Tier)** |
| **Control-Plane Subtotal** | | **< \$100 total per campaign** (< \$35 with `MODEL_ID=gemini-3.8-flash`) |

---

## 2. Backend 2: Containerized GX on GCP Cloud Batch (~3,000 GPU-Hour Grant Allocation)

Why **Cloud Batch + Spot NVIDIA L4** (`g2-standard-8`) is the primary GCP GX target:
- **GX is FP32 pseudo-spectral**: NVIDIA L4 delivers ~30 TFLOPS FP32 with 24 GB GDDR6 VRAM.
- **First-class Spot VMs (~60–75% discount)**:
  - Spot `g2-standard-8` (1× L4): ~\$0.28–\$0.35 / hr -> **3,000 GPU-hours = ~\$850–\$1,050** (stretching GCP credits ~3× vs on-demand or Cloud Run Jobs).
  - Spot `a2-highgpu-1g` (1× A100 40 GB fallback for kinetic-electron high-resolution runs > 20 GB VRAM): ~\$1.15–\$1.35 / hr, used selectively (~10–15% of runs) via `gpu_selector.py`.
- **Zero idle cluster fee**: Unlike GKE Standard/Autopilot, Cloud Batch has **\$0/month** cluster management overhead and scales cleanly to zero between campaigns.
