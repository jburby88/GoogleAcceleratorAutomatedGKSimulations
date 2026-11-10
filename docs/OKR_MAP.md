# OKR-to-Repository Map (`docs/OKR_MAP.md`)

**Main Objective:** *Accelerate transport-consistent magnetic fusion plasma optimization via automated gyrokinetic simulation and machine-learning surrogates.*

| Workstream / Layer | Team Leads | Target KPIs | Repository Path(s) | External Read-Only Links |
| :--- | :--- | :--- | :--- | :--- |
| **Shared Platform** | Edouard, Tan | Lightweight Python on TACC; `<2 KB` payloads; `<= 1` LLM call/failure; private IAP dashboard; KPIs count prod rows only | `platform/hpc-core/`, `platform/aurora-steer/`, `platform/telemetry/`, `platform/infra/` | — |
| **KR1: Multi-Fidelity Surrogates & Optimizer Orchestration** | Josh Burby, David Hatch, Alan Kaptanoglu, Tobias Blickhan, Nova | 3 coupled surrogates (`<5%` relative $L_2$ error) + Augmented Lagrangian outer loop | `surrogates/` | https://github.com/ToBlick/mrx/, https://arxiv.org/abs/2603.15240, https://journals.aps.org/prl/abstract/10.1103/n7gk-922h |
| **KR2: Automated Gyrokinetic Database (`sim-babysitter`)** | David Hatch, Edouard, Tan | `1,000–2,000` hands-free GX/GENE runs (`>=85%` hands-free rate) with manifest v2 provenance (campaign layout extending Zenodo `14867777`) | `simulations/{skill,hpc,gcp,docs}/` + `platform/dashboard/` | https://zenodo.org/records/14867777, https://gx.readthedocs.io/en/latest/ |
| **KR3: Transport-Consistent High-$\beta$ Stellarator Design** | Whole Team | `>3%` plasma $\beta$, `10x` faster design loop, `<5%` ground-truth validation error | `surrogates/kr3_design/` | `enqueue_validation_run` feeds `simulations/`; the active-learning loop itself is a Phase 3 item (deferred) |
