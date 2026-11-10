"""VRAM estimator & Spot GPU tier selector for containerized GX on GCP Cloud Batch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPUTierSpec:
    """Machine & accelerator specification for GCP Cloud Batch."""

    tier_name: str  # "l4", "t4", "a100", "g4", or "cpu"
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    vram_gb: float


T4_SPOT_SPEC = GPUTierSpec(
    tier_name="t4",
    machine_type="n1-standard-4",
    accelerator_type="nvidia-tesla-t4",
    accelerator_count=1,
    vram_gb=16.0,
)

L4_SPOT_SPEC = GPUTierSpec(
    tier_name="l4",
    machine_type="g2-standard-8",
    accelerator_type="nvidia-l4",
    accelerator_count=1,
    vram_gb=24.0,
)

A100_SPOT_SPEC = GPUTierSpec(
    tier_name="a100",
    machine_type="a2-highgpu-1g",
    accelerator_type="nvidia-tesla-a100",
    accelerator_count=1,
    vram_gb=40.0,
)

G4_SPOT_SPEC = GPUTierSpec(
    tier_name="g4",
    machine_type="g4-standard-48",
    accelerator_type="nvidia-rtx-pro-6000",
    accelerator_count=1,
    vram_gb=96.0,
)

CPU_DEMO_SPEC = GPUTierSpec(
    tier_name="cpu",
    machine_type="e2-standard-4",
    accelerator_type="",
    accelerator_count=0,
    vram_gb=16.0,
)

# Leave 2.5 GB / 4 GB CUDA/cuFFT workspace headroom below 16 GB T4 and 24 GB L4 ceilings
T4_SAFE_VRAM_CEILING_GB = 13.5
L4_SAFE_VRAM_CEILING_GB = 20.0

DEFAULT_CANDIDATE_REGIONS: tuple[str, ...] = (
    "us-central1",
    "us-east4",
    "us-west1",
    "us-east1",
    "us-west4",
)


def estimate_gx_vram_gb(parameters: dict[str, Any] | None = None) -> float:
    """Estimate GX FP32 pseudo-spectral GPU memory footprint (in GiB) including RK4 + cuFFT buffers.

    TODO(david): calibrate empirical cuFFT buffer multiplier against high-beta kinetic-electron GX runs.
    """
    params = parameters or {}
    nx = int(params.get("nx", 64))
    ny = int(params.get("ny", 32))
    nz = int(params.get("nz", params.get("ntheta", 48)))
    nhermite = int(params.get("nhermite", 16))
    nlaguerre = int(params.get("nlaguerre", 8))
    nspecies = int(params.get("nspecies", 2))

    # Complex64 (8 bytes) state array * ~18x for RK stages, fields, dealiasing (3/2)^2, and cuFFT plans
    dof = nx * ny * nz * nhermite * nlaguerre * nspecies
    state_bytes = dof * 8 * 18
    cuda_base_bytes = int(1.5 * (1024**3))
    return round((state_bytes + cuda_base_bytes) / (1024**3), 3)


def select_gpu_tier(
    parameters: dict[str, Any] | None = None,
    *,
    force_high_memory: bool = False,
    prefer_g4: bool = False,
    cpu_only: bool = False,
) -> GPUTierSpec:
    """Select Spot NVIDIA L4 (`g2-standard-8`) by default; escalate to Spot A100 (`a2-highgpu-1g`) or G4 (`g4-standard-48`, `nvidia-rtx-pro-6000`) when needed.

    When `cpu_only=True` (or `GCP_BATCH_CPU_ONLY=1`), selects `e2-standard-4` with 0 GPUs so
    Cloud Batch runs execute without GPU quota.
    """
    if cpu_only or os.getenv("GCP_BATCH_CPU_ONLY", "0").strip().lower() in ("1", "true", "yes"):
        return CPU_DEMO_SPEC
    if prefer_g4 or os.getenv("GCP_BATCH_PREFER_G4", "0").strip().lower() in ("1", "true", "yes"):
        return G4_SPOT_SPEC
    if force_high_memory:
        return A100_SPOT_SPEC
    est_gb = estimate_gx_vram_gb(parameters)
    if est_gb > L4_SAFE_VRAM_CEILING_GB:
        return A100_SPOT_SPEC
    return L4_SPOT_SPEC


class ObtainabilityExhaustedError(RuntimeError):
    """The bounded region/SKU waterfall has no step left (raised so the watchdog escalates)."""


def resolve_candidate_regions(
    candidate_regions: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Explicit list, else `GCP_BATCH_CANDIDATE_REGIONS`, else the defaults (order kept, deduped)."""
    env_regions_raw = os.getenv("GCP_BATCH_CANDIDATE_REGIONS", "").strip()
    if candidate_regions:
        regions = [r.strip() for r in candidate_regions if r and r.strip()]
    elif env_regions_raw:
        regions = [r.strip() for r in env_regions_raw.split(",") if r.strip()]
    else:
        regions = list(DEFAULT_CANDIDATE_REGIONS)
    return list(dict.fromkeys(regions)) or ["us-central1"]


def obtainability_waterfall(
    parameters: dict[str, Any] | None = None,
    *,
    candidate_regions: list[str] | tuple[str, ...] | None = None,
    force_high_memory: bool = False,
    prefer_g4: bool = False,
    spot_preferred: bool = True,
    allow_on_demand: bool = False,
    allow_a100_fallback: bool = False,
    allow_g4_fallback: bool = False,
) -> list[tuple[str, GPUTierSpec, str]]:
    """The finite, ordered list of `(region, gpu_spec, provisioning_model)` steps (GCP-04).

    1. primary tier (L4, or A100/G4 when the VRAM estimate, an OOM, or `prefer_g4` requires it) on SPOT, per region;
    2. a fallback SKU on SPOT, per region: T4 when the run fits its 13.5 GB ceiling;
       Spot G4 (`nvidia-rtx-pro-6000`, 96 GB) with `allow_g4_fallback=True`;
       Spot A100 only with `allow_a100_fallback=True` (about 5x the L4 Spot price);
    3. on-demand STANDARD capacity for the primary tier, per region, only with
       `allow_on_demand=True` (explicit operator opt-in; never reached by default).
    The list is the hard cap: once exhausted the backend refuses to submit.
    """
    regions = resolve_candidate_regions(candidate_regions)
    primary = select_gpu_tier(
        parameters,
        force_high_memory=force_high_memory,
        prefer_g4=prefer_g4,
        cpu_only=False,
    )
    fallback: GPUTierSpec | None = None
    if primary.tier_name == "l4":
        if estimate_gx_vram_gb(parameters) <= T4_SAFE_VRAM_CEILING_GB:
            fallback = T4_SPOT_SPEC
        elif allow_g4_fallback:
            fallback = G4_SPOT_SPEC
        elif allow_a100_fallback:
            fallback = A100_SPOT_SPEC

    steps: list[tuple[str, GPUTierSpec, str]] = []
    if spot_preferred:
        steps += [(r, primary, "SPOT") for r in regions]
        if fallback is not None:
            steps += [(r, fallback, "SPOT") for r in regions]
    if allow_on_demand:
        steps += [(r, primary, "STANDARD") for r in regions]
    return steps


def select_obtainable_target(
    parameters: dict[str, Any] | None = None,
    *,
    rotation_index: int = 0,
    candidate_regions: list[str] | tuple[str, ...] | None = None,
    force_high_memory: bool = False,
    prefer_g4: bool = False,
    spot_preferred: bool = True,
    cpu_only: bool = False,
    allow_on_demand: bool = False,
    allow_a100_fallback: bool = False,
    allow_g4_fallback: bool = False,
    attempt_index: int | None = None,
) -> tuple[str, GPUTierSpec, str]:
    """Return step `rotation_index` of the bounded obtainability waterfall.

    `rotation_index` must only advance when the previous attempt could not obtain capacity
    (stockout, quota, stuck in scheduling); preemption continuations and parameter retries keep
    the current step. `attempt_index` is a deprecated alias kept for old callers.

    Raises `ObtainabilityExhaustedError` past the last step, or when SPOT is disabled without
    the on-demand opt-in (fail closed: no silent on-demand spend).
    """
    idx = max(0, int(rotation_index if attempt_index is None else attempt_index))
    regions = resolve_candidate_regions(candidate_regions)

    if cpu_only or os.getenv("GCP_BATCH_CPU_ONLY", "0").strip().lower() in ("1", "true", "yes"):
        if not spot_preferred and not allow_on_demand:
            raise ObtainabilityExhaustedError(
                "spot_provisioning=false requires allow_on_demand=true"
            )
        if idx >= len(regions):
            raise ObtainabilityExhaustedError(
                f"CPU waterfall exhausted after {len(regions)} region(s)"
            )
        return regions[idx], CPU_DEMO_SPEC, ("SPOT" if spot_preferred else "STANDARD")

    steps = obtainability_waterfall(
        parameters,
        candidate_regions=regions,
        force_high_memory=force_high_memory,
        prefer_g4=prefer_g4,
        spot_preferred=spot_preferred,
        allow_on_demand=allow_on_demand,
        allow_a100_fallback=allow_a100_fallback,
        allow_g4_fallback=allow_g4_fallback,
    )
    if not steps:
        raise ObtainabilityExhaustedError(
            "spot_provisioning=false requires allow_on_demand=true (on-demand GPUs are opt-in)"
        )
    if idx >= len(steps):
        raise ObtainabilityExhaustedError(
            f"GPU obtainability waterfall exhausted after {len(steps)} step(s) "
            f"(allow_on_demand={allow_on_demand}, allow_a100_fallback={allow_a100_fallback})"
        )
    return steps[idx]
