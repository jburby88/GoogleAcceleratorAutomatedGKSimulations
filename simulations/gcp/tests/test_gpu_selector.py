"""Tests for the bounded GPU obtainability waterfall (GCP-04)."""

from __future__ import annotations

import pytest

from gcp_exec.gpu_selector import (
    ObtainabilityExhaustedError,
    obtainability_waterfall,
    select_obtainable_target,
)

SMALL = {"nx": 64, "ny": 32, "nz": 48}  # fits a T4
MEDIUM = {"nx": 160, "ny": 80, "nz": 64, "nhermite": 12, "nlaguerre": 6}  # 13.5-20 GB: L4 only
REGIONS = ["us-central1", "us-east4", "us-west1"]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCP_BATCH_CPU_ONLY", raising=False)
    monkeypatch.delenv("GCP_BATCH_CANDIDATE_REGIONS", raising=False)


def test_select_obtainable_target_cpu_only() -> None:
    region, spec, prov = select_obtainable_target(None, cpu_only=True)
    assert (spec.tier_name, spec.accelerator_count, prov) == ("cpu", 0, "SPOT")


def test_select_obtainable_target_force_high_mem() -> None:
    _, spec, prov = select_obtainable_target(None, force_high_memory=True)
    assert (spec.tier_name, prov) == ("a100", "SPOT")


def test_default_waterfall_is_spot_only_and_finite() -> None:
    steps = obtainability_waterfall(SMALL, candidate_regions=REGIONS)
    assert [s.tier_name for _, s, _ in steps] == ["l4"] * 3 + ["t4"] * 3
    assert {p for _, _, p in steps} == {"SPOT"}
    with pytest.raises(ObtainabilityExhaustedError):
        select_obtainable_target(SMALL, rotation_index=len(steps), candidate_regions=REGIONS)


def test_a100_capacity_fallback_and_on_demand_are_opt_in() -> None:
    medium = obtainability_waterfall(MEDIUM, candidate_regions=REGIONS)
    assert [s.tier_name for _, s, _ in medium] == ["l4"] * 3  # never silently 5x the price
    with_a100 = obtainability_waterfall(MEDIUM, candidate_regions=REGIONS, allow_a100_fallback=True)
    assert [s.tier_name for _, s, _ in with_a100][-3:] == ["a100"] * 3
    on_demand = obtainability_waterfall(SMALL, candidate_regions=REGIONS, allow_on_demand=True)
    assert [p for _, _, p in on_demand][-3:] == ["STANDARD"] * 3
    with pytest.raises(ObtainabilityExhaustedError, match="allow_on_demand"):
        select_obtainable_target(SMALL, spot_preferred=False)


def test_rotation_index_alias_and_env_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_BATCH_CANDIDATE_REGIONS", "us-west4, us-west4 ,us-east1")
    r0, _, _ = select_obtainable_target(SMALL, rotation_index=0)
    r1, _, _ = select_obtainable_target(SMALL, attempt_index=1)  # deprecated alias
    assert (r0, r1) == ("us-west4", "us-east1")  # deduplicated, order kept


def test_g4_rtx_pro_6000_selection_and_fallback() -> None:
    _, spec, prov = select_obtainable_target(MEDIUM, prefer_g4=True)
    assert (spec.tier_name, spec.machine_type, spec.accelerator_type, spec.vram_gb, prov) == (
        "g4",
        "g4-standard-48",
        "nvidia-rtx-pro-6000",
        96.0,
        "SPOT",
    )
    with_g4 = obtainability_waterfall(MEDIUM, candidate_regions=REGIONS, allow_g4_fallback=True)
    assert [s.tier_name for _, s, _ in with_g4] == ["l4"] * 3 + ["g4"] * 3
