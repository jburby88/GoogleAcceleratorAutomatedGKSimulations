"""KR1 leak-free Zenodo dataset loader & Augmented Lagrangian outer-loop contract."""

from kr1_surrogate.augmented_lagrangian import AugmentedLagrangianState
from kr1_surrogate.dataset_loader import (
    CampaignLoadReport,
    DatasetSample,
    ExcludedRun,
    load_campaign_report,
    load_campaign_samples,
    split_by_equilibrium,
)

__all__ = [
    "AugmentedLagrangianState",
    "CampaignLoadReport",
    "DatasetSample",
    "ExcludedRun",
    "load_campaign_report",
    "load_campaign_samples",
    "split_by_equilibrium",
]
