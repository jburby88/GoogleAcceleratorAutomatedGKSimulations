"""Gyrokinetic surrogate modeling, dataset ingestion, and active learning."""

from kr1_surrogate.augmented_lagrangian import AugmentedLagrangianState
from kr1_surrogate.dataset_loader import (
    CampaignReport,
    CampaignSample,
    load_campaign_report,
    load_campaign_samples,
    split_by_equilibrium,
)
from kr3_design.active_learning_queue import RunAlreadyQueuedError, enqueue_validation_run

__all__ = [
    "AugmentedLagrangianState",
    "CampaignReport",
    "CampaignSample",
    "RunAlreadyQueuedError",
    "enqueue_validation_run",
    "load_campaign_report",
    "load_campaign_samples",
    "split_by_equilibrium",
]
