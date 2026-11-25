"""Dataset loading, tabular physics features, and stellarator-grouped splits."""

from itg_surrogate.data.dataset import (
    PreparedStellaratorDataset,
    compute_local_shear,
    extract_field_line_channels,
    invert_target_to_q,
    load_stellarator_h5,
    transform_heat_flux,
)
from itg_surrogate.data.features import (
    compute_3_tabular_features,
    compute_analytic_proxy_fq,
    compute_paper_12_features,
)
from itg_surrogate.data.splits import (
    StellaratorSplitIndices,
    build_canonical_stellarator_groups,
    make_stellarator_grouped_split,
    verify_zero_stellarator_leakage,
)

__all__ = [
    "PreparedStellaratorDataset",
    "StellaratorSplitIndices",
    "build_canonical_stellarator_groups",
    "compute_3_tabular_features",
    "compute_analytic_proxy_fq",
    "compute_local_shear",
    "compute_paper_12_features",
    "extract_field_line_channels",
    "invert_target_to_q",
    "load_stellarator_h5",
    "make_stellarator_grouped_split",
    "transform_heat_flux",
    "verify_zero_stellarator_leakage",
]
