"""Stratified stellarator-grouped train/validation/test split (zero OOD leakage).

Design Decision — Why Only Stellarator-Grouped Splitting Is Supported:
  In the Landreman et al. (2025) GX stellarator dataset (`100,705` simulations,
  `98,932` with `Q > 0`), rows are NOT independent stellarators. Two distinct
  levels of shared stellarator identity exist across rows:

  1. Sibling flux tubes per 3D MHD equilibrium (`/equilibrium_files`, `23,577`
     unique VMEC files): Each 3D equilibrium contributes ~4 radial surfaces
     (and up to ~20 `(psi, alpha)` tubes in QUASR vacuum equilibria). Under a
     uniform random 80/10/10 flux-tube split, **99.09%** (`9,803 / 9,893`) of
     held-out test tubes share an exact `equilibrium_file` with training tubes.
  2. Shared boundary shapes across vacuum and finite-beta QUASR equilibria
     (`/QUASR_IDs`, `418` unique positive IDs; `-1` sentinel for non-QUASR):
     Class 4 (*QUASR configurations with added pressure*, `3,964` equilibria)
     was constructed by adding finite-beta pressure profiles to the same `413`
     Class 3 (*QUASR vacuum*) boundary shapes. Under a random split, **100.00%**
     (`2,397 / 2,397`) of QUASR test tubes share a `QUASR_ID` with training
     tubes (**99.35%** canonical stellarator overlap overall). Even grouping by
     `equilibrium_files` alone leaks QUASR boundary shapes between Class 3 and
     Class 4.

  Because the downstream scientific goal is surrogate-assisted optimization of
  **brand-new candidate stellarators** where no sibling flux tube from the
  candidate geometry exists in the training set, this codebase exclusively
  supports the stratified canonical stellarator-grouped split implemented below
  and automatically verifies `0.00%` overlap in both `equilibrium_files` and
  positive `QUASR_IDs`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from itg_surrogate.registry.specs import DataSplitConfig


@dataclasses.dataclass(frozen=True)
class StellaratorSplitIndices:
    """Index partition and group metadata for the unseen-stellarator split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    canonical_group_ids: np.ndarray
    n_train_groups: int
    n_val_groups: int
    n_test_groups: int


def build_canonical_stellarator_groups(
    equilibrium_files: np.ndarray,
    quasr_ids: np.ndarray,
) -> np.ndarray:
    """Constructs canonical stellarator family IDs unifying QUASR & VMEC siblings.

    Args:
      equilibrium_files: 1D array of decoded string (or bytes) equilibrium paths
        of shape `(N,)`.
      quasr_ids: 1D integer array of `/QUASR_IDs` of shape `(N,)`, where `>= 0`
        identifies a QUASR boundary family and `< 0` (`-1`) denotes non-QUASR
        equilibria (random Fourier modes and rotating ellipses).

    Returns:
      1D string array of canonical stellarator group keys (`'quasr_<id>'` when
      `quasr_id >= 0`, else `'eq_<equilibrium_file>'`).
    """
    if equilibrium_files.shape != quasr_ids.shape or equilibrium_files.ndim != 1:
        raise ValueError(
            "equilibrium_files and quasr_ids must be 1D arrays of identical length; "
            f"got {equilibrium_files.shape} and {quasr_ids.shape}."
        )
    decoded_eq = [
        ef.decode("utf-8") if isinstance(ef, (bytes, np.bytes_)) else str(ef)
        for ef in equilibrium_files
    ]
    return np.array(
        [
            f"quasr_{int(qid)}" if int(qid) >= 0 else f"eq_{ef}"
            for qid, ef in zip(quasr_ids, decoded_eq, strict=True)
        ],
        dtype=object,
    )


def verify_zero_stellarator_leakage(
    split: StellaratorSplitIndices,
    equilibrium_files: np.ndarray,
    quasr_ids: np.ndarray,
) -> None:
    """Raises `ValueError` if any equilibrium or QUASR family leaks across splits."""
    decoded_eq = np.array(
        [
            ef.decode("utf-8") if isinstance(ef, (bytes, np.bytes_)) else str(ef)
            for ef in equilibrium_files
        ],
        dtype=object,
    )
    pairs = (
        ("train", split.train_idx, "val", split.val_idx),
        ("train", split.train_idx, "test", split.test_idx),
        ("val", split.val_idx, "test", split.test_idx),
    )
    for name_a, idx_a, name_b, idx_b in pairs:
        if len(np.intersect1d(idx_a, idx_b)) > 0:
            raise ValueError(f"Row index overlap detected between {name_a} and {name_b}.")

        grp_overlap = set(split.canonical_group_ids[idx_a]) & set(split.canonical_group_ids[idx_b])
        if grp_overlap:
            raise ValueError(
                f"Canonical stellarator group leakage between {name_a} and {name_b}: "
                f"{len(grp_overlap)} shared groups."
            )

        eq_overlap = set(decoded_eq[idx_a]) & set(decoded_eq[idx_b])
        if eq_overlap:
            raise ValueError(
                f"equilibrium_files leakage between {name_a} and {name_b}: "
                f"{len(eq_overlap)} shared equilibria."
            )

        qa = quasr_ids[idx_a]
        qb = quasr_ids[idx_b]
        quasr_overlap = set(qa[qa >= 0].tolist()) & set(qb[qb >= 0].tolist())
        if quasr_overlap:
            raise ValueError(
                f"QUASR_ID leakage between {name_a} and {name_b}: "
                f"{len(quasr_overlap)} shared QUASR boundary IDs."
            )


def make_stellarator_grouped_split(
    equilibrium_files: np.ndarray,
    quasr_ids: np.ndarray,
    equilibrium_class: np.ndarray,
    config: DataSplitConfig,
) -> StellaratorSplitIndices:
    """Builds a stratified unseen-stellarator train/val/test split (`0%` overlap).

    Groups are partitioned independently within each disjoint stellarator-family
    stratum:
      - Stratum `0`: Random boundary Fourier modes (`QUASR_ID == -1`)
      - Stratum `1`: Rotating ellipses on a circle (`QUASR_ID == -1`)
      - Stratum `2`: Rotating ellipses with torsion (`QUASR_ID == -1`)
      - Stratum `3`: QUASR family unifying Class 3 vacuum + Class 4 finite-beta
        equilibria via `QUASR_ID >= 0`

    Args:
      equilibrium_files: 1D array of `/equilibrium_files` after dataset filtering.
      quasr_ids: 1D integer array of `/QUASR_IDs` after dataset filtering.
      equilibrium_class: 1D integer array of `/equilibrium_class` (`0..4`).
      config: `DataSplitConfig` specifying fractions and `split_seed`.

    Returns:
      Verified `StellaratorSplitIndices` with zero stellarator overlap.
    """
    canonical_groups = build_canonical_stellarator_groups(equilibrium_files, quasr_ids)
    quasr_stratum_id = int(np.max(equilibrium_class)) + 1
    group_stratum = np.where(quasr_ids >= 0, quasr_stratum_id, equilibrium_class)

    rng = np.random.RandomState(config.split_seed)
    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()

    for s_id in sorted(np.unique(group_stratum).tolist()):
        s_groups = np.unique(canonical_groups[group_stratum == s_id])
        perm = rng.permutation(len(s_groups))
        n_g = len(s_groups)
        n_test = max(1, int(round(config.test_frac * n_g))) if n_g >= 3 else 0
        n_val = max(1, int(round(config.val_frac * n_g))) if n_g >= 3 else 0
        n_train = n_g - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"Stratum {s_id} has only {n_g} groups; cannot form non-empty splits.")
        train_groups.update(s_groups[perm[:n_train]].tolist())
        val_groups.update(s_groups[perm[n_train : n_train + n_val]].tolist())
        test_groups.update(s_groups[perm[n_train + n_val :]].tolist())

    train_idx = np.array(
        [i for i, g in enumerate(canonical_groups) if g in train_groups],
        dtype=np.int64,
    )
    val_idx = np.array(
        [i for i, g in enumerate(canonical_groups) if g in val_groups],
        dtype=np.int64,
    )
    test_idx = np.array(
        [i for i, g in enumerate(canonical_groups) if g in test_groups],
        dtype=np.int64,
    )

    split = StellaratorSplitIndices(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        canonical_group_ids=canonical_groups,
        n_train_groups=len(train_groups),
        n_val_groups=len(val_groups),
        n_test_groups=len(test_groups),
    )
    verify_zero_stellarator_leakage(split, equilibrium_files, quasr_ids)
    return split
