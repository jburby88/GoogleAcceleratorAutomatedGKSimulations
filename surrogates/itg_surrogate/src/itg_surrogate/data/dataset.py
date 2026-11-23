"""HDF5 loading, field-line channel extraction, and train-only standardization."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np

from itg_surrogate.data.splits import StellaratorSplitIndices, make_stellarator_grouped_split
from itg_surrogate.registry.specs import DataSplitConfig, TargetMode


@dataclasses.dataclass(frozen=True)
class PreparedStellaratorDataset:
    """Preprocessed stellarator gyrokinetic dataset with train-only normalization.

    Attributes:
      z_norm: Standardized periodic field-line tensor `(N, N_z, C)` in `float32`.
      s_norm: Standardized scalar gradients `(N, 2)` `[a/LT, a/Ln]` in `float32`.
      z_raw: Unnormalized physical field-line tensor `(N, N_z, C)` in `float32`.
      scalars_raw: Unnormalized scalar gradients `(N, 2)` in `float32`.
      raw_feature_tensor: Filtered raw GX 7-channel tensor `(N, N_z, 7)`.
      targets: Transformed heat-flux regression targets `(N,)` in `float32`.
      q_varied: Physical turbulent heat flux `Q / Q_gB` `(N,)` in `float32`.
      q_fixed: Fixed-gradient turbulent heat flux `(N,)` in `float32`.
      unstable_mask: Boolean mask `(N,)` indicating `Q / Q_gB > 0.1`.
      equilibrium_class: Integer class IDs `(N,)` in `0..4`.
      equilibrium_descriptions: Human-readable names of the 5 equilibrium classes.
      channel_names: Ordered names of the `C` periodic field-line channels.
      z_mean: Training-split channel mean `(1, 1, C)` used for `z_norm`.
      z_std: Training-split channel std `(1, 1, C)` used for `z_norm`.
      s_mean: Training-split scalar mean `(1, 2)` used for `s_norm`.
      s_std: Training-split scalar std `(1, 2)` used for `s_norm`.
      split: Verified `StellaratorSplitIndices` with `0.00%` stellarator overlap.
      config: `DataSplitConfig` used to construct the dataset.
    """

    z_norm: np.ndarray
    s_norm: np.ndarray
    z_raw: np.ndarray
    scalars_raw: np.ndarray
    raw_feature_tensor: np.ndarray
    targets: np.ndarray
    q_varied: np.ndarray
    q_fixed: np.ndarray
    unstable_mask: np.ndarray
    equilibrium_class: np.ndarray
    equilibrium_descriptions: tuple[str, ...]
    channel_names: tuple[str, ...]
    z_mean: np.ndarray
    z_std: np.ndarray
    s_mean: np.ndarray
    s_std: np.ndarray
    split: StellaratorSplitIndices
    config: DataSplitConfig


def compute_local_shear(gds21: np.ndarray, gds22: np.ndarray) -> np.ndarray:
    """Computes spectral local magnetic shear `S = d/dz(gds21 / gds22)` on `z`."""
    ratio = gds21 / gds22
    num_z = ratio.shape[1]
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(num_z, d=1.0 / num_z)
    ratio_hat = np.fft.rfft(ratio, axis=1)
    shear = np.fft.irfft(1j * wavenumbers[None, :] * ratio_hat, n=num_z, axis=1)
    return shear.astype(np.float32)


def extract_field_line_channels(
    raw_tensor: np.ndarray,
    include_local_shear: bool = False,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extracts periodic geometric `z`-profiles (scaling GX drifts by 1/2 per Eq. 2.11)."""
    bmag = raw_tensor[:, :, 0:1]
    gbdrift = raw_tensor[:, :, 1:2] / 2.0
    cvdrift = raw_tensor[:, :, 2:3] / 2.0
    gbdrift0 = raw_tensor[:, :, 3:4] / 2.0
    gds2 = raw_tensor[:, :, 4:5]
    gds21 = raw_tensor[:, :, 5:6]
    gds22 = raw_tensor[:, :, 6:7]

    parts = [bmag, gbdrift, cvdrift, gbdrift0, gds2, gds21, gds22]
    names = ["B", "gbdrift", "cvdrift", "gbdrift0", "gds2", "gds21", "gds22"]
    if include_local_shear:
        shear = compute_local_shear(raw_tensor[:, :, 5], raw_tensor[:, :, 6])[:, :, None]
        parts.append(shear)
        names.append("local_shear")
    return np.concatenate(parts, axis=-1).astype(np.float32), tuple(names)


def transform_heat_flux(q_values: np.ndarray, mode: TargetMode) -> np.ndarray:
    """Maps physical heat flux `Q / Q_gB` to regression target `y`."""
    if mode == "log1p":
        return np.log1p(np.maximum(q_values, 0.0)).astype(np.float32)
    if mode == "log_q":
        return np.log(np.maximum(q_values, 0.1)).astype(np.float32)
    raise ValueError(f"Unsupported target_mode: {mode!r}")


def invert_target_to_q(y_values: np.ndarray, mode: TargetMode) -> np.ndarray:
    """Inverts regression predictions back to physical heat flux `Q / Q_gB`."""
    if mode == "log1p":
        return np.maximum(np.expm1(y_values), 0.0)
    if mode == "log_q":
        return np.exp(y_values)
    raise ValueError(f"Unsupported target_mode: {mode!r}")


def load_stellarator_h5(
    h5_path: str | Path,
    config: DataSplitConfig | None = None,
) -> PreparedStellaratorDataset:
    """Loads `20250102-01_GX_stellarator_dataset.h5` and builds the OOD split.

    Standardization statistics (`z_mean`, `z_std`, `s_mean`, `s_std`) are computed
    strictly from `split.train_idx` so validation and test stellarators remain
    100% unseen.
    """
    cfg = config if config is not None else DataSplitConfig()
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"GX stellarator HDF5 dataset not found: {path}")

    with h5py.File(path, "r") as f:
        raw_feature_tensor = f["/raw_feature_tensor"][()]
        a_over_lt = f["/varied_gradient_simulations/a_over_LT"][()]
        a_over_ln = f["/varied_gradient_simulations/a_over_Ln"][()]
        q_varied_all = f["/varied_gradient_simulations/Q_avgs"][()]
        q_fixed_all = f["/fixed_gradient_simulations/Q_avgs"][()]
        eq_files_all = f["/equilibrium_files"][()]
        quasr_ids_all = f["/QUASR_IDs"][()].astype(np.int64)
        eq_class_all = f["/equilibrium_class"][()].astype(np.int64)
        eq_desc = tuple(d.decode("utf-8") for d in f["/equilibrium_class_descriptions"][()])

    finite_mask = np.isfinite(q_varied_all) & np.all(np.isfinite(raw_feature_tensor), axis=(1, 2))
    if cfg.filter_mode == "positive_q":
        valid_mask = finite_mask & (q_varied_all > 0.0)
    elif cfg.filter_mode == "unstable_only":
        valid_mask = finite_mask & (q_varied_all > 0.1)
    else:
        raise ValueError(f"Unsupported filter_mode: {cfg.filter_mode!r}")

    raw_valid = raw_feature_tensor[valid_mask]
    lt_valid = a_over_lt[valid_mask].astype(np.float32)
    ln_valid = a_over_ln[valid_mask].astype(np.float32)
    q_valid = q_varied_all[valid_mask].astype(np.float32)
    q_fixed_valid = q_fixed_all[valid_mask].astype(np.float32)
    eq_files_valid = eq_files_all[valid_mask]
    quasr_ids_valid = quasr_ids_all[valid_mask]
    eq_class_valid = eq_class_all[valid_mask]

    z_raw, channel_names = extract_field_line_channels(
        raw_valid, include_local_shear=cfg.include_local_shear
    )
    scalars_raw = np.column_stack([lt_valid, ln_valid]).astype(np.float32)
    targets = transform_heat_flux(q_valid, cfg.target_mode)
    unstable_mask = q_valid > 0.1

    split = make_stellarator_grouped_split(
        equilibrium_files=eq_files_valid,
        quasr_ids=quasr_ids_valid,
        equilibrium_class=eq_class_valid,
        config=cfg,
    )

    z_mean = z_raw[split.train_idx].mean(axis=(0, 1), keepdims=True)
    z_std = z_raw[split.train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
    z_norm = ((z_raw - z_mean) / z_std).astype(np.float32)

    s_mean = scalars_raw[split.train_idx].mean(axis=0, keepdims=True)
    s_std = scalars_raw[split.train_idx].std(axis=0, keepdims=True) + 1e-8
    s_norm = ((scalars_raw - s_mean) / s_std).astype(np.float32)

    return PreparedStellaratorDataset(
        z_norm=z_norm,
        s_norm=s_norm,
        z_raw=z_raw,
        scalars_raw=scalars_raw,
        raw_feature_tensor=raw_valid,
        targets=targets,
        q_varied=q_valid,
        q_fixed=q_fixed_valid,
        unstable_mask=unstable_mask,
        equilibrium_class=eq_class_valid,
        equilibrium_descriptions=eq_desc,
        channel_names=channel_names,
        z_mean=z_mean.astype(np.float32),
        z_std=z_std.astype(np.float32),
        s_mean=s_mean.astype(np.float32),
        s_std=s_std.astype(np.float32),
        split=split,
        config=cfg,
    )
