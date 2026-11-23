"""Engineered physical geometry features from Landreman et al. (2025) Sec. 5."""

from __future__ import annotations

import numpy as np

from itg_surrogate.data.dataset import extract_field_line_channels


def compute_analytic_proxy_fq(raw_tensor: np.ndarray) -> np.ndarray:
    """Computes Eq. (5.1) scalar heat-flux proxy `f_Q` from raw GX z-functions."""
    bmag = raw_tensor[:, :, 0]
    cvdrift_gx = raw_tensor[:, :, 2]
    grad_x_sq = raw_tensor[:, :, 6]
    return np.mean(
        (np.heaviside(cvdrift_gx, 0.0) + 0.2) * (grad_x_sq**1.5) / bmag,
        axis=1,
    ).astype(np.float32)


def compute_3_tabular_features(
    raw_tensor: np.ndarray,
    a_over_lt: np.ndarray,
    a_over_ln: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Returns `(a/LT, a/Ln, f_Q)` matrix `(N, 3)` and feature names."""
    fq = compute_analytic_proxy_fq(raw_tensor)
    matrix = np.column_stack([a_over_lt, a_over_ln, fq]).astype(np.float32)
    return matrix, ("a/LT", "a/Ln", "f_Q")


def compute_paper_12_features(
    raw_tensor: np.ndarray,
    a_over_lt: np.ndarray,
    a_over_ln: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Computes the 12 forward-selected tabular features from Section 5 / Fig. 13."""
    ch, _ = extract_field_line_channels(raw_tensor, include_local_shear=True)
    bmag = ch[:, :, 0]
    gbdrift = ch[:, :, 1]
    cvdrift = ch[:, :, 2]
    gbdrift0 = ch[:, :, 3]
    gds2 = ch[:, :, 4]
    gds21 = ch[:, :, 5]
    gds22 = ch[:, :, 6]
    shear = ch[:, :, 7]

    num_z = raw_tensor.shape[1]
    f0 = a_over_lt
    f1 = a_over_ln
    f2 = np.mean(((gds22 / bmag) * np.heaviside(cvdrift, 0.0)) ** 2, axis=1)
    f3 = np.median(((gds22**2) * gbdrift0 / bmag) ** 2, axis=1)
    f4 = np.abs(np.fft.rfft(np.maximum(0.0, gbdrift0) / (bmag**2), axis=1)[:, 1]) / (num_z / 2.0)
    f5 = np.mean(gbdrift / bmag, axis=1)
    f6 = np.var(np.maximum(0.0, gbdrift) / (gds2 * bmag), axis=1)
    f7 = np.mean((gds22 / (bmag**3)) ** 2, axis=1)
    f8 = np.mean(gds21 * gbdrift0 / (bmag**3), axis=1)
    f9 = np.var((np.heaviside(gbdrift, 0.0) / gds2) ** 2, axis=1)
    f10 = np.mean(np.abs(shear / (bmag**3)), axis=1)
    f11 = np.mean((gbdrift0 * bmag) ** 2, axis=1)

    names = (
        "a/LT",
        "a/Ln",
        "mean((|grad x|^2 * B^-1 * Theta(cvdrift))^2)",
        "median((|grad x|^4 * gbdrift0 * B^-1)^2)",
        "absFFTCoeff1(ReLU(gbdrift0) * B^-2)",
        "mean(gbdrift * B^-1)",
        "var(ReLU(gbdrift) / (|grad y|^2 * B))",
        "meanSquare(|grad x|^2 * B^-3)",
        "mean((grad_x . grad_y) * gbdrift0 * B^-3)",
        "var((Theta(gbdrift) / |grad y|^2)^2)",
        "L1Norm(S * B^-3)",
        "mean((gbdrift0 * B)^2)",
    )
    matrix = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]).astype(np.float32)
    return matrix, names
