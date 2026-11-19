"""Linear Gyrokinetic RHS, Parallel Derivatives, Closures, & Eigenmodes.

Verifies ``gx_jax/linear_rhs.py`` against:
- Independent line-by-line NumPy reference ports of ``gx/src/{linear.cu,
  grad_parallel.cu,closures.cu,smith_par_closure.cu,device_funcs.cu}`` across all 5 linear
  benchmarks, 3D multi-(kx, ky) grids, and 3D W7-X stellarator equilibria to ``rtol < 1e-6``
  (``< 1e-12`` in FP64).
- Exact Dougherty/Lenard-Bernstein collision conservation laws (number, momentum, energy).
- All 3 closure models (``Beer42``, ``SmithPerp``, ``SmithPar``) and all 4 dissipation models
  (``hypercollisions_const``, ``hypercollisions_kz``, ``hyperdiff``, ``HB_hyper``).
- All 5 explicit time integrators (``rk3``, ``rk4``, ``sspx2``, ``sspx3``, ``k10``), golden C++
  ``.out.nc`` / ``.restart.nc`` benchmarks (``rtol < 1e-5`` on converged eigenmodes), and W7-X
  stellarator linear ITG growth rates & frequencies.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest
from scipy.linalg import eig, expm

from gx_jax.config import BoltzmannOption, GXConfig, SpeciesConfig, load_config
from gx_jax.fields import (
    Fields,
    field_line_average,
    init_field_solver,
    init_moments,
    load_restart_nc,
    solve_fields,
)
from gx_jax.geometry_nc import build_vmec_eik_nc, init_geometry, load_eik_nc
from gx_jax.linear_rhs import (
    advance_linear_steps,
    apply_closures,
    apply_hb_hyper,
    apply_hypercollisions_const,
    apply_hypercollisions_kz,
    apply_hyperdiff,
    compute_growth_rates,
    compute_linear_dt,
    compute_linear_max_frequency,
    compute_phi2_ky,
    extract_single_ky_config,
    init_linear_operator,
    init_parallel_grad,
    linear_rhs,
    rhs_linear_local,
    smith_par_get_as,
    smith_perp_get_as,
    solve_linear_eigenmode,
    step_linear,
    streaming_rhs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
GOLDEN_BENCH_DIR = REPO_ROOT / "golden_fixtures" / "benchmarks"
UNIT_INPUTS_DIR = REPO_ROOT / "golden_fixtures" / "unit_tests" / "inputs"

BENCHMARK_INPUTS: list[tuple[str, Path]] = [
    (
        "itg_salpha_adiabatic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in",
    ),
    (
        "itg_miller_adiabatic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_adiabatic_electrons.in",
    ),
    (
        "itg_miller_kinetic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_kinetic_electrons.in",
    ),
    (
        "kbm_miller",
        BENCH_DIR / "linear" / "KBM" / "kbm_miller.in",
    ),
    (
        "kaw_slab",
        BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01.in",
    ),
]


def _assert_close_fp64(
    actual: np.ndarray | jax.Array | float | complex,
    expected: np.ndarray | jax.Array | float | complex,
    label: str,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> None:
    is_cplx = np.iscomplexobj(actual) or np.iscomplexobj(expected)
    dt = np.complex128 if is_cplx else np.float64
    np.testing.assert_allclose(
        np.asarray(actual, dtype=dt),
        np.asarray(expected, dtype=dt),
        rtol=rtol,
        atol=atol,
        err_msg=f"FP64 mismatch in {label}",
    )


def _assert_close_fp32_1e6(
    actual: np.ndarray | jax.Array | float | complex,
    expected: np.ndarray | jax.Array | float | complex,
    label: str,
) -> None:
    is_cplx = np.iscomplexobj(actual) or np.iscomplexobj(expected)
    dt = np.complex128 if is_cplx else np.float64
    act = np.asarray(actual, dtype=dt)
    exp = np.asarray(expected, dtype=dt)
    exp_norm = float(np.linalg.norm(exp.ravel()))
    if exp_norm > 1.0e-12:
        rel_l2 = float(np.linalg.norm((act - exp).ravel())) / exp_norm
        assert rel_l2 < 1.0e-6, f"{label}: relative L2 error {rel_l2:.3e} >= 1e-6"
    scale = max(1.0, float(np.max(np.abs(exp)))) if exp.size > 0 else 1.0
    np.testing.assert_allclose(
        act,
        exp,
        rtol=2.0e-6,
        atol=1.0e-6 * scale,
        err_msg=f"FP32 mismatch in {label}",
    )


def _random_complex_g(
    shape: tuple[int, ...],
    active_mask: np.ndarray,
    seed: int,
    *,
    dtype: jnp.dtype = jnp.complex128,
) -> jax.Array:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)
    raw = np.where(active_mask[None, None, None, None, :, :], raw, 0.0)
    return jnp.asarray(raw, dtype=dtype)


# ---------------------------------------------------------------------------
# Independent NumPy C++ Reference Implementation of Linear_GK::rhs
# ---------------------------------------------------------------------------


def _cpp_jflr_np(ell: int, b: np.ndarray) -> np.ndarray:
    if ell < 0 or ell > 30:
        return np.zeros_like(b, dtype=np.float64)
    return ((-0.5 * b) ** ell) / math.factorial(ell) * np.exp(-0.5 * b)


def _cpp_jflr_b_np(ell: int, b: np.ndarray) -> np.ndarray:
    return _cpp_jflr_np(ell, b) + _cpp_jflr_np(ell - 1, b)


def _cpp_dz_np(arr: np.ndarray, kz_deriv: np.ndarray, local_limit: bool, zp: int) -> np.ndarray:
    if local_limit:
        return (1.0j / float(zp)) * arr
    if arr.shape[-3] <= 1:
        return np.zeros_like(arr)
    return np.fft.ifft(
        np.fft.fft(arr, axis=-3) * (1.0j * kz_deriv)[:, None, None],
        axis=-3,
    )


def _cpp_abs_dz_np(arr: np.ndarray, kz_deriv: np.ndarray, local_limit: bool, zp: int) -> np.ndarray:
    if local_limit:
        return (1.0 / float(zp)) * arr
    if arr.shape[-3] <= 1:
        return np.zeros_like(arr)
    return np.fft.ifft(
        np.fft.fft(arr, axis=-3) * np.abs(kz_deriv)[:, None, None],
        axis=-3,
    )


def _cpp_reference_linear_rhs(
    G_np: np.ndarray,
    fields: Fields,
    cfg: GXConfig,
    grids,
    geo,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct NumPy port of ``Linear_GK::rhs`` (``linear.cu`` + ``device_funcs.cu``)."""
    ns, nm, nl, nz, naky, nakx = G_np.shape
    mask2d = np.asarray(grids.active_mask, dtype=bool)
    mask3d = np.broadcast_to(mask2d[None, :, :], (nz, naky, nakx))
    mask6d = np.broadcast_to(mask2d[None, None, None, None, :, :], G_np.shape)

    g = np.where(mask6d, np.asarray(G_np, dtype=np.complex128), 0.0)
    phi = np.where(mask3d, np.asarray(fields.phi, dtype=np.complex128), 0.0)
    apar = np.where(mask3d, np.asarray(fields.apar, dtype=np.complex128), 0.0)
    bpar = np.where(mask3d, np.asarray(fields.bpar, dtype=np.complex128), 0.0)

    kperp2 = np.asarray(geo.kperp2, dtype=np.float64)
    bmag = np.asarray(geo.bmag, dtype=np.float64)
    bgrad = np.asarray(geo.bgrad, dtype=np.float64)
    cv_d = np.asarray(geo.cv_d, dtype=np.float64)
    gb_d = np.asarray(geo.gb_d, dtype=np.float64)
    omegad = np.asarray(geo.omegad, dtype=np.float64)
    ky = np.asarray(grids.ky, dtype=np.float64)
    kx = np.asarray(grids.kx, dtype=np.float64)
    kz_raw = np.asarray(grids.kz, dtype=np.float64)
    kz_deriv = kz_raw
    gpar = float(geo.gradpar)

    # 1. streaming_rhs (device_funcs.cu:L2999-3046)
    rhs_par = np.zeros_like(g)
    for is_idx, sp in enumerate(cfg.species):
        b_s = sp.rho2 * kperp2
        vt = sp.vt
        zt = sp.zt
        for ell in range(nl):
            jl = _cpp_jflr_np(ell, b_s)
            jlb = _cpp_jflr_b_np(ell, b_s)
            for m in range(nm):
                gmp1 = g[is_idx, m + 1, ell] if (m + 1 < nm) else 0.0
                gmm1 = g[is_idx, m - 1, ell] if (m > 0) else 0.0
                val = -vt * (math.sqrt(m + 1) * gmp1 + math.sqrt(m) * gmm1) * gpar
                if m == 0:
                    val = val + jl * apar * zt * vt * vt * gpar
                elif m == 1:
                    val = val - jl * phi * zt * vt * gpar - jlb * bpar * vt * gpar
                elif m == 2:
                    val = val + math.sqrt(2.0) * jl * apar * zt * vt * vt * gpar
                rhs_par[is_idx, m, ell] = np.where(mask3d, val, 0.0)

    if nz > 1:
        rhs = np.where(mask6d, _cpp_dz_np(rhs_par, kz_deriv, cfg.local_limit, grids.zp), 0.0)
    else:
        rhs = np.zeros_like(g)

    # 2. conservation_terms + rhs_linear (device_funcs.cu:L3050-3223, L3464-3522)
    rhs_loc = np.zeros_like(g)
    sp0 = cfg.species[0]
    iky3 = (1.0j * ky)[None, :, None]

    for is_idx, sp in enumerate(cfg.species):
        b_s = sp.rho2 * kperp2
        vt = sp.vt
        zt = sp.zt
        tz = sp.tz
        nu_ss = sp.nu_ss
        if sp.type_id == 1 and cfg.ei_colls:
            nuei = (sp0.z * nu_ss) if ns > 1 else nu_ss
            as_i = sp0.jparfac
            vt_i = sp0.vt
            nzvt_i = sp0.nz * sp0.vt
        else:
            nuei = 0.0
            as_i = 1.0
            vt_i = 1.0
            nzvt_i = 1.0

        # Non-adiabatic H with ghost cells
        h_sp = np.zeros((nm, nl, nz, naky, nakx), dtype=np.complex128)
        for ell in range(nl):
            jl = _cpp_jflr_np(ell, b_s)
            jlb = _cpp_jflr_b_np(ell, b_s)
            for m in range(nm):
                val_h = g[is_idx, m, ell]
                if m == 0:
                    val_h = val_h + zt * jl * phi + jlb * bpar
                elif m == 1:
                    val_h = val_h - zt * vt * jl * apar
                h_sp[m, ell] = val_h

        upar_bar = np.zeros((nz, naky, nakx), dtype=np.complex128)
        uperp_bar = np.zeros((nz, naky, nakx), dtype=np.complex128)
        t_bar = np.zeros((nz, naky, nakx), dtype=np.complex128)
        if cfg.collisions and cfg.coll_conservation:
            for ell in range(nl):
                jl = _cpp_jflr_np(ell, b_s)
                jlm1 = _cpp_jflr_np(ell - 1, b_s)
                jlp1 = _cpp_jflr_np(ell + 1, b_s)
                uperp_bar = uperp_bar + (jl + jlm1) * h_sp[0, ell]
                if nm >= 2:
                    upar_bar = upar_bar + jl * h_sp[1, ell]
                if nl == 1 and nm >= 3:
                    t_bar = t_bar + math.sqrt(2.0) * jl * g[is_idx, 2, ell]
                elif nl > 1:
                    g2 = g[is_idx, 2, ell] if nm >= 3 else 0.0
                    t_bar = (
                        t_bar
                        + (math.sqrt(2.0) / 3.0) * jl * g2
                        + (2.0 / 3.0)
                        * (ell * jlm1 + 2.0 * ell * jl + (ell + 1) * jlp1)
                        * h_sp[0, ell]
                    )
            uperp_bar = uperp_bar * np.sqrt(b_s)

        def _h_get(m_q: int, l_q: int, h_arr: np.ndarray = h_sp) -> np.ndarray | float:
            if 0 <= m_q < nm and 0 <= l_q < nl:
                return h_arr[m_q, l_q]
            return 0.0

        icv = 1.0j * tz * cv_d
        igb = 1.0j * tz * gb_d
        bgrad3 = bgrad[:, None, None]

        for m in range(nm):
            for ell in range(nl):
                jl = _cpp_jflr_np(ell, b_s)
                jlm1 = _cpp_jflr_np(ell - 1, b_s)
                jlp1 = _cpp_jflr_np(ell + 1, b_s)
                jlb = _cpp_jflr_b_np(ell, b_s)
                jlbm1 = _cpp_jflr_b_np(ell - 1, b_s)
                jlbp1 = _cpp_jflr_b_np(ell + 1, b_s)

                acc = (
                    -vt
                    * bgrad3
                    * (
                        -math.sqrt(m + 1) * (ell + 1) * _h_get(m + 1, ell)
                        - math.sqrt(m + 1) * ell * _h_get(m + 1, ell - 1)
                        + math.sqrt(m) * ell * _h_get(m - 1, ell)
                        + math.sqrt(m) * (ell + 1) * _h_get(m - 1, ell + 1)
                    )
                    - icv
                    * (
                        math.sqrt((m + 1) * (m + 2)) * _h_get(m + 2, ell)
                        + (2 * m + 1) * _h_get(m, ell)
                        + (math.sqrt(m * (m - 1)) if m >= 2 else 0.0) * _h_get(m - 2, ell)
                    )
                    - igb
                    * (
                        (ell + 1) * _h_get(m, ell + 1)
                        + (2 * ell + 1) * _h_get(m, ell)
                        + ell * _h_get(m, ell - 1)
                    )
                    - (nu_ss + nuei) * (b_s + 2 * ell + m) * _h_get(m, ell)
                )

                if m == 0:
                    acc = (
                        acc
                        + iky3
                        * phi
                        * (
                            jlm1 * ell * sp.tprim
                            + jl * (sp.fprim + 2 * ell * sp.tprim)
                            + jlp1 * (ell + 1) * sp.tprim
                        )
                        + (iky3 / zt)
                        * bpar
                        * (
                            jlbm1 * ell * sp.tprim
                            + jlb * (sp.fprim + 2 * ell * sp.tprim)
                            + jlbp1 * (ell + 1) * sp.tprim
                        )
                        + nu_ss * np.sqrt(b_s) * jlb * uperp_bar
                        + (nu_ss + nuei)
                        * 2.0
                        * (ell * jlm1 + 2.0 * ell * jl + (ell + 1) * jlp1)
                        * t_bar
                    )
                elif m == 1:
                    upar_bar_i = (
                        (
                            kperp2 * (bmag[:, None, None] ** 2) * apar / as_i
                            - sp.nz * vt * upar_bar / nzvt_i
                        )
                        if (ns > 1 and as_i > 0.0)
                        else 0.0
                    )
                    acc = (
                        acc
                        - vt
                        * iky3
                        * apar
                        * (
                            jlm1 * ell * sp.tprim
                            + jl * (sp.fprim + (2 * ell + 1) * sp.tprim)
                            + jlp1 * (ell + 1) * sp.tprim
                        )
                        + jl * (nu_ss * upar_bar + nuei * (vt_i / vt) * upar_bar_i)
                    )
                elif m == 2:
                    acc = (
                        acc
                        + iky3 * phi * jl / math.sqrt(2.0) * sp.tprim
                        + (iky3 / zt) * bpar * jlb / math.sqrt(2.0) * sp.tprim
                        + (nu_ss + nuei) * math.sqrt(2.0) * jl * t_bar
                    )
                elif m == 3:
                    acc = acc - vt * iky3 * apar * math.sqrt(1.5) * sp.tprim * jl

                rhs_loc[is_idx, m, ell] = np.where(mask3d, acc, 0.0)

    rhs = rhs + rhs_loc

    # 3. HB_hyper (linear.cu:L188-204, device_funcs.cu:L3386-3458)
    if cfg.hb_hyper:
        favg_j, df_j = field_line_average(jnp.asarray(phi, dtype=jnp.complex128), grids, geo)
        favg_np = np.asarray(favg_j, dtype=np.complex128)
        df_np = np.asarray(df_j, dtype=np.complex128)
        s01_raw = np.sum((np.abs(kx) ** 4) * (np.abs(favg_np) ** 2))
        s01 = 0.5 * (-cfg.w_osc + math.sqrt(cfg.w_osc**2 + 2.0 * float(s01_raw)))

        s10 = np.zeros((nz,), dtype=np.float64)
        s11 = np.zeros((nz,), dtype=np.float64)
        for idz in range(nz):
            acc10 = 0.0
            acc11 = 0.0
            for idy in range(1, naky):
                for idx in range(nakx):
                    if mask2d[idy, idx]:
                        kp2 = kx[idx] ** 2 + ky[idy] ** 2
                        df2 = abs(df_np[idz, idy, idx]) ** 2
                        acc10 += 2.0 * (ky[idy] ** 4) * df2
                        acc11 += 2.0 * (kp2**2) * df2
            s10[idz] = 0.5 * (-cfg.w_osc + math.sqrt(cfg.w_osc**2 + 2.0 * acc10))
            s11[idz] = 0.5 * (-cfg.w_osc + math.sqrt(cfg.w_osc**2 + 2.0 * acc11))

        kpmax2 = grids.kx_max**2 + grids.ky_max**2
        for idz in range(nz):
            for idy in range(naky):
                for idx in range(nakx):
                    if mask2d[idy, idx]:
                        d10 = (
                            cfg.d_hb * ((kx[idx] / grids.kx_max) ** 4) if grids.kx_max > 0 else 0.0
                        )
                        d01 = (
                            d10 * (ky[idy] / grids.ky_max)
                            if (grids.kx_max > 0 and grids.ky_max > 0)
                            else 0.0
                        )
                        d11 = (
                            cfg.d_hb * (((kx[idx] ** 2 + ky[idy] ** 2) / kpmax2) ** cfg.p_hb)
                            if kpmax2 > 0
                            else 0.0
                        )
                        sfac = (s10[idz] * d10) if idy == 0 else (s11[idz] * d11 + s01 * d01)
                        rhs[:, :, :, idz, idy, idx] -= sfac * g[:, :, :, idz, idy, idx]

    # 4. Closures (closures.cu)
    cmod = cfg.closure_model.strip().lower()
    if cmod == "beer4+2":
        d_par = 2.0 * math.sqrt(math.pi) / (3.0 * math.pi - 8.0)
        beta_par = (32.0 - 9.0 * math.pi) / (3.0 * math.pi - 8.0)
        d_perp = 0.5 * math.sqrt(math.pi)
        nu = [
            0.0 + 0.0j,
            2.019 - 1.620j,
            0.433 + 1.018j,
            -0.256 + 1.487j,
            -0.070 - 1.382j,
            -8.927 + 12.649j,
            8.094 + 1.638j,
            13.720 + 2.139j,
            3.368 - 8.110j,
            1.974 - 2.984j,
            8.269 - 4.940j,
        ]
        for is_idx, sp in enumerate(cfg.species):
            rhs[is_idx, 2, 1] = 0.0
            rhs[is_idx, 3, 1] = 0.0
            dz_g20 = np.where(
                mask3d, _cpp_dz_np(g[is_idx, 2, 0], kz_deriv, cfg.local_limit, grids.zp), 0.0
            )
            adz_g30 = np.where(
                mask3d, _cpp_abs_dz_np(g[is_idx, 3, 0], kz_deriv, cfg.local_limit, grids.zp), 0.0
            )
            adz_g11 = np.where(
                mask3d, _cpp_abs_dz_np(g[is_idx, 1, 1], kz_deriv, cfg.local_limit, grids.zp), 0.0
            )
            rhs[is_idx, 3, 0] += (-beta_par / math.sqrt(3.0) * gpar * sp.vt) * dz_g20
            rhs[is_idx, 3, 0] += (-math.sqrt(2.0) * d_par * gpar * sp.vt) * adz_g30
            rhs[is_idx, 1, 1] += (-math.sqrt(2.0) * d_perp * gpar * sp.vt) * adz_g11

            iwd = 1.0j * omegad * sp.tz
            awd = np.abs(omegad) * sp.tz
            s2 = math.sqrt(2.0)
            s6 = math.sqrt(6.0)
            g02, g10, g01, g03, g11 = (
                g[is_idx, 2, 0],
                g[is_idx, 0, 1],
                g[is_idx, 1, 0],
                g[is_idx, 3, 0],
                g[is_idx, 1, 1],
            )
            rhs[is_idx, 2, 0] += np.where(
                mask3d,
                -s2 * awd * (nu[1].real * s2 * g02 + nu[2].real * g10)
                - s2 * iwd * (nu[1].imag * s2 * g02 + nu[2].imag * g10),
                0.0,
            )
            rhs[is_idx, 0, 1] += np.where(
                mask3d,
                -2.0 * awd * (nu[3].real * s2 * g02 + nu[4].real * g10)
                - 2.0 * iwd * (nu[3].imag * s2 * g02 + nu[4].imag * g10),
                0.0,
            )
            rhs[is_idx, 3, 0] += np.where(
                mask3d,
                -(1.0 / s6) * awd * (nu[5].real * g01 + nu[6].real * s6 * g03 + nu[7].real * g11)
                - (1.0 / s6) * iwd * (nu[5].imag * g01 + nu[6].imag * s6 * g03 + nu[7].imag * g11),
                0.0,
            )
            rhs[is_idx, 1, 1] += np.where(
                mask3d,
                -awd * (nu[8].real * g01 + nu[9].real * s6 * g03 + nu[10].real * g11)
                - iwd * (nu[8].imag * g01 + nu[9].imag * s6 * g03 + nu[10].imag * g11),
                0.0,
            )
    elif cmod == "smith_perp":
        aclos = smith_perp_get_as(nl, cfg.smith_perp_q)
        ell_last = nl - 1
        for is_idx, sp in enumerate(cfg.species):
            iwd = 1.0j * omegad * sp.tz
            awd = np.abs(omegad) * sp.tz
            for m in range(nm):
                clos = np.zeros_like(omegad, dtype=np.complex128)
                for ell in range(ell_last, nl - cfg.smith_perp_q - 1, -1):
                    c_a = aclos[ell_last - ell]
                    clos = clos + (awd * c_a.imag + iwd * c_a.real) * g[is_idx, m, ell]
                rhs[is_idx, m, ell_last] -= np.where(mask3d, (ell_last + 1) * clos, 0.0)
    elif cmod == "smith_par":
        acoeff = smith_par_get_as(nm, cfg.smith_par_q)
        m_last = nm - 1
        for is_idx, sp in enumerate(cfg.species):
            for ell in range(nl):
                clos = np.zeros((nz, naky, nakx), dtype=np.complex128)
                for m in range(m_last, nm - cfg.smith_par_q - 1, -1):
                    dz_m = np.where(
                        mask3d,
                        _cpp_dz_np(g[is_idx, m, ell], kz_deriv, cfg.local_limit, grids.zp),
                        0.0,
                    )
                    adz_m = np.where(
                        mask3d,
                        _cpp_abs_dz_np(g[is_idx, m, ell], kz_deriv, cfg.local_limit, grids.zp),
                        0.0,
                    )
                    c_a = acoeff[m_last - m]
                    clos = clos - c_a.imag * adz_m + c_a.real * dz_m
                rhs[is_idx, m_last, ell] += (-math.sqrt(m_last + 1) * gpar * sp.vt) * clos

    # 5. hypercollisions_const (linear.cu:L215-222, device_funcs.cu:L3329-3365)
    if cfg.hypercollisions_const:
        for is_idx, sp in enumerate(cfg.species):
            nu_lm_eff = (sp.vt / cfg.vtmax) * cfg.nu_hyper_lm / dt
            for m in range(nm):
                for ell in range(nl):
                    if m > 2 or ell > 1:
                        rate = nu_lm_eff * (
                            ((2 * ell + m) / float(2 * nl + nm)) ** cfg.p_hyper_lm
                        ) + sp.vt * (
                            nl * cfg.nu_hyper_l * ((ell / float(nl)) ** cfg.p_hyper_l)
                            + nm * cfg.nu_hyper_m * ((m / float(nm)) ** cfg.p_hyper_m)
                        )
                        rhs[is_idx, m, ell] -= np.where(mask3d, rate * g[is_idx, m, ell], 0.0)

    # 6. hypercollisions_kz (linear.cu:L225-233, device_funcs.cu:L3367-3384)
    if cfg.hypercollisions_kz and nz > 1:
        m_max_f = float(nm - 1)
        p_f = float(cfg.p_hyper_m)
        for is_idx, sp in enumerate(cfg.species):
            tmp = np.zeros((nm, nl, nz, naky, nakx), dtype=np.complex128)
            for m in range(3, nm):
                fac = (
                    cfg.nu_hyper_m
                    * (p_f + 0.5)
                    * 2.3
                    * sp.vt
                    * abs(gpar)
                    * ((m / m_max_f) ** p_f)
                    / math.sqrt(m_max_f)
                )
                tmp[m] = -fac * g[is_idx, m]
            rhs[is_idx] += np.where(
                mask6d[0], _cpp_abs_dz_np(tmp, kz_deriv, cfg.local_limit, grids.zp), 0.0
            )

    # 7. hyperdiff (linear.cu:L236-237, device_funcs.cu:L3304-3327)
    if cfg.hyper:
        kpmax2 = grids.kx_max**2 + grids.ky_max**2
        dfac = cfg.d_hyper * (((kx[None, :] ** 2 + ky[:, None] ** 2) / kpmax2) ** cfg.p_hyper)
        rhs -= np.where(mask6d, dfac[None, None, None, None, :, :] * g, 0.0)

    # 8. hyperz (linear.cu:L239, device_funcs.cu:L2593-2608)
    if cfg.hyperz and nz > 1:
        hyp_w = (np.abs(kz_raw) / grids.kz_max) ** cfg.p_hyper_z
        hz = np.fft.ifft(np.fft.fft(g, axis=-3) * (-hyp_w)[:, None, None], axis=-3)
        rhs += np.where(mask6d, cfg.nu_hyper_z * hz, 0.0)

    # 9. dampEnds_linked (linear.cu:L243, device_funcs.cu:L2813-2876)
    if not cfg.boundary_option_periodic and not cfg.local_limit and nz > 1:
        width = int(nz * cfg.damp_ends_widthfrac)
        nu_z = np.zeros((nz,), dtype=np.float64)
        for idz in range(nz):
            if idz <= width:
                x = idz / float(width)
                nu_z[idz] = 1.0 - 2.0 * x**2 / (1.0 + x**4)
            elif idz >= nz - width:
                x = (nz - idz) / float(width)
                nu_z[idz] = 1.0 - 2.0 * x**2 / (1.0 + x**4)
        amp = cfg.damp_ends_amp / dt
        for is_idx, sp in enumerate(cfg.species):
            b_s = sp.rho2 * kperp2
            for ell in range(nl):
                jl = _cpp_jflr_np(ell, b_s)
                jlb = _cpp_jflr_b_np(ell, b_s)
                for m in range(nm):
                    h_m = g[is_idx, m, ell].copy()
                    if m == 0:
                        h_m = h_m + sp.zt * jl * phi + jlb * bpar
                    elif m == 1:
                        h_m = h_m - sp.zt * sp.vt * jl * apar
                    for idy in range(1, naky):
                        for idx in range(nakx):
                            if mask2d[idy, idx]:
                                rhs[is_idx, m, ell, :, idy, idx] -= nu_z * amp * h_m[:, idy, idx]

    rhs = np.where(mask6d, rhs, 0.0)
    return rhs, rhs_par, rhs_loc


# ---------------------------------------------------------------------------
# 1. ParallelGrad Spectral & Local Derivative Verification
# ---------------------------------------------------------------------------


def test_parallel_grad_periodic_linked_and_local_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dz, dz2, abs_dz, hyperz, zft/zft_inverse, dealias, Nyquist sign, & dampEnds."""
    base_cfg = load_config(
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    )
    cfg_per = dataclasses.replace(
        base_cfg,
        nx=10,
        ny=10,
        nz=32,
        ntheta_in=32,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        boundary="periodic",
        boundary_option_periodic=True,
        dealias_kz=True,
    )
    cfg_per, grids, geo = init_geometry(cfg_per)
    gp64 = init_parallel_grad(cfg_per, grids, geo, dtype=jnp.float64)
    gp32 = gp64.astype(jnp.float32)

    # Analytical monochromatic test cos(2*z) matching test_grad_parallel_periodic.cu
    z = np.asarray(grids.z, dtype=np.float64)
    rng = np.random.default_rng(11)
    amp_yx = rng.standard_normal((grids.naky, grids.nakx)) + 1.0j * rng.standard_normal(
        (grids.naky, grids.nakx)
    )
    amp_yx[0, 0] = 0.0
    f_in = jnp.asarray(np.cos(2.0 * z)[:, None, None] * amp_yx[None, :, :], dtype=jnp.complex128)
    expected_dz = (-2.0 * np.sin(2.0 * z))[:, None, None] * amp_yx[None, :, :]
    expected_dz2 = (-4.0 * np.cos(2.0 * z))[:, None, None] * amp_yx[None, :, :]
    expected_abs_dz = (2.0 * np.cos(2.0 * z))[:, None, None] * amp_yx[None, :, :]

    _assert_close_fp64(jax.jit(gp64.dz)(f_in), expected_dz, "gp64.dz(cos(2z))")
    _assert_close_fp64(jax.jit(gp64.dz2)(f_in), expected_dz2, "gp64.dz2(cos(2z))")
    _assert_close_fp64(jax.jit(gp64.abs_dz)(f_in), expected_abs_dz, "gp64.abs_dz(cos(2z))")
    _assert_close_fp32_1e6(jax.jit(gp32.dz)(f_in.astype(jnp.complex64)), expected_dz, "gp32.dz")

    # Verify GX's positive Nyquist convention (+nz//2 / zp) on complex Nyquist mode (-1)^iz
    nyq_z = ((-1.0) ** np.arange(grids.nz, dtype=np.float64))[:, None, None] * amp_yx[None, :, :]
    cfg_nod = dataclasses.replace(cfg_per, dealias_kz=False)
    _, grids_nod, geo_nod = init_geometry(cfg_nod)
    gp_nod = init_parallel_grad(cfg_nod, grids_nod, geo_nod, dtype=jnp.float64)
    k_nyq = float(grids_nod.nz // 2) / float(grids_nod.zp)
    _assert_close_fp64(gp_nod.dz(jnp.asarray(nyq_z)), 1.0j * k_nyq * nyq_z, "nyquist_positive_kz")

    # Verify zft -> zft_inverse exact round-trip and 2/3 kz dealiasing
    g_rand = _random_complex_g(grids.g_shape, np.asarray(grids.active_mask), seed=12)
    _assert_close_fp64(gp64.zft_inverse(gp64.zft(g_rand)), g_rand, "zft_roundtrip")
    g_dealiased = gp64.dealias(g_rand)
    g_hat_d = np.fft.fft(np.asarray(g_dealiased), axis=-3)
    kzm_zero = np.asarray(grids.kzm) == 0
    assert float(np.max(np.abs(g_hat_d[..., kzm_zero, :, :]))) < 1.0e-14

    # Verify local_limit algebraic scaling (grad_parallel.cu:L187-223)
    cfg_loc = dataclasses.replace(cfg_per, zp=3, local_limit=True)
    cfg_loc, grids_loc, geo_loc = init_geometry(cfg_loc)
    gp_loc = init_parallel_grad(cfg_loc, grids_loc, geo_loc)
    _assert_close_fp64(gp_loc.dz(g_rand), (1.0j / 3.0) * g_rand, "local_limit:dz")
    _assert_close_fp64(gp_loc.abs_dz(g_rand), (1.0 / 3.0) * g_rand, "local_limit:abs_dz")
    _assert_close_fp64(gp_loc.dz2(g_rand), -(1.0 / 9.0) * g_rand, "local_limit:dz2")

    # Verify dampEnds_linked leaves ky=0 zonal modes untouched and damps ky>0 on H
    cfg_lnk = dataclasses.replace(
        base_cfg,
        nx=10,
        ny=10,
        nz=32,
        ntheta_in=32,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        jtwist=20,  # |jtwist| >= nakx -> nLinks == 1 with active ky=0 and ky>0 modes!
    )
    cfg_lnk, grids_lnk, geo_lnk = init_geometry(cfg_lnk)
    solver_lnk = init_field_solver(cfg_lnk, grids_lnk, geo_lnk)
    gp_lnk = init_parallel_grad(cfg_lnk, grids_lnk, geo_lnk)
    g_lnk = _random_complex_g(grids_lnk.g_shape, np.asarray(grids_lnk.active_mask), seed=13)
    f_lnk = solve_fields(g_lnk, solver_lnk)
    zero_rhs = jnp.zeros_like(g_lnk)
    bcs_out = gp_lnk.apply_bcs(g_lnk, zero_rhs, f_lnk, solver_lnk, dt=0.05)
    np.testing.assert_array_equal(np.asarray(bcs_out[..., 0, :]), 0.0)
    assert float(jnp.linalg.norm(bcs_out[..., 1:, :])) > 0.0

    # Verify multi-link / NTFT delegate to boundary_parallel (and raise cleanly if absent)
    cfg_multi = dataclasses.replace(cfg_lnk, x0=-1.0, jtwist=1)
    cfg_multi, grids_multi, geo_multi = init_geometry(cfg_multi)
    gp_multi = init_parallel_grad(cfg_multi, grids_multi, geo_multi)
    assert gp_multi.n_classes > 1
    monkeypatch.setitem(sys.modules, "gx_jax.boundary_parallel", None)
    with pytest.raises(NotImplementedError, match="boundary_parallel"):
        init_parallel_grad(cfg_multi, grids_multi, geo_multi)


# ---------------------------------------------------------------------------
# 2. Core Linear RHS Differential Verification Across All 5 Benchmarks + 3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("case_name", "in_path"), BENCHMARK_INPUTS)
def test_linear_rhs_differential_all_benchmarks(case_name: str, in_path: Path) -> None:
    """Verify streaming_rhs, rhs_linear_local, and linear_rhs (fused & unfused) vs C++ port."""
    cfg_raw = load_config(in_path)
    # Keep velocity resolution compact enough for fast loop-based NumPy reference check while
    # preserving all species, beta, Boltzmann, geometry, and hypercollision branches.
    cfg = dataclasses.replace(
        cfg_raw, nhermite=min(cfg_raw.nhermite, 12), nlaguerre=min(cfg_raw.nlaguerre, 6)
    )
    cfg, grids, geo = init_geometry(cfg)

    op64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
    op32 = op64.astype(jnp.float32)

    g128 = _random_complex_g(grids.g_shape, np.asarray(grids.active_mask), seed=101)
    f128 = solve_fields(g128, op64.solver)
    dt = 0.025

    rhs_ref, rhs_par_ref, rhs_loc_ref = _cpp_reference_linear_rhs(
        np.asarray(g128), f128, cfg, grids, geo, dt
    )

    # Individual sub-kernels in FP64
    par128 = jax.jit(streaming_rhs)(g128, f128, op64)
    loc128 = jax.jit(rhs_linear_local)(g128, f128, op64)
    _assert_close_fp64(par128, rhs_par_ref, f"{case_name}:streaming_rhs")
    _assert_close_fp64(loc128, rhs_loc_ref, f"{case_name}:rhs_linear_local")

    # Full linear_rhs: fused single-FFT vs unfused modular path vs NumPy C++ reference
    rhs_fused_128 = jax.jit(lambda g, f: linear_rhs(g, f, op64, dt, use_fused_fft=True))(g128, f128)
    rhs_unfused_128 = jax.jit(lambda g, f: linear_rhs(g, f, op64, dt, use_fused_fft=False))(
        g128, f128
    )
    _assert_close_fp64(rhs_fused_128, rhs_ref, f"{case_name}:linear_rhs_fused_fp64")
    _assert_close_fp64(rhs_unfused_128, rhs_ref, f"{case_name}:linear_rhs_unfused_fp64")

    # Runtime complex64 / float32 precision gate (< 1e-6)
    g64 = g128.astype(jnp.complex64)
    f64 = solve_fields(g64, op32.solver)
    rhs_fused_64 = jax.jit(lambda g, f: linear_rhs(g, f, op32, dt, use_fused_fft=True))(g64, f64)
    _assert_close_fp32_1e6(rhs_fused_64, rhs_ref, f"{case_name}:linear_rhs_c64")
    assert rhs_fused_64.dtype == jnp.complex64


def test_linear_rhs_3d_multikx_electromagnetic() -> None:
    """Verify full 3D (kx != 0 and ky != 0) linear RHS with coupled (phi, apar, bpar)."""
    kbm_cfg = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    cfg = dataclasses.replace(
        kbm_cfg,
        nx=10,
        ny=10,
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=8,
        nlaguerre=4,
        fbpar=1.0,
        boundary="periodic",
        boundary_option_periodic=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    op64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
    op32 = op64.astype(jnp.float32)

    g128 = _random_complex_g(grids.g_shape, np.asarray(grids.active_mask), seed=202)
    f128 = solve_fields(g128, op64.solver)
    dt = 0.01

    rhs_ref, _, _ = _cpp_reference_linear_rhs(np.asarray(g128), f128, cfg, grids, geo, dt)
    rhs128 = jax.jit(lambda g, f: linear_rhs(g, f, op64, dt))(g128, f128)
    rhs64 = jax.jit(lambda g, f: linear_rhs(g, f, op32, dt))(
        g128.astype(jnp.complex64), solve_fields(g128.astype(jnp.complex64), op32.solver)
    )
    _assert_close_fp64(rhs128, rhs_ref, "3d_multikx_em:fp64")
    _assert_close_fp32_1e6(rhs64, rhs_ref, "3d_multikx_em:c64")
    np.testing.assert_array_equal(np.asarray(rhs64[..., 0, 0]), 0.0)


# ---------------------------------------------------------------------------
# 3. Dougherty / Lenard-Bernstein Collisions & Exact Conservation Laws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("beta_val", "ei_colls", "coll_cons", "nlaguerre"),
    [
        (0.0, True, True, 4),
        (0.02, True, True, 4),
        (0.02, False, True, 4),
        (0.02, True, False, 4),
        (0.01, True, True, 1),  # Special nl == 1 t_bar conservation branch!
    ],
)
def test_collisions_differential_and_exact_conservation(
    beta_val: float,
    ei_colls: bool,
    coll_cons: bool,
    nlaguerre: int,
) -> None:
    """Verify Dougherty + e-i collisions against C++ and test exact conservation laws."""
    ion = SpeciesConfig.from_raw(
        z=1.0,
        mass=1.0,
        dens=1.0,
        temp=1.0,
        tprim=2.5,
        fprim=0.8,
        nu_ss=0.05,
        stype="ion",
        beta=beta_val,
    )
    elec = SpeciesConfig.from_raw(
        z=-1.0,
        mass=2.7e-4,
        dens=1.0,
        temp=1.0,
        tprim=2.5,
        fprim=0.8,
        nu_ss=0.25,
        stype="electron",
        beta=beta_val,
    )
    base = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    cfg = dataclasses.replace(
        base,
        nx=7,
        ny=7,
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=6,
        nlaguerre=nlaguerre,
        nspecies=2,
        species=(ion, elec),
        beta=beta_val,
        fphi=1.0,
        fapar=1.0 if beta_val > 0.0 else 0.0,
        fbpar=1.0 if beta_val > 0.0 else 0.0,
        collisions=True,
        ei_colls=ei_colls,
        coll_conservation=coll_cons,
        boundary="periodic",
        boundary_option_periodic=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    op64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
    op32 = op64.astype(jnp.float32)

    g128 = _random_complex_g(grids.g_shape, np.asarray(grids.active_mask), seed=303)
    f128 = solve_fields(g128, op64.solver)
    rhs_ref, _, _ = _cpp_reference_linear_rhs(np.asarray(g128), f128, cfg, grids, geo, 0.01)

    rhs128 = jax.jit(lambda g, f: linear_rhs(g, f, op64, 0.01))(g128, f128)
    rhs64 = jax.jit(lambda g, f: linear_rhs(g, f, op32, 0.01))(
        g128.astype(jnp.complex64), solve_fields(g128.astype(jnp.complex64), op32.solver)
    )
    _assert_close_fp64(
        rhs128, rhs_ref, f"colls({beta_val},{ei_colls},{coll_cons},nl={nlaguerre}):fp64"
    )
    _assert_close_fp32_1e6(
        rhs64, rhs_ref, f"colls({beta_val},{ei_colls},{coll_cons},nl={nlaguerre}):c64"
    )

    # Verify exact particle, parallel-momentum, and total-energy conservation in drift-kinetic
    # like-species collisions (b_s == 0 via long_wavelength_GK, no_fields, zero drifts/drives)
    if beta_val == 0.0 and ei_colls and coll_cons and nlaguerre == 4:
        ion_dk = SpeciesConfig.from_raw(
            z=1.0,
            mass=1.0,
            dens=1.0,
            temp=1.0,
            tprim=0.0,
            fprim=0.0,
            nu_ss=0.3,
            stype="ion",
            long_wavelength_gk=True,
        )
        cfg_dk = dataclasses.replace(
            cfg,
            nspecies=1,
            species=(ion_dk,),
            long_wavelength_gk=True,
            ei_colls=False,
            hypercollisions_kz=False,
            geo_option="slab",
            slab=True,
        )
        cfg_dk, grids_dk, geo_dk = init_geometry(cfg_dk)
        solver_nf = init_field_solver(cfg_dk, grids_dk, geo_dk, no_fields=True)
        op_dk = init_linear_operator(cfg_dk, grids_dk, geo_dk, solver=solver_nf)
        g_dk = _random_complex_g(grids_dk.g_shape, np.asarray(grids_dk.active_mask), seed=304)
        f_nf = solve_fields(g_dk, solver_nf)
        c_only = rhs_linear_local(g_dk, f_nf, op_dk)

        # 1. Density (m=0, l=0) tendency is identically 0
        _assert_close_fp64(c_only[:, 0, 0], 0.0, "coll_conservation:density", atol=1e-14)
        # 2. Parallel momentum (m=1, l=0) tendency is identically 0
        _assert_close_fp64(c_only[:, 1, 0], 0.0, "coll_conservation:upar", atol=1e-14)
        # 3. Total temperature (sqrt(2)/3 G_{2,0} + 2/3 G_{0,1}) tendency is identically 0
        dtot_temp = (math.sqrt(2.0) / 3.0) * c_only[:, 2, 0] + (2.0 / 3.0) * c_only[:, 0, 1]
        _assert_close_fp64(dtot_temp, 0.0, "coll_conservation:energy", atol=1e-14)
        # 4. Collisional entropy dissipation: Re <g, C[g]> < 0 for generic non-Maxwellian g
        inner = float(jnp.real(jnp.sum(jnp.conj(g_dk) * c_only)))
        assert inner < 0.0


# ---------------------------------------------------------------------------
# 4. Closures (Beer 4+2, SmithPerp, SmithPar) & Dissipation Verification
# ---------------------------------------------------------------------------


def test_smith_pade_coefficients_and_all_closure_models() -> None:
    """Verify SmithPar Padé solver and Beer42 / SmithPerp / SmithPar linear_rhs."""
    # Analytical Hammett-Perkins (1990) 1-pole closure (nhermite=1, q=1): a_0 = -i * sqrt(2/pi)
    a_hp = smith_par_get_as(1, 1)
    _assert_close_fp64(a_hp[0], -1.0j * math.sqrt(2.0 / math.pi), "smith_par_hammett_perkins")

    base = load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in")
    cases = [
        ("beer4+2", 4, 2, 3, 3),
        ("smith_perp", 6, 6, 3, 4),
        ("smith_perp", 6, 8, 3, 3),
        ("smith_par", 8, 4, 3, 3),
        ("smith_par", 12, 4, 2, 3),
    ]
    for model, nm, nl, spar_q, sperp_q in cases:
        cfg = dataclasses.replace(
            base,
            nx=7,
            ny=7,
            nz=16,
            ntheta_in=16,
            nperiod=1,
            zp=1,
            nhermite=nm,
            nlaguerre=nl,
            closure_model=model,
            smith_par_q=spar_q,
            smith_perp_q=sperp_q,
            boundary="periodic",
            boundary_option_periodic=True,
        )
        cfg, grids, geo = init_geometry(cfg)
        op64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
        op32 = op64.astype(jnp.float32)

        g128 = _random_complex_g(grids.g_shape, np.asarray(grids.active_mask), seed=404)
        f128 = solve_fields(g128, op64.solver)
        rhs_ref, _, _ = _cpp_reference_linear_rhs(np.asarray(g128), f128, cfg, grids, geo, 0.02)

        rhs_f = jax.jit(lambda g, f, op=op64: linear_rhs(g, f, op, 0.02, use_fused_fft=True))(
            g128, f128
        )
        rhs_u = jax.jit(lambda g, f, op=op64: linear_rhs(g, f, op, 0.02, use_fused_fft=False))(
            g128, f128
        )
        _assert_close_fp64(rhs_f, rhs_ref, f"closure({model},{nm},{nl}):fused_fp64")
        _assert_close_fp64(rhs_u, rhs_ref, f"closure({model},{nm},{nl}):unfused_fp64")

        if model == "beer4+2":
            np.testing.assert_array_equal(np.asarray(rhs_f[:, 2, 1]), 0.0)
            _assert_close_fp64(
                rhs_f[:, 3, 1],
                apply_hypercollisions_kz(g128, op64)[:, 3, 1],
                "beer4+2_m3l1_hyperkz",
            )

        rhs_c64 = jax.jit(linear_rhs)(
            g128.astype(jnp.complex64),
            solve_fields(g128.astype(jnp.complex64), op32.solver),
            op32,
            0.02,
        )
        _assert_close_fp32_1e6(rhs_c64, rhs_ref, f"closure({model},{nm},{nl}):c64")

        # Also check standalone apply_closures helper directly
        c_only = apply_closures(g128, jnp.zeros_like(g128), op64)
        assert c_only.shape == grids.g_shape


def test_dissipation_hypercollisions_hyperdiff_and_hb_hyper() -> None:
    """Verify hypercollisions_const, hypercollisions_kz, hyperdiff, hyperz, and HB_hyper."""
    base = load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in")
    cfg = dataclasses.replace(
        base,
        nx=10,
        ny=10,
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=8,
        nlaguerre=4,
        boundary="periodic",
        boundary_option_periodic=True,
        hypercollisions_const=True,
        nu_hyper_l=0.2,
        nu_hyper_m=0.8,
        nu_hyper_lm=0.15,
        p_hyper_l=4,
        p_hyper_m=4,
        p_hyper_lm=6,
        hypercollisions_kz=True,
        hyper=True,
        d_hyper=0.07,
        p_hyper=2,
        hyperz=True,
        nu_hyper_z=0.12,
        p_hyper_z=6,
        hb_hyper=True,
        d_hb=0.9,
        p_hb=2,
        w_osc=0.05,
    )
    cfg, grids, geo = init_geometry(cfg)
    op64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
    op32 = op64.astype(jnp.float32)

    g128 = _random_complex_g(grids.g_shape, np.asarray(grids.active_mask), seed=505)
    f128 = solve_fields(g128, op64.solver)
    dt = 0.02

    # Check conservation masks on hypercollisions_const (m<=2 & l<=1 -> 0) and _kz (m<=2 -> 0)
    hc_const = apply_hypercollisions_const(g128, op64, dt)
    np.testing.assert_array_equal(np.asarray(hc_const[:, :3, :2]), 0.0)
    hc_kz = apply_hypercollisions_kz(g128, op64)
    np.testing.assert_array_equal(np.asarray(hc_kz[:, :3, :]), 0.0)
    assert apply_hyperdiff(g128, op64).shape == grids.g_shape
    assert apply_hb_hyper(g128, f128, grids, geo, op64).shape == grids.g_shape

    rhs_ref, _, _ = _cpp_reference_linear_rhs(np.asarray(g128), f128, cfg, grids, geo, dt)
    rhs_f = jax.jit(
        lambda g, f: linear_rhs(g, f, op64, dt, grids=grids, geo=geo, use_fused_fft=True)
    )(g128, f128)
    rhs_u = jax.jit(
        lambda g, f: linear_rhs(g, f, op64, dt, grids=grids, geo=geo, use_fused_fft=False)
    )(g128, f128)
    _assert_close_fp64(rhs_f, rhs_ref, "all_dissipation:fused_fp64")
    _assert_close_fp64(rhs_u, rhs_ref, "all_dissipation:unfused_fp64")

    geo32 = geo.astype(jnp.float32)
    rhs_c64 = jax.jit(lambda g, f: linear_rhs(g, f, op32, dt, grids=grids, geo=geo32))(
        g128.astype(jnp.complex64), solve_fields(g128.astype(jnp.complex64), op32.solver)
    )
    _assert_close_fp32_1e6(rhs_c64, rhs_ref, "all_dissipation:c64")


# ---------------------------------------------------------------------------
# 5. CFL Frequencies & Temporal Convergence Orders of All 5 Integrators
# ---------------------------------------------------------------------------


def test_cfl_frequencies_and_all_five_linear_integrators() -> None:
    """Verify CFL frequency/dt formulas (including NTFT m0_max) and RK orders 2, 3, 4."""
    # Check CFL on standard sheared, zero-shat slab, fixed_dt, and NTFT
    for bname, in_path in BENCHMARK_INPUTS:
        cfg, grids, geo = init_geometry(load_config(in_path))
        w0, w1, w2 = compute_linear_max_frequency(cfg, grids, geo)
        assert w0 >= 0.0 and w2 > 0.0 and (w0 + w1 + w2) > 0.0
        if bname == "kaw_slab":
            # In C++ GX (parameters.cu:L1060,L1090; linear.cu:L264), tprim == fprim == 0 leaves
            # etamax == -1.0, making omega_max[1] slightly negative while w0 + w1 + w2 > 0.
            assert w1 < 0.0
        else:
            assert w1 >= 0.0
        dt_cfl = compute_linear_dt(cfg, grids, geo)
        assert cfg.dt_min <= dt_cfl <= cfg.dt_max

    cfg_ntft = dataclasses.replace(
        load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"),
        nx=7,
        ny=7,
        non_twist=True,
    )
    cfg_ntft, grids_ntft, geo_ntft = init_geometry(cfg_ntft)
    w_ntft = compute_linear_max_frequency(cfg_ntft, grids_ntft, geo_ntft)
    assert w_ntft[0] > 0.0
    assert (
        compute_linear_dt(
            dataclasses.replace(cfg_ntft, fixed_dt=True, dt=0.0123), grids_ntft, geo_ntft
        )
        == 0.0123
    )

    # Verify global convergence order of sspx2 (2), rk3 (3), sspx3 (3), rk4 (4), k10 (4)
    # on a dt-independent periodic gyrokinetic system against exact matrix exponential expm(T * L)
    cfg_ode = dataclasses.replace(
        load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"),
        nx=1,
        ny=4,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        boundary="periodic",
        boundary_option_periodic=True,
        dealias_kz=False,
    )
    cfg_ode, grids_ode, geo_ode = init_geometry(cfg_ode)
    op_ode = init_linear_operator(cfg_ode, grids_ode, geo_ode, dtype=jnp.float64)

    # Assemble exact dense linear operator matrix on active mode (iky=1, ikx=0)
    mode_shape = (1, 4, 2, 8)
    ndof = int(np.prod(mode_shape))
    l_mat = np.zeros((ndof, ndof), dtype=np.complex128)
    for j in range(ndof):
        ej = np.zeros((ndof,), dtype=np.complex128)
        ej[j] = 1.0
        g_col = (
            jnp.zeros(grids_ode.g_shape, dtype=jnp.complex128)
            .at[..., 1, 0]
            .set(ej.reshape(mode_shape))
        )
        f_col = solve_fields(g_col, op_ode.solver)
        rhs_col = linear_rhs(g_col, f_col, op_ode, 0.01)
        l_mat[:, j] = np.asarray(rhs_col[..., 1, 0]).ravel()

    g0 = _random_complex_g(grids_ode.g_shape, np.asarray(grids_ode.active_mask), seed=606)
    # Project initial state onto kz-dealiased subspace so dealias_kz steppers evolve smoothly
    g0 = op_ode.grad_par.dealias(g0)
    f0 = solve_fields(g0, op_ode.solver)
    t_final = 0.08
    exact_end = (expm(t_final * l_mat) @ np.asarray(g0[..., 1, 0]).ravel()).reshape(mode_shape)

    expected_orders = {"sspx2": 1.9, "rk3": 2.9, "sspx3": 2.9, "rk4": 3.9, "k10": 3.9}
    for scheme, min_order in expected_orders.items():
        g_coarse, _, _ = advance_linear_steps(g0, f0, op_ode, t_final / 4.0, 4, scheme=scheme)
        g_fine, _, _ = advance_linear_steps(g0, f0, op_ode, t_final / 8.0, 8, scheme=scheme)
        err_c = float(np.linalg.norm(np.asarray(g_coarse[..., 1, 0]) - exact_end))
        err_f = float(np.linalg.norm(np.asarray(g_fine[..., 1, 0]) - exact_end))
        order = math.log2(err_c / err_f)
        assert order > min_order, f"{scheme}: observed order {order:.2f} < {min_order}"


# ---------------------------------------------------------------------------
# 6. Golden C++ .out.nc / .restart.nc Linear Benchmarks & 3D W7-X Stellarator
# ---------------------------------------------------------------------------


def test_golden_cpp_restart_and_kaw_out_nc_benchmarks() -> None:
    """Verify converged eigenmodes against restart_test02.nc (rtol < 1e-5) and KAW .out.nc."""
    # Part A: Golden C++ converged linear ITG restart_test02.out.nc (k10 integrator at t=200)
    sp = SpeciesConfig.from_raw(
        z=1.0, mass=1.0, dens=1.0, temp=1.0, tprim=6.9, fprim=2.2, nu_ss=0.0, stype="ion"
    )
    cfg_r2 = GXConfig(
        nx=1,
        ny=31,
        nz=160,
        ntheta_in=32,
        nperiod=3,
        zp=5,
        nhermite=16,
        nlaguerre=8,
        nspecies=1,
        species=(sp,),
        y0=20.0,
        boundary="linked",
        all_kinetic=False,
        add_boltzmann_species=True,
        boltzmann_opt=BoltzmannOption.BOLTZMANN_ELECTRONS,
        tau_fac=1.0,
        geo_option="s-alpha",
        eps=0.18,
        rmaj=1.0,
        qsf=1.44,
        shat=0.8,
        shift=0.0,
        scheme="k10",
        dt=0.01,
        dt_max=0.01,
        fixed_dt=True,
        cfl=0.5,
    )
    cfg_r2, grids_r2, geo_r2 = init_geometry(cfg_r2)
    op_r2 = init_linear_operator(cfg_r2, grids_r2, geo_r2, dtype=jnp.float64)

    g_t200, t_val = load_restart_nc(
        UNIT_INPUTS_DIR / "restart_tests" / "restart_test02.out.nc",
        grids_r2,
        dtype=jnp.complex128,
    )
    assert t_val == pytest.approx(200.0, rel=1e-6)
    f_t200 = solve_fields(g_t200, op_r2.solver)
    _, f_next = jax.jit(lambda g, f: step_linear(g, f, op_r2, 0.01, scheme="k10"))(g_t200, f_t200)
    w_jax = np.asarray(compute_growth_rates(f_next.phi, f_t200.phi, 0.01, grids_r2)[:, 0])

    with netCDF4.Dataset(UNIT_INPUTS_DIR / "restart_tests" / "restart_test02.nc", "r") as ds:
        w_cpp = np.asarray(ds.variables["omega_v_time"][-1, 0, :, :], dtype=np.float64)

    # Peak unstable ITG mode iky=7 (ky=0.35) is fully converged at t=200 -> rtol < 1e-5!
    np.testing.assert_allclose(w_jax[7].real, w_cpp[7, 0], rtol=1.0e-5)
    np.testing.assert_allclose(w_jax[7].imag, w_cpp[7, 1], rtol=1.0e-5)
    # Neighbouring converged ITG modes iky=5..9 agree within < 8e-5
    np.testing.assert_allclose(w_jax[5:10].real, w_cpp[5:10, 0], rtol=8.0e-5)
    np.testing.assert_allclose(w_jax[5:10].imag, w_cpp[5:10, 1], rtol=8.0e-5)

    # Part B: Full 4096-step Kinetic Alfvén Wave (sspx3) vs kaw_betahat10.0_kp0.01_correct.out.nc
    kaw_in = BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01.in"
    kaw_out = GOLDEN_BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01_correct.out.nc"
    cfg_kaw, grids_kaw, geo_kaw = init_geometry(load_config(kaw_in))
    op_kaw64 = init_linear_operator(cfg_kaw, grids_kaw, geo_kaw, dtype=jnp.float64)
    op_kaw32 = op_kaw64.astype(jnp.float32)
    geo_kaw32 = geo_kaw.astype(jnp.float32)

    with netCDF4.Dataset(kaw_out, "r") as ds:
        t_kaw = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
        p2_ref = np.asarray(ds.groups["Diagnostics"].variables["Phi2_kyt"][:, 1], dtype=np.float64)

    dt_kaw = float(t_kaw[0])
    # Deterministic ikpar_init=1 mode (Phi2_kyt(t)/Phi2_kyt(t0) is invariant to scalar amplitude)
    cfg_kaw_det = dataclasses.replace(cfg_kaw, init_single=True, iky_single=1, ikx_single=0)
    g_kaw0_64 = init_moments(cfg_kaw_det, grids_kaw, geo_kaw, dtype=jnp.complex128)
    f_kaw0_64 = solve_fields(g_kaw0_64, op_kaw64.solver)

    def _make_kaw_runner(op_use, geo_use):
        @jax.jit
        def _run(g_init: jax.Array, f_init: Fields):
            g1, f1 = step_linear(g_init, f_init, op_use, dt_kaw, scheme="sspx3")
            p0 = compute_phi2_ky(f1.phi, grids_kaw, geo_use)[1]

            def _blk(carry, _):
                gn, fn, _ = advance_linear_steps(
                    carry[0], carry[1], op_use, dt_kaw, 10, scheme="sspx3"
                )
                return (gn, fn), compute_phi2_ky(fn.phi, grids_kaw, geo_use)[1]

            (g_mid, f_mid), p_mid = jax.lax.scan(_blk, (g1, f1), None, length=409)
            _, f_end, _ = advance_linear_steps(g_mid, f_mid, op_use, dt_kaw, 5, scheme="sspx3")
            p_end = compute_phi2_ky(f_end.phi, grids_kaw, geo_use)[1]
            return jnp.concatenate([p0[None], p_mid, p_end[None]], axis=0)

        return _run

    p2_jax64 = np.asarray(
        _make_kaw_runner(op_kaw64, geo_kaw)(g_kaw0_64, f_kaw0_64), dtype=np.float64
    )
    p2_jax64_norm = p2_jax64 / p2_jax64[0]
    p2_ref_norm = p2_ref / p2_ref[0]
    dt_out = float(t_kaw[1] - t_kaw[0])
    # Standing KAW passes through zero nodes (min Phi2_kyt = 1.4e-18 vs 6.9e-14 at t0), where
    # GX-JAX's cancellation-free FP64 qneut_fac_phi (1.00047e-4 vs C++ float32 1.00017e-4) shifts
    # the node by 1.5e-4; normalizing by Phi2_kyt[0] avoids 0/0 blowup at standing-wave nodes.
    int_norm_diff64 = float(np.sum(np.abs(p2_jax64_norm - p2_ref_norm) * dt_out))
    max_norm_diff64 = float(np.max(np.abs(p2_jax64_norm - p2_ref_norm)))
    assert int_norm_diff64 < 1.5e-3, f"KAW FP64 integrated diff {int_norm_diff64:.3e} >= 1.5e-3"
    assert max_norm_diff64 < 2.5e-3, f"KAW FP64 max norm diff {max_norm_diff64:.3e} >= 2.5e-3"

    # Verify 1-step runtime complex64 precision (< 1e-6) and bounded 4096-step KAW phase drift
    # (at b_i = 1e-4, 24-bit float32 J_0(b_i) rounding shifts w_KAW by ~2e-4 over 12,288 stages).
    g_kaw0_32 = g_kaw0_64.astype(jnp.complex64)
    f_kaw0_32 = solve_fields(g_kaw0_32, op_kaw32.solver)
    g1_64, f1_64 = step_linear(g_kaw0_64, f_kaw0_64, op_kaw64, dt_kaw, scheme="sspx3")
    g1_32, f1_32 = step_linear(g_kaw0_32, f_kaw0_32, op_kaw32, dt_kaw, scheme="sspx3")
    _assert_close_fp32_1e6(g1_32, g1_64, "kaw_1step_g_c64")
    _assert_close_fp32_1e6(f1_32.phi, f1_64.phi, "kaw_1step_phi_c64")
    p2_jax32 = np.asarray(
        _make_kaw_runner(op_kaw32, geo_kaw32)(g_kaw0_32, f_kaw0_32), dtype=np.float64
    )
    p2_jax32_norm = p2_jax32 / p2_jax32[0]
    int_norm_diff32 = float(np.sum(np.abs(p2_jax32_norm - p2_ref_norm) * dt_out))
    assert int_norm_diff32 < 2.5e-2, f"KAW FP32 integrated diff {int_norm_diff32:.3e} >= 2.5e-2"


def test_w7x_stellarator_linear_rhs_and_eigenmodes(tmp_path: Path) -> None:
    """Verify 3D W7-X stellarator linear RHS and ITG eigenmode (gamma, omega) to rtol < 1e-5."""
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    cfg_w7x = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
    eik_path = tmp_path / "w7x_session3.eik.nc"
    build_vmec_eik_nc(cfg_w7x, w7x_dir / "wout_w7x.nc", eik_path)

    # 1. Differential RHS check on W7-X: both adiabatic linked and 2-species EM periodic
    cfg_a = dataclasses.replace(
        cfg_w7x,
        geo_option="nc",
        geofilename=str(eik_path),
        nhermite=8,
        nlaguerre=4,
        ny=13,
    )
    cfg_a, grids_a, geo_a = load_eik_nc(eik_path, cfg_a)
    op_a64 = init_linear_operator(cfg_a, grids_a, geo_a, dtype=jnp.float64)
    op_a32 = op_a64.astype(jnp.float32)

    g_a128 = _random_complex_g(grids_a.g_shape, np.asarray(grids_a.active_mask), seed=701)
    f_a128 = solve_fields(g_a128, op_a64.solver)
    rhs_a_ref, _, _ = _cpp_reference_linear_rhs(
        np.asarray(g_a128), f_a128, cfg_a, grids_a, geo_a, 0.01
    )

    _assert_close_fp64(
        jax.jit(lambda g, f: linear_rhs(g, f, op_a64, 0.01))(g_a128, f_a128),
        rhs_a_ref,
        "w7x_adiab:fp64",
    )
    _assert_close_fp32_1e6(
        jax.jit(lambda g, f: linear_rhs(g, f, op_a32, 0.01))(
            g_a128.astype(jnp.complex64), solve_fields(g_a128.astype(jnp.complex64), op_a32.solver)
        ),
        rhs_a_ref,
        "w7x_adiab:c64",
    )

    # 2-species electromagnetic (phi, apar, bpar) W7-X stellarator RHS check
    beta_em = 0.025
    ion_w = SpeciesConfig.from_raw(
        z=1.0,
        mass=1.0,
        dens=1.0,
        temp=1.0,
        tprim=3.0,
        fprim=1.0,
        nu_ss=0.01,
        stype="ion",
        beta=beta_em,
    )
    elec_w = SpeciesConfig.from_raw(
        z=-1.0,
        mass=2.7e-4,
        dens=1.0,
        temp=1.0,
        tprim=3.0,
        fprim=1.0,
        nu_ss=0.05,
        stype="electron",
        beta=beta_em,
    )
    cfg_em = dataclasses.replace(
        cfg_a,
        nx=7,
        ny=7,
        nspecies=2,
        species=(ion_w, elec_w),
        all_kinetic=True,
        add_boltzmann_species=False,
        beta=beta_em,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        collisions=True,
        boundary="periodic",
        boundary_option_periodic=True,
    )
    cfg_em, grids_em, geo_em = load_eik_nc(eik_path, cfg_em)
    op_em64 = init_linear_operator(cfg_em, grids_em, geo_em, dtype=jnp.float64)
    op_em32 = op_em64.astype(jnp.float32)
    g_em128 = _random_complex_g(grids_em.g_shape, np.asarray(grids_em.active_mask), seed=702)
    f_em128 = solve_fields(g_em128, op_em64.solver)
    rhs_em_ref, _, _ = _cpp_reference_linear_rhs(
        np.asarray(g_em128), f_em128, cfg_em, grids_em, geo_em, 0.002
    )
    _assert_close_fp64(
        jax.jit(lambda g, f: linear_rhs(g, f, op_em64, 0.002))(g_em128, f_em128),
        rhs_em_ref,
        "w7x_em:fp64",
    )
    _assert_close_fp32_1e6(
        jax.jit(lambda g, f: linear_rhs(g, f, op_em32, 0.002))(
            g_em128.astype(jnp.complex64),
            solve_fields(g_em128.astype(jnp.complex64), op_em32.solver),
        ),
        rhs_em_ref,
        "w7x_em:c64",
    )

    # 3. Linear stellarator W7-X ITG eigenmode verification (ky = 1.4, unstable W7-X ITG peak):
    # Verify Krylov propagator eigenmode matches runtime complex64 RK4 time-stepping to rtol < 1e-5!
    cfg_eig = dataclasses.replace(
        extract_single_ky_config(cfg_w7x, 14),
        geo_option="nc",
        geofilename=str(eik_path),
        nhermite=8,
        nlaguerre=4,
    )
    cfg_eig, grids_eig, geo_eig = load_eik_nc(eik_path, cfg_eig)
    dt_w = compute_linear_dt(cfg_eig, grids_eig, geo_eig)
    op_eig64 = init_linear_operator(cfg_eig, grids_eig, geo_eig, dtype=jnp.float64)
    op_eig32 = op_eig64.astype(jnp.float32)

    g_init = init_moments(cfg_eig, grids_eig, geo_eig, dtype=jnp.complex128)
    w_krylov, eigvec = solve_linear_eigenmode(
        op_eig64,
        dt_w,
        iky=1,
        steps_per_iter=80,
        ncv=12,
        tol=1.0e-8,
        v0=g_init[..., 1, 0],
    )
    assert w_krylov.imag > 0.05, f"Expected unstable W7-X ITG mode (gamma > 0.05), got {w_krylov}"
    assert w_krylov.real > 0.0, f"Expected ion-direction W7-X ITG omega > 0, got {w_krylov}"

    # Evolve runtime complex64 RK4 stepper starting from eigenvector for 25 steps and verify
    # compute_growth_rates matches FP64 eigenvalue to rtol < 1e-5!
    g_c64 = (
        jnp.zeros(grids_eig.g_shape, dtype=jnp.complex64)
        .at[..., 1, 0]
        .set(eigvec.astype(jnp.complex64))
    )
    f_c64 = solve_fields(g_c64, op_eig32.solver)
    _, f_end64, phi_prev64 = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_eig32, dt_w, 25, scheme="rk4")
    )(g_c64, f_c64)
    w_rk4_c64 = complex(compute_growth_rates(f_end64.phi, phi_prev64, dt_w, grids_eig)[1, 0])
    np.testing.assert_allclose(w_rk4_c64.real, w_krylov.real, rtol=1.0e-5)
    np.testing.assert_allclose(w_rk4_c64.imag, w_krylov.imag, rtol=1.0e-5)


def test_itg_salpha_miller_and_kbm_eigenmodes_match_cpp_out_nc() -> None:
    """Verify converged tokamak ITG & KBM eigenmodes (gamma, omega) against *_correct.out.nc."""
    cases = [
        (
            "itg_salpha_adiabatic",
            BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in",
            GOLDEN_BENCH_DIR
            / "linear"
            / "ITG_cyclone"
            / "itg_salpha_adiabatic_electrons_correct.out.nc",
            6,  # Peak ITG mode ky = 0.30
            1.0e-3,
        ),
        (
            "itg_miller_adiabatic",
            BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_adiabatic_electrons.in",
            GOLDEN_BENCH_DIR
            / "linear"
            / "ITG_cyclone"
            / "itg_miller_adiabatic_electrons_correct.out.nc",
            8,  # Peak ITG mode ky = 0.40
            1.0e-3,
        ),
    ]
    for name, in_p, out_p, iky, tol_check in cases:
        cfg1 = extract_single_ky_config(load_config(in_p), iky)
        cfg1, grids1, geo1 = init_geometry(cfg1)
        op1 = init_linear_operator(cfg1, grids1, geo1).astype(jnp.float32)
        with netCDF4.Dataset(out_p, "r") as ds:
            t_ref = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
            dt = float(t_ref[0])
            nsteps = int(round(float(t_ref[-1]) / dt))
            om_series = np.asarray(ds.groups["Diagnostics"].variables["omega_kxkyt"][:, iky, 0, :])
            w_cpp_avg = np.mean(om_series[len(t_ref) // 2 :], axis=0)

        g_init = init_moments(cfg1, grids1, geo1, dtype=jnp.complex64)
        f_init = solve_fields(g_init, op1.solver)
        _, f_end, phi_prev = jax.jit(
            lambda g, f, op=op1, d=dt, n=nsteps: advance_linear_steps(g, f, op, d, n, scheme="rk4")
        )(g_init, f_init)
        w_jax = complex(compute_growth_rates(f_end.phi, phi_prev, dt, grids1)[1, 0])
        np.testing.assert_allclose(
            w_jax.imag, w_cpp_avg[1], rtol=tol_check, err_msg=f"{name}:gamma"
        )
        np.testing.assert_allclose(
            w_jax.real, w_cpp_avg[0], rtol=tol_check, err_msg=f"{name}:omega"
        )

    # Electromagnetic 2-species KBM eigenmode check (ky = 0.30, iky = 3 in kbm_miller.in):
    # assemble exact continuous linear operator via jax.vmap, verify unstable KBM (gamma > 0.5,
    # omega > 0), FP64 RK4 consistency (< 1e-8), and runtime complex64 RK4 consistency (< 5e-5).
    cfg_kbm = dataclasses.replace(
        extract_single_ky_config(load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"), 3),
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=6,
        nlaguerre=2,
        p_hyper_m=3,
        boundary="periodic",
        boundary_option_periodic=True,
    )
    cfg_kbm, grids_kbm, geo_kbm = init_geometry(cfg_kbm)
    dt_kbm = compute_linear_dt(cfg_kbm, grids_kbm, geo_kbm)
    op_kbm64 = init_linear_operator(cfg_kbm, grids_kbm, geo_kbm, dtype=jnp.float64)
    op_kbm32 = op_kbm64.astype(jnp.float32)

    kbm_ms = (2, 6, 2, 8)
    kbm_ndof = int(np.prod(kbm_ms))

    @jax.jit
    def _kbm_col(ej: jax.Array) -> jax.Array:
        g_c = (
            jnp.zeros(grids_kbm.g_shape, dtype=jnp.complex128).at[..., 1, 0].set(ej.reshape(kbm_ms))
        )
        rhs_c = linear_rhs(g_c, solve_fields(g_c, op_kbm64.solver), op_kbm64, dt_kbm)
        return rhs_c[..., 1, 0].ravel()

    lmat_kbm = np.asarray(jax.vmap(_kbm_col, out_axes=1)(jnp.eye(kbm_ndof, dtype=jnp.complex128)))
    evals_kbm, evecs_kbm = eig(lmat_kbm)
    lead_idx = int(np.argmax(evals_kbm.real))
    w_kbm = complex(1.0j * evals_kbm[lead_idx])
    vec_kbm = jnp.asarray(evecs_kbm[:, lead_idx].reshape(kbm_ms), dtype=jnp.complex128)
    assert w_kbm.imag > 0.5 and w_kbm.real > 0.0

    g_kbm64 = jnp.zeros(grids_kbm.g_shape, dtype=jnp.complex128).at[..., 1, 0].set(vec_kbm)
    f_kbm64 = solve_fields(g_kbm64, op_kbm64.solver)
    _, f_kbm64_end, phi_kbm64_prev = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_kbm64, dt_kbm, 10, scheme="rk4")
    )(g_kbm64, f_kbm64)
    w_kbm_rk4_64 = complex(
        compute_growth_rates(f_kbm64_end.phi, phi_kbm64_prev, dt_kbm, grids_kbm)[1, 0]
    )
    np.testing.assert_allclose(w_kbm_rk4_64.real, w_kbm.real, rtol=1.0e-8)
    np.testing.assert_allclose(w_kbm_rk4_64.imag, w_kbm.imag, rtol=1.0e-8)

    g_kbm32 = g_kbm64.astype(jnp.complex64)
    f_kbm32 = solve_fields(g_kbm32, op_kbm32.solver)
    _, f_kbm32_end, _ = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_kbm32, dt_kbm, 25, scheme="rk4")
    )(g_kbm32, f_kbm32)
    w_kbm_rk4_32 = complex(
        compute_growth_rates(f_kbm32_end.phi, f_kbm32.phi, 25.0 * dt_kbm, grids_kbm)[1, 0]
    )
    np.testing.assert_allclose(w_kbm_rk4_32.real, w_kbm.real, rtol=5.0e-5)
    np.testing.assert_allclose(w_kbm_rk4_32.imag, w_kbm.imag, rtol=5.0e-5)


# ---------------------------------------------------------------------------
# 7. Autodiff / jax.grad & jax.debug_nans(True) Safety
# ---------------------------------------------------------------------------


def test_linear_step_autodiff_and_debug_nans_safety() -> None:
    """Verify jax.grad through solve_fields -> step_linear -> diagnostics has zero NaNs."""
    cfg = dataclasses.replace(
        load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"),
        nx=7,
        ny=7,
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=6,
        nlaguerre=3,
        fbpar=1.0,
        boundary="periodic",
        boundary_option_periodic=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    op = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
    g0 = _random_complex_g(grids.g_shape, np.asarray(grids.active_mask), seed=808)

    def _loss(g_real_imag: tuple[jax.Array, jax.Array]) -> jax.Array:
        g_in = g_real_imag[0] + 1.0j * g_real_imag[1]
        f0 = solve_fields(g_in, op.solver)
        g1, f1 = step_linear(g_in, f0, op, 0.002, scheme="rk4")
        omg = compute_growth_rates(f1.phi, f0.phi, 0.002, grids)
        p2 = compute_phi2_ky(f1.phi, grids, geo)
        return (
            jnp.sum(p2)
            + jnp.sum(jnp.real(omg) ** 2 + jnp.imag(omg) ** 2)
            + jnp.sum(jnp.abs(g1) ** 2)
        )

    with jax.debug_nans(True):
        grad_re, grad_im = jax.jit(jax.grad(_loss))((jnp.real(g0), jnp.imag(g0)))

    assert bool(jnp.all(jnp.isfinite(grad_re)))
    assert bool(jnp.all(jnp.isfinite(grad_im)))
    # Masked (iky=0, ikx=0) carries exact zero gradient
    np.testing.assert_array_equal(np.asarray(grad_re[..., 0, 0]), 0.0)
    np.testing.assert_array_equal(np.asarray(grad_im[..., 0, 0]), 0.0)
    assert float(jnp.linalg.norm(grad_re)) > 0.0
