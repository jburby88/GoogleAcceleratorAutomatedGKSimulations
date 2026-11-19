"""Flux-tube magnetic geometry loader and operator precomputation for GX-JAX.

Ported strictly from:
- ``gx/include/geometry.h``
- ``gx/src/geometry.cu`` (``S_alpha_geo``, ``geo_nc``, ``Eik_geo``,
  ``Geometry::initializeOperatorArrays``, ``Geometry::calculate_bgrad``)
- ``gx/src/device_funcs.cu`` (``init_kperp2``, ``init_omegad``, ``calc_bgrad``,
  ``init_ftwist``, ``init_m0``, ``init_deltaKx``, ``init_kperp2_ntft``, ``init_omegad_ntft``)
- ``gx/geometry_modules/vmec/src/{solver.cpp,geometric_coefficients.cpp}``
- ``gx/geometry_modules/miller/{gx_geo.py,utils.py}``
"""

from __future__ import annotations

import dataclasses
import io
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.optimize import brentq

from gx_jax.config import GXConfig, c_round, resolve_jtwist_x0
from gx_jax.grids import Grids, init_grids

_MU_0 = 4.0 * math.pi * 1.0e-7


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Geometry:
    """Immutable flux-tube geometry & 3D drift/FLR operator arrays registered as a JAX PyTree."""

    # 1D field-line arrays of shape (nz,)
    z: jax.Array
    bmag: jax.Array
    bmag_inv: jax.Array
    bgrad: jax.Array
    gds2: jax.Array
    gds21: jax.Array
    gds22: jax.Array
    gbdrift: jax.Array
    gbdrift0: jax.Array
    cvdrift: jax.Array
    cvdrift0: jax.Array
    grho: jax.Array
    jacobian: jax.Array
    vol_fac: jax.Array
    flux_fac: jax.Array

    # 3D active spectral operator arrays of shape (nz, naky, nakx) (Option A)
    kperp2: jax.Array
    omegad: jax.Array
    cv_d: jax.Array
    gb_d: jax.Array

    # Non-twisting flux-tube (NTFT) arrays (populated when cfg.non_twist is True, else size-0)
    ftwist: jax.Array  # (nz,) or (0,)
    m0: jax.Array  # (nz, naky) int32 or (0, 0)
    delta_kx: jax.Array  # (nz, naky) or (0, 0)

    # Scalar equilibrium parameters (static PyTree aux_data)
    gradpar: float
    drhodpsi: float
    kxfac: float
    shat: float
    rmaj: float
    qsf: float
    shift: float
    aminor: float
    grho_avg: float
    surfarea: float
    b_ref: float
    theta_scale: float
    alpha: float
    zeta_center: float
    nfp: int
    nperiod: int
    zero_shat: bool

    # Non-twisting flux-tube real-space phase factors exp(+-i * deltaKx(z, ky) * x)
    # (geometry.cu:L938-939, device_funcs.cu:L774-791; shape (nz, naky, nx) or (0, 0, 0))
    phasefac_ntft: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0, 0, 0), dtype=jnp.complex64)
    )
    phasefacminus_ntft: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0, 0, 0), dtype=jnp.complex64)
    )

    def tree_flatten(self):
        children = (
            self.z,
            self.bmag,
            self.bmag_inv,
            self.bgrad,
            self.gds2,
            self.gds21,
            self.gds22,
            self.gbdrift,
            self.gbdrift0,
            self.cvdrift,
            self.cvdrift0,
            self.grho,
            self.jacobian,
            self.vol_fac,
            self.flux_fac,
            self.kperp2,
            self.omegad,
            self.cv_d,
            self.gb_d,
            self.ftwist,
            self.m0,
            self.delta_kx,
            self.phasefac_ntft,
            self.phasefacminus_ntft,
        )
        aux_data = (
            self.gradpar,
            self.drhodpsi,
            self.kxfac,
            self.shat,
            self.rmaj,
            self.qsf,
            self.shift,
            self.aminor,
            self.grho_avg,
            self.surfarea,
            self.b_ref,
            self.theta_scale,
            self.alpha,
            self.zeta_center,
            self.nfp,
            self.nperiod,
            self.zero_shat,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Geometry:
        (
            gradpar,
            drhodpsi,
            kxfac,
            shat,
            rmaj,
            qsf,
            shift,
            aminor,
            grho_avg,
            surfarea,
            b_ref,
            theta_scale,
            alpha,
            zeta_center,
            nfp,
            nperiod,
            zero_shat,
        ) = aux_data
        (
            z,
            bmag,
            bmag_inv,
            bgrad,
            gds2,
            gds21,
            gds22,
            gbdrift,
            gbdrift0,
            cvdrift,
            cvdrift0,
            grho,
            jacobian,
            vol_fac,
            flux_fac,
            kperp2,
            omegad,
            cv_d,
            gb_d,
            ftwist,
            m0,
            delta_kx,
            phasefac_ntft,
            phasefacminus_ntft,
        ) = children
        return cls(
            z=z,
            bmag=bmag,
            bmag_inv=bmag_inv,
            bgrad=bgrad,
            gds2=gds2,
            gds21=gds21,
            gds22=gds22,
            gbdrift=gbdrift,
            gbdrift0=gbdrift0,
            cvdrift=cvdrift,
            cvdrift0=cvdrift0,
            grho=grho,
            jacobian=jacobian,
            vol_fac=vol_fac,
            flux_fac=flux_fac,
            kperp2=kperp2,
            omegad=omegad,
            cv_d=cv_d,
            gb_d=gb_d,
            ftwist=ftwist,
            m0=m0,
            delta_kx=delta_kx,
            phasefac_ntft=phasefac_ntft,
            phasefacminus_ntft=phasefacminus_ntft,
            gradpar=gradpar,
            drhodpsi=drhodpsi,
            kxfac=kxfac,
            shat=shat,
            rmaj=rmaj,
            qsf=qsf,
            shift=shift,
            aminor=aminor,
            grho_avg=grho_avg,
            surfarea=surfarea,
            b_ref=b_ref,
            theta_scale=theta_scale,
            alpha=alpha,
            zeta_center=zeta_center,
            nfp=nfp,
            nperiod=nperiod,
            zero_shat=zero_shat,
        )

    def astype(self, dtype: jnp.dtype) -> Geometry:
        """Cast floating-point/complex geometry leaves to ``dtype`` (e.g. ``float32``)."""
        target = jnp.dtype(dtype)
        r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        c_target = jnp.complex128 if r_target == jnp.float64 else jnp.complex64

        def _cast(leaf):
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                return (
                    leaf.astype(c_target)
                    if isinstance(leaf, jax.core.Tracer)
                    else jnp.asarray(np.asarray(leaf), dtype=c_target)
                )
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                return (
                    leaf.astype(r_target)
                    if isinstance(leaf, jax.core.Tracer)
                    else jnp.asarray(np.asarray(leaf), dtype=r_target)
                )
            return leaf

        return jax.tree_util.tree_map(_cast, self)


def interp_to_new_grid(
    geo_array_in: np.ndarray,
    z_on_theta_grid: np.ndarray,
    uniform_grid: np.ndarray,
) -> np.ndarray:
    """Natural cubic spline interpolation matching GSL ``gsl_interp_cspline`` (``solver.cpp``)."""
    z_in = np.asarray(z_on_theta_grid, dtype=np.float64)
    y_in = np.asarray(geo_array_in, dtype=np.float64)
    z_eval = np.asarray(uniform_grid, dtype=np.float64).copy()
    if z_eval.size > 0 and z_eval[0] < z_in[0]:
        z_eval[0] = z_in[0]
    spline = CubicSpline(z_in, y_in, bc_type="natural")
    return np.asarray(spline(z_eval), dtype=np.float64)


def calculate_bgrad(bmag: jax.Array, kz: jax.Array, gradpar: float) -> jax.Array:
    """Spectral parallel derivative ``bgrad = (d|B|/dz / |B|) * gradpar`` (``geometry.cu:L1007``).

    Matches ``GradParallel1D::dz1D`` (``grad_parallel.cu:L225-265``) + ``calc_bgrad``
    (``device_funcs.cu:L650-654``): 1D ``rfft`` along ``z``, multiply by ``i * kz``, then
    ``irfft(..., n=nz)`` (which zeros the Nyquist imaginary component for even ``nz``, identical
    to ``cufftExecC2R``).
    """
    nz = bmag.shape[0]
    if nz <= 1:
        return jnp.zeros_like(bmag)
    b64 = np.asarray(bmag, dtype=np.float64)
    kz_rfft = np.asarray(kz[: nz // 2 + 1], dtype=np.float64)
    b_hat = np.fft.rfft(b64)
    dz_bmag = np.fft.irfft(1j * kz_rfft * b_hat, n=nz)
    return jnp.asarray((dz_bmag / b64) * float(gradpar), dtype=bmag.dtype)


def _compute_operator_arrays(
    cfg: GXConfig,
    grids: Grids,
    *,
    shat: float,
    bmag_inv: jax.Array,
    gds2: jax.Array,
    gds21: jax.Array,
    gds22: jax.Array,
    gbdrift: jax.Array,
    gbdrift0: jax.Array,
    cvdrift: jax.Array,
    cvdrift0: jax.Array,
    dtype: jnp.dtype,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Port ``Geometry::initializeOperatorArrays`` onto active ``(nz, naky, nakx)`` (Option A)."""
    kx = grids.kx[None, None, :]  # (1, 1, nakx)
    ky = grids.ky[None, :, None]  # (1, naky, 1)
    gds2_3 = gds2[:, None, None]
    gds21_3 = gds21[:, None, None]
    gds22_3 = gds22[:, None, None]
    bmag_inv_3 = bmag_inv[:, None, None]
    cv_3 = cvdrift[:, None, None]
    gb_3 = gbdrift[:, None, None]
    cv0_3 = cvdrift0[:, None, None]
    gb0_3 = gbdrift0[:, None, None]
    mask_3 = grids.active_mask[None, :, :]
    c_dtype = jnp.complex128 if jnp.dtype(dtype) == jnp.float64 else jnp.complex64

    if cfg.non_twist:
        # device_funcs.cu:L656-715, L774-791
        nz = grids.nz
        shat_inv = 1.0 / float(shat)
        ftwist64 = float(shat) * gds21 / gds22  # (nz,)
        delta = 0.01313
        idz = np.arange(nz, dtype=np.int64)
        ft_np = np.asarray(ftwist64, dtype=np.float64)
        ky_np = np.asarray(grids.ky, dtype=np.float64)
        shift_edge = 2.0 * math.pi * float(grids.zp) * float(cfg.kxfac) * float(shat)
        term_z = (1.0 - delta) * (
            ft_np[idz % nz] - shift_edge * np.floor(idz / float(nz))
        ) + delta * (ft_np[(idz + 1) % nz] - shift_edge * np.floor((idz + 1) / float(nz)))
        term_0 = (1.0 - delta) * ft_np[nz // 2] + delta * ft_np[(nz // 2 + 1) % nz]

        arg1 = float(grids.x0) * ky_np[None, :] * term_z[:, None]
        arg0 = float(grids.x0) * ky_np[None, :] * term_0
        vec_round = np.vectorize(c_round, otypes=[np.int32])
        m0_np = -vec_round(arg1) + vec_round(arg0)
        m0_arr = jnp.asarray(m0_np, dtype=jnp.int32)  # (nz, naky)
        delta_kx64 = grids.ky[None, :] * ftwist64[:, None] + m0_arr.astype(jnp.float64) / float(
            grids.x0
        )

        ft_3 = ftwist64[:, None, None]
        dkx_3 = delta_kx64[:, :, None]
        m0_over_x0 = (m0_arr.astype(jnp.float64) / float(grids.x0))[:, :, None]

        kperp2_raw = (
            (ky**2) * (gds2_3 - 2.0 * ft_3 * gds21_3 * shat_inv + (ft_3**2) * gds22_3 * shat_inv**2)
            + ((kx + dkx_3) ** 2) * gds22_3 * (shat_inv**2)
        ) * (bmag_inv_3**2)
        cv_d_raw = ky * cv_3 + shat_inv * (kx + m0_over_x0) * cv0_3
        gb_d_raw = ky * gb_3 + shat_inv * (kx + m0_over_x0) * gb0_3
        omegad_raw = cv_d_raw + gb_d_raw

        phase_arg_np = (
            np.asarray(delta_kx64, dtype=np.float64)[:, :, None]
            * np.asarray(grids.x, dtype=np.float64)[None, None, :]
        )
        np_c_dtype = np.complex128 if c_dtype == jnp.complex128 else np.complex64
        phasefac_ntft_out = jnp.asarray(
            np.exp(1.0j * phase_arg_np).astype(np_c_dtype), dtype=c_dtype
        )
        phasefacminus_ntft_out = jnp.asarray(
            np.exp(-1.0j * phase_arg_np).astype(np_c_dtype), dtype=c_dtype
        )

        ftwist_out = jnp.asarray(np.asarray(ftwist64), dtype=dtype)
        delta_kx_out = jnp.asarray(np.asarray(delta_kx64), dtype=dtype)
    else:
        # device_funcs.cu:L794-855
        shat_inv = 1.0 if shat == 0.0 else (1.0 / float(shat))
        kx_eff = kx * shat_inv
        kperp2_raw = (ky * (ky * gds2_3 + 2.0 * kx_eff * gds21_3) + (kx_eff**2) * gds22_3) * (
            bmag_inv_3**2
        )
        cv_d_raw = ky * cv_3 + kx_eff * cv0_3
        gb_d_raw = ky * gb_3 + kx_eff * gb0_3
        omegad_raw = cv_d_raw + gb_d_raw

        ftwist_out = jnp.zeros((0,), dtype=dtype)
        m0_arr = jnp.zeros((0, 0), dtype=jnp.int32)
        delta_kx_out = jnp.zeros((0, 0), dtype=dtype)
        phasefac_ntft_out = jnp.zeros((0, 0, 0), dtype=c_dtype)
        phasefacminus_ntft_out = jnp.zeros((0, 0, 0), dtype=c_dtype)

    kperp2 = jnp.asarray(np.asarray(jnp.where(mask_3, kperp2_raw, 0.0)), dtype=dtype)
    cv_d = jnp.asarray(np.asarray(jnp.where(mask_3, cv_d_raw, 0.0)), dtype=dtype)
    gb_d = jnp.asarray(np.asarray(jnp.where(mask_3, gb_d_raw, 0.0)), dtype=dtype)
    omegad = jnp.asarray(np.asarray(jnp.where(mask_3, omegad_raw, 0.0)), dtype=dtype)
    return (
        kperp2,
        omegad,
        cv_d,
        gb_d,
        ftwist_out,
        m0_arr,
        delta_kx_out,
        phasefac_ntft_out,
        phasefacminus_ntft_out,
    )


def _finalize_geometry(
    cfg_raw: GXConfig,
    *,
    z_np: np.ndarray,
    bmag_np: np.ndarray,
    bmag_inv_np: np.ndarray,
    gds2_np: np.ndarray,
    gds21_np: np.ndarray,
    gds22_np: np.ndarray,
    gbdrift_np: np.ndarray,
    gbdrift0_np: np.ndarray,
    cvdrift_np: np.ndarray,
    cvdrift0_np: np.ndarray,
    grho_np: np.ndarray,
    jacobian_np: np.ndarray,
    bgrad_analytical_np: np.ndarray | None,
    gradpar: float,
    drhodpsi: float,
    kxfac: float,
    shat: float,
    rmaj: float,
    qsf: float,
    shift: float = 0.0,
    b_ref: float = 1.0,
    a_ref: float = 1.0,
    grhoavg_in: float | None = None,
    surfarea: float = 1.0,
    theta_scale: float = 1.0,
    alpha: float = 0.0,
    zeta_center: float = 0.0,
    nfp: int = 1,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[GXConfig, Grids, Geometry]:
    """Finalize ``cfg``, ``grids``, ``bgrad``, 3D operators, and surface-average weights."""
    out_dtype = jnp.dtype(dtype)

    cfg_updated = dataclasses.replace(
        cfg_raw,
        drhodpsi=float(drhodpsi),
        kxfac=float(kxfac),
        shat=float(shat),
        rmaj=float(rmaj),
        qsf=float(qsf),
        shift=float(shift),
        b_ref=float(b_ref),
        a_ref=float(a_ref),
        surfarea=float(surfarea),
    )
    cfg_final = resolve_jtwist_x0(
        cfg_updated,
        shat=float(shat),
        gds21_0=float(gds21_np[0]),
        gds22_0=float(gds22_np[0]),
    )
    grids = init_grids(cfg_final)

    z = jnp.asarray(z_np, dtype=jnp.float64)
    bmag = jnp.asarray(bmag_np, dtype=jnp.float64)
    bmag_inv = jnp.asarray(bmag_inv_np, dtype=jnp.float64)
    gds2 = jnp.asarray(gds2_np, dtype=jnp.float64)
    gds21 = jnp.asarray(gds21_np, dtype=jnp.float64)
    gds22 = jnp.asarray(gds22_np, dtype=jnp.float64)
    gbdrift = jnp.asarray(gbdrift_np, dtype=jnp.float64)
    gbdrift0 = jnp.asarray(gbdrift0_np, dtype=jnp.float64)
    cvdrift = jnp.asarray(cvdrift_np, dtype=jnp.float64)
    cvdrift0 = jnp.asarray(cvdrift0_np, dtype=jnp.float64)
    grho = jnp.asarray(grho_np, dtype=jnp.float64)
    jacobian = jnp.asarray(jacobian_np, dtype=jnp.float64)

    if bgrad_analytical_np is not None:
        bgrad = jnp.asarray(bgrad_analytical_np, dtype=jnp.float64)
    else:
        bgrad = calculate_bgrad(bmag, grids.kz, gradpar)

    vol_denom = jnp.sum(jacobian)
    flux_denom = jnp.sum(jacobian * grho)
    vol_fac = jacobian / vol_denom
    flux_fac = jacobian / flux_denom
    grho_avg = float(flux_denom / vol_denom) if grhoavg_in is None else float(grhoavg_in)
    cfg_final = dataclasses.replace(cfg_final, grhoavg=grho_avg)

    (
        kperp2,
        omegad,
        cv_d,
        gb_d,
        ftwist,
        m0,
        delta_kx,
        phasefac_ntft,
        phasefacminus_ntft,
    ) = _compute_operator_arrays(
        cfg_final,
        grids,
        shat=float(shat),
        bmag_inv=bmag_inv,
        gds2=gds2,
        gds21=gds21,
        gds22=gds22,
        gbdrift=gbdrift,
        gbdrift0=gbdrift0,
        cvdrift=cvdrift,
        cvdrift0=cvdrift0,
        dtype=out_dtype,
    )

    def _hc(arr: jax.Array) -> jax.Array:
        return jnp.asarray(np.asarray(arr), dtype=out_dtype)

    geo = Geometry(
        z=_hc(z),
        bmag=_hc(bmag),
        bmag_inv=_hc(bmag_inv),
        bgrad=_hc(bgrad),
        gds2=_hc(gds2),
        gds21=_hc(gds21),
        gds22=_hc(gds22),
        gbdrift=_hc(gbdrift),
        gbdrift0=_hc(gbdrift0),
        cvdrift=_hc(cvdrift),
        cvdrift0=_hc(cvdrift0),
        grho=_hc(grho),
        jacobian=_hc(jacobian),
        vol_fac=_hc(vol_fac),
        flux_fac=_hc(flux_fac),
        kperp2=kperp2,
        omegad=omegad,
        cv_d=cv_d,
        gb_d=gb_d,
        ftwist=ftwist,
        m0=m0,
        delta_kx=delta_kx,
        phasefac_ntft=phasefac_ntft,
        phasefacminus_ntft=phasefacminus_ntft,
        gradpar=float(gradpar),
        drhodpsi=float(drhodpsi),
        kxfac=float(kxfac),
        shat=float(shat),
        rmaj=float(rmaj),
        qsf=float(qsf),
        shift=float(shift),
        aminor=float(a_ref),
        grho_avg=float(flux_denom / vol_denom),
        surfarea=float(surfarea),
        b_ref=float(b_ref),
        theta_scale=float(theta_scale),
        alpha=float(alpha),
        zeta_center=float(zeta_center),
        nfp=int(nfp),
        nperiod=int(cfg_final.nperiod),
        zero_shat=bool(cfg_final.zero_shat),
    )
    return cfg_final, grids, geo


def s_alpha_geometry(
    cfg: GXConfig,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[GXConfig, Grids, Geometry]:
    """Analytical ``s-alpha`` / ``slab`` / ``const-curv`` geometry (``S_alpha_geo``)."""
    nz = cfg.nz
    qsf = cfg.qsf
    rmaj = cfg.rmaj
    shat = cfg.shat
    drhodpsi = 1.0
    kxfac = 1.0
    gradpar = abs(1.0 / (qsf * rmaj))

    shift = cfg.shift
    if shift < 0.0:
        shift = 0.0
        if cfg.species:
            t_last = cfg.species[-1].temp
            for sp in cfg.species:
                shift += qsf * qsf * rmaj * cfg.beta * (sp.temp / t_last) * (sp.tprim + sp.fprim)

    k_idx = np.arange(nz, dtype=np.float64)
    z_grid = 2.0 * math.pi * float(cfg.zp) * (k_idx - (nz // 2)) / float(nz)
    theta = np.zeros_like(z_grid) if cfg.local_limit else z_grid.copy()

    bmag = 1.0 / (1.0 + cfg.eps * np.cos(theta))
    bgrad = gradpar * cfg.eps * np.sin(theta) * bmag

    s_term = shat * theta - shift * np.sin(theta)
    gds2 = 1.0 + s_term**2
    gds21 = -shat * s_term
    gds22 = np.full(nz, shat**2, dtype=np.float64)

    gbdrift = (np.cos(theta) + s_term * np.sin(theta)) / rmaj
    cvdrift = gbdrift.copy()
    gbdrift0 = -shat * np.sin(theta) / rmaj
    cvdrift0 = gbdrift0.copy()
    grho = np.ones(nz, dtype=np.float64)

    if cfg.const_curv:
        cvdrift = np.full(nz, 1.0 / rmaj, dtype=np.float64)
        gbdrift = np.full(nz, 1.0 / rmaj, dtype=np.float64)
        cvdrift0 = np.zeros(nz, dtype=np.float64)
        gbdrift0 = np.zeros(nz, dtype=np.float64)

    if cfg.slab:
        cvdrift = np.zeros(nz, dtype=np.float64)
        gbdrift = np.zeros(nz, dtype=np.float64)
        cvdrift0 = np.zeros(nz, dtype=np.float64)
        gbdrift0 = np.zeros(nz, dtype=np.float64)
        bgrad = np.zeros(nz, dtype=np.float64)
        bmag = np.ones(nz, dtype=np.float64)
        gradpar = (1.0 / cfg.z0) if cfg.z0 > 0.0 else 1.0
        if cfg.zero_shat:
            gds21 = np.zeros(nz, dtype=np.float64)
            gds22 = np.ones(nz, dtype=np.float64)
            shat = 0.0

    if cfg.local_limit:
        gradpar = 1.0

    bmag_inv = 1.0 / bmag
    jacobian = 1.0 / np.abs(drhodpsi * gradpar * bmag)

    return _finalize_geometry(
        cfg,
        z_np=z_grid,
        bmag_np=bmag,
        bmag_inv_np=bmag_inv,
        gds2_np=gds2,
        gds21_np=gds21,
        gds22_np=gds22,
        gbdrift_np=gbdrift,
        gbdrift0_np=gbdrift0,
        cvdrift_np=cvdrift,
        cvdrift0_np=cvdrift0,
        grho_np=grho,
        jacobian_np=jacobian,
        bgrad_analytical_np=bgrad,
        gradpar=gradpar,
        drhodpsi=drhodpsi,
        kxfac=kxfac,
        shat=shat,
        rmaj=rmaj,
        qsf=qsf,
        shift=shift,
        dtype=dtype,
    )


def _read_nc_scalar(ds: netCDF4.Dataset, *names: str, default: float | None = None) -> float:
    for name in names:
        if name in ds.variables:
            val = np.asarray(ds.variables[name][:]).reshape(-1)[0]
            return float(val)
    if default is not None:
        return float(default)
    raise KeyError(f"None of scalar variables {names} found in {ds.filepath()}.")


def load_eik_nc(
    nc_path: str | Path,
    cfg: GXConfig,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[GXConfig, Grids, Geometry]:
    """Load a ``.eik.nc`` NetCDF geometry file porting ``geo_nc`` (``geometry.cu:L434-678``)."""
    path = Path(nc_path)
    nz = cfg.nz

    with netCDF4.Dataset(path, "r") as ds:
        if "z" in ds.dimensions:
            n_file = len(ds.dimensions["z"])
        elif "nt" in ds.dimensions:
            n_file = len(ds.dimensions["nt"])
        else:
            raise KeyError(f"Expected dimension 'z' or 'nt' in {path}.")

        if nz != n_file - 1:
            raise ValueError(
                f"Number of field-line points in {path} (N-1={n_file - 1}) "
                f"does not match cfg.nz={nz}."
            )

        nc_z = np.asarray(ds.variables["theta"][:], dtype=np.float64)
        nc_bmag = np.asarray(ds.variables["bmag"][:], dtype=np.float64)
        nc_bmag_inv = 1.0 / nc_bmag
        nc_gradpar = np.asarray(ds.variables["gradpar"][:], dtype=np.float64)
        if not math.isclose(
            float(nc_gradpar[0]), float(nc_gradpar[n_file // 2]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                "GX requires an equal-arc theta coordinate so that gradpar is constant "
                f"(got gradpar[0]={nc_gradpar[0]}, gradpar[N//2]={nc_gradpar[n_file // 2]})."
            )
        gradpar = float(nc_gradpar[0])

        nc_grho = np.asarray(ds.variables["grho"][:], dtype=np.float64)
        nc_gds2 = np.asarray(ds.variables["gds2"][:], dtype=np.float64)
        nc_gds21 = np.asarray(ds.variables["gds21"][:], dtype=np.float64)
        nc_gds22 = np.asarray(ds.variables["gds22"][:], dtype=np.float64)
        # Critical factor of 1/2 on all four magnetic drift coefficients (geometry.cu:L532-544)
        nc_gbdrift = 0.5 * np.asarray(ds.variables["gbdrift"][:], dtype=np.float64)
        nc_gbdrift0 = 0.5 * np.asarray(ds.variables["gbdrift0"][:], dtype=np.float64)
        nc_cvdrift = 0.5 * np.asarray(ds.variables["cvdrift"][:], dtype=np.float64)
        nc_cvdrift0 = 0.5 * np.asarray(ds.variables["cvdrift0"][:], dtype=np.float64)

        drhodpsi = _read_nc_scalar(ds, "drhodpsi")
        kxfac = _read_nc_scalar(ds, "kxfac")
        shat = _read_nc_scalar(ds, "shat")
        rmaj = _read_nc_scalar(ds, "Rmaj", "rmaj")
        qsf = _read_nc_scalar(ds, "q")

        b_ref = _read_nc_scalar(ds, "B_T", default=cfg.b_ref)
        a_ref = _read_nc_scalar(ds, "aminor", default=cfg.a_ref)
        grhoavg_opt = _read_nc_scalar(ds, "grhoavg", default=-1.0)
        surfarea = _read_nc_scalar(ds, "surfarea", default=cfg.surfarea)
        theta_scale = _read_nc_scalar(ds, "scale", default=1.0)
        nfp = int(round(_read_nc_scalar(ds, "nfp", default=1.0)))
        alpha = _read_nc_scalar(ds, "alpha", default=cfg.alpha)
        zeta_center = _read_nc_scalar(ds, "zeta_center", default=0.0)

    k_idx = np.arange(nz, dtype=np.float64)
    z_grid = 2.0 * math.pi * float(cfg.zp) * (k_idx - (nz // 2)) / float(nz)

    bmag = interp_to_new_grid(nc_bmag, nc_z, z_grid)
    bmag_inv = interp_to_new_grid(nc_bmag_inv, nc_z, z_grid)
    gds2 = interp_to_new_grid(nc_gds2, nc_z, z_grid)
    gds21 = interp_to_new_grid(nc_gds21, nc_z, z_grid)
    gds22 = interp_to_new_grid(nc_gds22, nc_z, z_grid)
    gbdrift = interp_to_new_grid(nc_gbdrift, nc_z, z_grid)
    gbdrift0 = interp_to_new_grid(nc_gbdrift0, nc_z, z_grid)
    cvdrift = interp_to_new_grid(nc_cvdrift, nc_z, z_grid)
    cvdrift0 = interp_to_new_grid(nc_cvdrift0, nc_z, z_grid)
    grho = interp_to_new_grid(nc_grho, nc_z, z_grid)
    jacobian = 1.0 / np.abs(drhodpsi * gradpar * bmag)

    return _finalize_geometry(
        cfg,
        z_np=z_grid,
        bmag_np=bmag,
        bmag_inv_np=bmag_inv,
        gds2_np=gds2,
        gds21_np=gds21,
        gds22_np=gds22,
        gbdrift_np=gbdrift,
        gbdrift0_np=gbdrift0,
        cvdrift_np=cvdrift,
        cvdrift0_np=cvdrift0,
        grho_np=grho,
        jacobian_np=jacobian,
        bgrad_analytical_np=None,
        gradpar=gradpar,
        drhodpsi=drhodpsi,
        kxfac=kxfac,
        shat=shat,
        rmaj=rmaj,
        qsf=qsf,
        shift=cfg.shift,
        b_ref=b_ref,
        a_ref=a_ref,
        grhoavg_in=(grhoavg_opt if grhoavg_opt > 0.0 else None),
        surfarea=surfarea,
        theta_scale=theta_scale,
        alpha=alpha,
        zeta_center=zeta_center,
        nfp=nfp,
        dtype=dtype,
    )


def _load_eik_out_stream(
    lines: list[str],
    cfg: GXConfig,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[GXConfig, Grids, Geometry]:
    """Parse ASCII ``eik.out`` lines strictly following ``Eik_geo`` (``geometry.cu:L681-883``)."""
    header_vals = lines[1].split()
    _ntgrid = int(header_vals[0])
    nperiod_file = int(header_vals[1])
    ntheta_file = int(header_vals[2])
    drhodpsi = float(header_vals[3])
    rmaj = float(header_vals[4])
    shat = float(header_vals[5])
    kxfac = float(header_vals[6])
    qsf = float(header_vals[7])

    b_ref = cfg.b_ref
    a_ref = cfg.a_ref
    grhoavg_opt: float | None = None
    surfarea = cfg.surfarea
    if len(header_vals) >= 12:
        b_ref = float(header_vals[8])
        a_ref = float(header_vals[9])
        grhoavg_opt = float(header_vals[10])
        surfarea = float(header_vals[11])

    new_nz = (2 * nperiod_file - 1) * ntheta_file
    if cfg.nz != new_nz:
        raise ValueError(f"Mismatch between cfg.nz={cfg.nz} and eik.out newNz={new_nz}.")
    if cfg.nperiod != nperiod_file:
        raise ValueError(
            f"Mismatch between cfg.nperiod={cfg.nperiod} and eik.out nperiod={nperiod_file}."
        )

    n_pts = new_nz + 1
    eik_gbdrift = np.empty(n_pts, dtype=np.float64)
    eik_gradpar = np.empty(n_pts, dtype=np.float64)
    eik_grho = np.empty(n_pts, dtype=np.float64)
    eik_z = np.empty(n_pts, dtype=np.float64)
    idx = 3  # skip line 2 ("gbdrift gradpar grho tgrid")
    for i in range(n_pts):
        parts = lines[idx + i].split()
        eik_gbdrift[i] = 0.5 * float(parts[0])
        eik_gradpar[i] = float(parts[1])
        eik_grho[i] = float(parts[2])
        eik_z[i] = float(parts[3])
    idx += n_pts

    if not math.isclose(
        float(eik_gradpar[0]), float(eik_gradpar[new_nz // 2]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("GX requires an equal-arc theta coordinate (gradpar = const).")
    gradpar = float(eik_gradpar[0])

    idx += 1  # skip "cvdrift gds2 bmag tgrid"
    eik_cvdrift = np.empty(n_pts, dtype=np.float64)
    eik_gds2 = np.empty(n_pts, dtype=np.float64)
    eik_bmag = np.empty(n_pts, dtype=np.float64)
    for i in range(n_pts):
        parts = lines[idx + i].split()
        eik_cvdrift[i] = 0.5 * float(parts[0])
        eik_gds2[i] = float(parts[1])
        eik_bmag[i] = float(parts[2])
    idx += n_pts
    eik_bmag_inv = 1.0 / eik_bmag
    eik_jacobian = 1.0 / np.abs(drhodpsi * gradpar * eik_bmag)

    idx += 1  # skip "gds21 gds22 tgrid"
    eik_gds21 = np.empty(n_pts, dtype=np.float64)
    eik_gds22 = np.empty(n_pts, dtype=np.float64)
    for i in range(n_pts):
        parts = lines[idx + i].split()
        eik_gds21[i] = float(parts[0])
        eik_gds22[i] = float(parts[1])
    idx += n_pts

    idx += 1  # skip "cvdrift0 gbdrift0 tgrid"
    eik_cvdrift0 = np.empty(n_pts, dtype=np.float64)
    eik_gbdrift0 = np.empty(n_pts, dtype=np.float64)
    for i in range(n_pts):
        parts = lines[idx + i].split()
        eik_cvdrift0[i] = 0.5 * float(parts[0])
        eik_gbdrift0[i] = 0.5 * float(parts[1])

    k_idx = np.arange(new_nz, dtype=np.float64)
    z_grid = 2.0 * math.pi * float(cfg.zp) * (k_idx - (new_nz // 2)) / float(new_nz)

    bmag = interp_to_new_grid(eik_bmag, eik_z, z_grid)
    bmag_inv = interp_to_new_grid(eik_bmag_inv, eik_z, z_grid)
    gds2 = interp_to_new_grid(eik_gds2, eik_z, z_grid)
    gds21 = interp_to_new_grid(eik_gds21, eik_z, z_grid)
    gds22 = interp_to_new_grid(eik_gds22, eik_z, z_grid)
    gbdrift = interp_to_new_grid(eik_gbdrift, eik_z, z_grid)
    gbdrift0 = interp_to_new_grid(eik_gbdrift0, eik_z, z_grid)
    cvdrift = interp_to_new_grid(eik_cvdrift, eik_z, z_grid)
    cvdrift0 = interp_to_new_grid(eik_cvdrift0, eik_z, z_grid)
    grho = interp_to_new_grid(eik_grho, eik_z, z_grid)
    jacobian = interp_to_new_grid(eik_jacobian, eik_z, z_grid)

    return _finalize_geometry(
        cfg,
        z_np=z_grid,
        bmag_np=bmag,
        bmag_inv_np=bmag_inv,
        gds2_np=gds2,
        gds21_np=gds21,
        gds22_np=gds22,
        gbdrift_np=gbdrift,
        gbdrift0_np=gbdrift0,
        cvdrift_np=cvdrift,
        cvdrift0_np=cvdrift0,
        grho_np=grho,
        jacobian_np=jacobian,
        bgrad_analytical_np=None,
        gradpar=gradpar,
        drhodpsi=drhodpsi,
        kxfac=kxfac,
        shat=shat,
        rmaj=rmaj,
        qsf=qsf,
        shift=cfg.shift,
        b_ref=b_ref,
        a_ref=a_ref,
        grhoavg_in=grhoavg_opt,
        surfarea=surfarea,
        dtype=dtype,
    )


def load_eik_out(
    eik_path: str | Path,
    cfg: GXConfig,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[GXConfig, Grids, Geometry]:
    """Load an ASCII ``.eik.out`` geometry file strictly porting ``Eik_geo`` (``L681-883``)."""
    lines = Path(eik_path).read_text(encoding="utf-8").splitlines()
    return _load_eik_out_stream(lines, cfg, dtype=dtype)


# ---------------------------------------------------------------------------
# Miller Local Tokamak Equilibrium Generator (ports geometry_modules/miller/)
# ---------------------------------------------------------------------------


def _miller_derm(arr: np.ndarray, ch: str, par: str = "e") -> np.ndarray:
    """Port ``geometry_modules/miller/utils.py:derm`` faithfully."""
    shape = np.shape(arr)
    if len(shape) == 1 and ch == "l":
        d1 = shape[0]
        diff_arr = np.zeros((1, d1), dtype=np.float64)
        a2 = np.reshape(arr, (1, d1))
        if par != "e":
            diff_arr[0, 0] = 2.0 * (a2[0, 1] - a2[0, 0])
            diff_arr[0, -1] = 2.0 * (a2[0, -1] - a2[0, -2])
        diff_arr[0, 1:-1] = np.diff(a2[0, :-1]) + np.diff(a2[0, 1:])
        return diff_arr
    if len(shape) == 1 and ch == "r":
        d1 = shape[0]
        diff_arr = np.zeros((d1, 1), dtype=np.float64)
        a2 = np.reshape(arr, (d1, 1))
        diff_arr[0, 0] = 2.0 * (a2[1, 0] - a2[0, 0])
        diff_arr[-1, 0] = 2.0 * (a2[-1, 0] - a2[-2, 0])
        diff_arr[1:-1, 0] = np.diff(a2[:-1, 0]) + np.diff(a2[1:, 0])
        return diff_arr
    d1, d2 = shape[0], shape[1]
    diff_arr = np.zeros((d1, d2), dtype=np.float64)
    if ch == "r":
        diff_arr[0, :] = 2.0 * (arr[1, :] - arr[0, :])
        diff_arr[-1, :] = 2.0 * (arr[-1, :] - arr[-2, :])
        diff_arr[1:-1, :] = np.diff(arr[:-1, :], axis=0) + np.diff(arr[1:, :], axis=0)
    else:
        if par != "e":
            diff_arr[:, 0] = 2.0 * (arr[:, 1] - arr[:, 0])
            diff_arr[:, -1] = 2.0 * (arr[:, -1] - arr[:, -2])
        diff_arr[:, 1:-1] = np.diff(arr[:, :-1], axis=1) + np.diff(arr[:, 1:], axis=1)
    return diff_arr


def _miller_dermv(arr: np.ndarray, brr: np.ndarray, ch: str, par: str = "e") -> np.ndarray:
    """Port ``geometry_modules/miller/utils.py:dermv`` faithfully."""
    shape = np.shape(arr)
    if len(shape) == 1 and ch == "l":
        d1 = shape[0]
        a2 = np.reshape(arr, (1, d1))
        b2 = np.reshape(brr, (1, d1))
        diff_arr = np.zeros((1, d1), dtype=np.float64)
        if par != "e":
            diff_arr[0, 0] = (4.0 * a2[0, 1] - 3.0 * a2[0, 0] - a2[0, 2]) / (
                2.0 * (b2[0, 1] - b2[0, 0])
            )
            diff_arr[0, -1] = (-4.0 * a2[0, -2] + 3.0 * a2[0, -1] + a2[0, -3]) / (
                2.0 * (b2[0, -1] - b2[0, -2])
            )
        for i in range(1, d1 - 1):
            h1 = b2[0, i + 1] - b2[0, i]
            h0 = b2[0, i] - b2[0, i - 1]
            diff_arr[0, i] = (
                a2[0, i + 1] / h1**2 + a2[0, i] * (1.0 / h0**2 - 1.0 / h1**2) - a2[0, i - 1] / h0**2
            ) / (1.0 / h1 + 1.0 / h0)
        return diff_arr
    d1, d2 = shape[0], shape[1]
    diff_arr = np.zeros((d1, d2), dtype=np.float64)
    if ch == "r":
        diff_arr[0, :] = (arr[1, :] - arr[0, :]) / (brr[1, :] - brr[0, :])
        diff_arr[-1, :] = (arr[-1, :] - arr[-2, :]) / (brr[-1, :] - brr[-2, :])
        for i in range(1, d1 - 1):
            h1 = brr[i + 1, :] - brr[i, :]
            h0 = brr[i, :] - brr[i - 1, :]
            diff_arr[i, :] = (
                arr[i + 1, :] / h1**2
                + arr[i, :] * (1.0 / h0**2 - 1.0 / h1**2)
                - arr[i - 1, :] / h0**2
            ) / (1.0 / h1 + 1.0 / h0)
    else:
        if par != "e":
            diff_arr[:, 0] = (arr[:, 1] - arr[:, 0]) / (brr[:, 1] - brr[:, 0])
            diff_arr[:, -1] = (arr[:, -1] - arr[:, -2]) / (brr[:, -1] - brr[:, -2])
        for i in range(1, d2 - 1):
            h1 = brr[:, i + 1] - brr[:, i]
            h0 = brr[:, i] - brr[:, i - 1]
            diff_arr[:, i] = (
                arr[:, i + 1] / h1**2
                + arr[:, i] * (1.0 / h0**2 - 1.0 / h1**2)
                - arr[:, i - 1] / h0**2
            ) / (1.0 / h1 + 1.0 / h0)
    return diff_arr


def _miller_nperiod_extend(
    arr: np.ndarray,
    nperiod: int,
    *,
    istheta: int = 0,
    par: str = "e",
) -> np.ndarray:
    if nperiod > 1:
        if istheta:
            arr_dum = arr
            for i in range(nperiod - 1):
                arr_app = np.concatenate(
                    (2.0 * np.pi * (i + 1) - arr_dum[::-1][1:], 2.0 * np.pi * (i + 1) + arr_dum[1:])
                )
                arr = np.concatenate((arr, arr_app))
        elif par == "e":
            arr_app = np.concatenate((arr[::-1][1:], arr[1:]))
            for _ in range(nperiod - 1):
                arr = np.concatenate((arr, arr_app))
        else:
            arr_app = np.concatenate((-arr[::-1][1:], arr[1:]))
            for _ in range(nperiod - 1):
                arr = np.concatenate((arr, arr_app))
    return arr


def _miller_reflect_n_append(arr: np.ndarray, ch: str) -> np.ndarray:
    if ch == "e":
        return np.concatenate((arr[::-1][:-1], arr[0:]))
    return np.concatenate((-arr[::-1][:-1], np.array([0.0]), arr[1:]))


def miller_geometry(
    cfg: GXConfig,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[GXConfig, Grids, Geometry]:
    """Generate Miller local tokamak geometry matching ``gx_geo.py`` + ``Eik_geo``."""
    ntgrid = int(cfg.ntheta_in // 2 + 1)
    nperiod = cfg.nperiod
    rhoc = cfg.rhoc
    qinp = cfg.qsf
    s_hat_input = cfg.shat
    if s_hat_input == 0.0 or abs(s_hat_input) < cfg.zero_shat_threshold:
        s_hat_input = 1.0e-6
    rmaj_in = cfg.rmaj
    r_geo = cfg.r_geo
    shift = cfg.shift
    akappa = cfg.akappa
    tri = cfg.tri
    akappri = cfg.akappri
    tripri = cfg.tripri
    beta_prime_input = cfg.betaprim

    delrho = 0.001
    theta = np.linspace(0.0, np.pi, ntgrid)

    r_0 = np.array(
        [rmaj_in + abs(shift) * delrho, rmaj_in, rmaj_in - abs(shift) * delrho], dtype=np.float64
    )
    rho = np.array([rhoc - delrho, rhoc, rhoc + delrho], dtype=np.float64)
    qfac = np.array(
        [
            qinp - s_hat_input * (qinp / rhoc) * delrho,
            qinp,
            qinp + s_hat_input * (qinp / rhoc) * delrho,
        ],
        dtype=np.float64,
    )
    kappa = np.array([akappa - akappri * delrho, akappa, akappa + akappri * delrho])
    delta = np.array([tri - tripri * delrho, tri, tri + tripri * delrho])
    r_mag_ax = rmaj_in
    dpdrho = beta_prime_input / 2.0

    r_surf = np.array(
        [r_0[i] + rho[i] * np.cos(theta + np.arcsin(delta[i]) * np.sin(theta)) for i in range(3)]
    )
    z_surf = np.array([kappa[i] * rho[i] * np.sin(theta) for i in range(3)])
    theta_comn = np.array([np.arctan2(z_surf[i], r_surf[i] - r_mag_ax) for i in range(3)])

    dl = np.sqrt(_miller_derm(r_surf, "l", "e") ** 2 + _miller_derm(z_surf, "l", "o") ** 2)
    l_arc = np.zeros((3, ntgrid), dtype=np.float64)
    for i in range(3):
        l_arc[i, 1:] = np.cumsum(np.sqrt(np.diff(r_surf[i]) ** 2 + np.diff(z_surf[i]) ** 2))

    rho_diff = _miller_derm(rho, "r")
    dr_drho = _miller_derm(r_surf, "r") / rho_diff
    dr_dt = _miller_dermv(r_surf, theta_comn, "l", "e")
    dz_drho = _miller_derm(z_surf, "r") / rho_diff
    dz_dt = _miller_dermv(z_surf, theta_comn, "l", "o")
    jac = dr_drho * dz_dt - dz_drho * dr_dt

    drhodr = dz_dt / jac
    drhodz = -dr_dt / jac
    dt_dr = -dz_drho / jac
    dt_dz = dr_drho / jac

    dpsidrho_arr = -(r_geo / np.abs(2.0 * np.pi * qfac)) * np.abs(
        2.0 * cumulative_trapezoid(jac / r_surf, theta_comn)[:, -1]
    )
    dpsidrho = float(dpsidrho_arr[1])
    f_pol = np.ones((3,), dtype=np.float64) * r_geo
    drhodpsi = 1.0 / dpsidrho
    dpdpsi = dpdrho * drhodpsi

    psi = np.array([1.0 - delrho / drhodpsi, 1.0, 1.0 + delrho / drhodpsi])
    psi_diff = _miller_derm(psi, "r")
    dtdr_geo = (
        np.sign(psi_diff) * (dt_dr * drhodr + dt_dz * drhodz) / np.sqrt(drhodr**2 + drhodz**2)
    )

    b_p = np.abs(dpsidrho) * np.array(
        [np.sqrt(drhodr[i] ** 2 + drhodz[i] ** 2) / r_surf[i] for i in range(3)]
    )
    b_t = np.array([np.reshape(f_pol, (-1, 1))[i] / r_surf[i] for i in range(3)])
    b_tot = np.sqrt(b_p**2 + b_t**2)
    gradpar_0 = (
        1.0
        / (r_surf * b_tot)
        * np.array(
            [np.abs(dpsidrho_arr[i]) * np.sqrt(drhodr[i] ** 2 + drhodz[i] ** 2) for i in range(3)]
        )
        * (1.0 / _miller_dermv(l_arc, np.ones((3, ntgrid)) * theta, "l", "o"))
    )

    theta_st = np.zeros((3, ntgrid), dtype=np.float64)
    for i in range(3):
        theta_st[i, 1:] = cumulative_trapezoid(
            np.abs(np.reshape(f_pol, (-1, 1))[i] * (1.0 / dpsidrho_arr[i]) * jac[i] / r_surf[i]),
            theta_comn[i],
        )
        theta_st[i, 1:] = np.pi * (theta_st[i, 1:] / theta_st[i, -1])

    spl1 = InterpolatedUnivariateSpline(theta_st[1], theta)
    theta_st_new = theta_st
    theta_comn_new = np.zeros((3, ntgrid), dtype=np.float64)
    for i in range(3):
        r_surf[i] = np.interp(theta_st_new[i], theta_st[i], r_surf[i])
        z_surf[i] = np.interp(theta_st_new[i], theta_st[i], z_surf[i])
        b_tot[i] = np.interp(theta_st_new[i], theta_st[i], b_tot[i])
        b_p[i] = np.interp(theta_st_new[i], theta_st[i], b_p[i])
        gradpar_0[i] = np.interp(theta_st_new[i], theta_st[i], gradpar_0[i])
        dtdr_geo[i] = np.interp(theta_st_new[i], theta_st[i], dtdr_geo[i])
        theta_comn_new[i] = np.arctan2(z_surf[i], r_surf[i] - r_mag_ax)

    l_st = np.zeros((3, ntgrid), dtype=np.float64)
    for i in range(3):
        l_st[i, 1:] = np.cumsum(np.sqrt(np.diff(r_surf[i]) ** 2 + np.diff(z_surf[i]) ** 2))
    dr_dpsi = _miller_derm(r_surf, "r") / psi_diff
    dr_dt = _miller_dermv(r_surf, theta_st_new, "l", "e")
    dz_dpsi = _miller_derm(z_surf, "r") / psi_diff
    dz_dt = _miller_dermv(z_surf, theta_st_new, "l", "o")
    jac = dr_dpsi * dz_dt - dz_dpsi * dr_dt
    dpsidr = dz_dt / jac
    dpsidz = -dr_dt / jac

    dl = np.sqrt(_miller_derm(r_surf, "l", "e") ** 2 + _miller_derm(z_surf, "l", "o") ** 2)
    phi_n = np.zeros((3, ntgrid), dtype=np.float64)
    for i in range(3):
        dr_j = _miller_derm(r_surf[i, :], "l", "e")[0]
        dz_j = _miller_derm(z_surf[i, :], "l", "o")[0]
        phi_tmp = np.arctan2(dz_j, dr_j)
        phi_n[i, :] = np.concatenate(
            (phi_tmp[phi_tmp >= 0] - np.pi / 2.0, phi_tmp[phi_tmp < 0] + 3.0 * np.pi / 2.0)
        )
    u_ml = np.arctan2(_miller_derm(z_surf, "l", "o"), _miller_derm(r_surf, "l", "e"))
    r_c = dl / _miller_derm(phi_n, "l", "o")

    b_p_ex = _miller_nperiod_extend(np.abs(b_p[1]), nperiod, istheta=0, par="e")
    r_ex = _miller_nperiod_extend(r_surf[1], nperiod, istheta=0, par="e")
    z_ex = _miller_nperiod_extend(z_surf[1], nperiod, istheta=0, par="o")
    theta_st_new_ex = _miller_nperiod_extend(theta_st_new[1], nperiod, istheta=1)
    u_ml_ex = _miller_nperiod_extend(u_ml[1], nperiod)
    r_c_ex = _miller_nperiod_extend(r_c[1], nperiod)
    dl_ex = _miller_nperiod_extend(dl[1], nperiod)
    diffrho = _miller_derm(rho, "r")

    a_s = -(
        2.0 * qfac[1] / f_pol[1] * theta_st_new_ex
        + 2.0
        * f_pol[1]
        * qfac[1]
        * cumulative_trapezoid(1.0 / (r_ex**2 * b_p_ex**2), theta_st_new_ex, initial=0.0)
    )
    b_s = -(2.0 * qfac[1] * cumulative_trapezoid(1.0 / (b_p_ex**2), theta_st_new_ex, initial=0.0))
    c_s = (
        2.0
        * qfac[1]
        * cumulative_trapezoid(
            (2.0 * np.sin(u_ml_ex) / r_ex - 2.0 / r_c_ex) * 1.0 / (r_ex * b_p_ex),
            theta_st_new_ex,
            initial=0.0,
        )
    )
    dfdpsi = (
        -s_hat_input
        / (
            rho[1]
            * (psi_diff[1] / diffrho[1])
            * (1.0 / (2.0 * np.pi * qfac[1] * (2 * nperiod - 1)))
        )
        - (b_s[-1] * dpdpsi - c_s[-1])
    ) / a_s[-1]
    df_val = float((dfdpsi * (psi_diff[1] / 2.0)).item())
    f_pol[0], f_pol[2] = f_pol[1] - df_val, f_pol[1] + df_val

    dpsi_dr = np.sign(psi_diff) * np.sqrt(dpsidr**2 + dpsidz**2)
    b_p = np.abs(dpsi_dr) / r_surf
    b_t = np.array([np.reshape(f_pol, (-1, 1))[i] / r_surf[i] for i in range(3)])
    b2 = b_p**2 + b_t**2
    b_tot = np.sqrt(b2)

    b_p_ex = _miller_nperiod_extend(b_p[1], nperiod, istheta=0, par="e")
    b_ex = _miller_nperiod_extend(b_tot[1], nperiod, istheta=0, par="e")
    b2_ex = _miller_nperiod_extend(b2[1], nperiod, istheta=0, par="e")
    diffq = _miller_derm(qfac, "r")
    db2l_ex = _miller_derm(b_ex**2, "l")[0]

    dpsi_dr_ex = _miller_nperiod_extend(dpsi_dr[1], nperiod)
    gds22 = (diffq / diffrho) ** 2 * np.abs(dpsi_dr_ex) ** 2
    grho = drhodpsi * dpsi_dr_ex
    dqdr = diffq * dpsi_dr_ex / psi_diff
    dt_st_l_dl_ex = _miller_nperiod_extend(
        1.0 / _miller_dermv(l_st, theta_st_new, "l", par="o")[1],
        nperiod,
        istheta=0,
        par="e",
    )

    gradpar_ex = -1.0 / (r_ex * b_ex) * dpsi_dr_ex * dt_st_l_dl_ex
    aprime_bish = -r_ex * b_p_ex * (a_s * dfdpsi + b_s * dpdpsi - c_s) / (2.0 * abs(drhodpsi))
    gds21 = diffq / diffrho * (-dpsi_dr_ex) * aprime_bish
    dtdr_st_ex = (aprime_bish * drhodpsi - dqdr * theta_st_new_ex) / np.reshape(qfac, (-1, 1))
    gds2 = (psi_diff / diffrho) ** 2 * (
        1.0 / r_ex**2
        + (dqdr * theta_st_new_ex) ** 2
        + (np.reshape(qfac, (-1, 1))) ** 2 * (dtdr_st_ex**2 + dt_st_l_dl_ex**2)
        + 2.0 * np.reshape(qfac, (-1, 1)) * dqdr * theta_st_new_ex * dtdr_st_ex
    )
    gbdrift0 = 1.0 / (b2_ex**2) * dpsidrho * f_pol[1] / r_ex * (dqdr[1] * db2l_ex / dl_ex)
    dbdr = (
        b_p_ex
        / b_ex
        * (-b_p_ex / r_c_ex + dpdpsi * r_ex - f_pol[1] ** 2 * np.sin(u_ml_ex) / (r_ex**3 * b_p_ex))
    )
    gbdrift = (
        1.0
        / np.abs(drhodpsi * b_ex**3)
        * (
            2.0 * b2_ex * dbdr / dpsi_dr_ex
            + aprime_bish * drhodpsi * f_pol[1] / r_ex * db2l_ex / dl_ex / b_ex
        )
    )
    cvdrift = 1.0 / np.abs(drhodpsi * b_ex**3) * (2.0 * b_ex * dpdpsi) + gbdrift

    # Equal-arc transformation (gx_geo.py:L448-471)
    mask_pi = theta_st_new_ex <= np.pi
    gradpar_lim = gradpar_ex[mask_pi]
    b_lim = b_ex[mask_pi]
    b_p_lim = b_p_ex[mask_pi]
    theta_lim = theta_st_new_ex[mask_pi]
    l_eqarc = cumulative_trapezoid(b_p_lim / (b_lim * gradpar_lim), theta_lim, initial=0.0)
    gradpar_eqarc = np.pi / cumulative_trapezoid(1.0 / gradpar_lim, theta_lim, initial=0.0)[-1]

    theta_eqarc = cumulative_trapezoid(b_lim / b_p_lim * gradpar_eqarc, l_eqarc, initial=0.0)
    theta_eqarc_new = np.linspace(0.0, np.pi, ntgrid)
    theta_eqarc_ex = _miller_nperiod_extend(theta_eqarc, nperiod, istheta=1)
    theta_eqarc_new_ex = _miller_nperiod_extend(theta_eqarc_new, nperiod, istheta=1)

    gradpar_eq = np.interp(
        theta_eqarc_new_ex, theta_eqarc_ex, gradpar_eqarc * np.ones(len(theta_eqarc_ex))
    )
    r_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, r_ex)
    gds21_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gds21[1])
    gds22_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gds22[1])
    gds2_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gds2[1])
    grho_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, grho)
    gbdrift0_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gbdrift0)
    b_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, b_ex)
    cvdrift_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, cvdrift)
    gbdrift_eq = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gbdrift)

    _ = spl1  # matches gx_geo.py sequence
    _ = z_ex

    gradpar_ball = _miller_reflect_n_append(gradpar_eq, "e")
    theta_ball = _miller_reflect_n_append(theta_eqarc_new_ex, "o")
    cvdrift_ball = _miller_reflect_n_append(cvdrift_eq, "e")
    gbdrift_ball = _miller_reflect_n_append(gbdrift_eq, "e")
    gbdrift0_ball = _miller_reflect_n_append(gbdrift0_eq, "o")
    b_ball = _miller_reflect_n_append(b_eq, "e")
    gds2_ball = _miller_reflect_n_append(gds2_eq, "e")
    gds21_ball = _miller_reflect_n_append(gds21_eq, "o")
    gds22_ball = _miller_reflect_n_append(gds22_eq, "e")
    grho_ball = _miller_reflect_n_append(grho_eq, "e")
    rplot_ball = _miller_reflect_n_append(r_eq, "e")

    ntheta = len(theta_ball)
    # Format through exact eik.out representation used by C++ gx (gx_geo.py:L566-594)
    buf = io.StringIO()
    buf.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q\n")
    ntgrid_hdr = (ntheta - 1) // 2
    ntheta_hdr = (ntheta - 1) // (2 * nperiod - 1)
    drhodpsi_hdr = abs(1.0 / float(dpsidrho))
    rmaj_hdr = float((np.max(rplot_ball) + np.min(rplot_ball)) / 2.0)
    kxfac_hdr = abs(float(qfac[1] / rhoc * dpsidrho))
    q_hdr = float(qfac[1])
    buf.write(
        f"  {ntgrid_hdr:d}    {nperiod:d}    {ntheta_hdr:d}"
        f"   {drhodpsi_hdr:0.3f}   {rmaj_hdr:0.1f}    {s_hat_input:.7e}"
        f"   {kxfac_hdr:.1f}   {q_hdr:.4f}\n"
    )
    buf.write("gbdrift gradpar grho tgrid\n")
    for i in range(ntheta):
        buf.write(
            f"    {gbdrift_ball[i]:.9e}    {gradpar_ball[i]:.9e}"
            f"    {grho_ball[i]:.9e}    {theta_ball[i]:.9e}\n"
        )
    buf.write("cvdrift gds2 bmag tgrid\n")
    for i in range(ntheta):
        buf.write(
            f"    {cvdrift_ball[i]:.9e}    {gds2_ball[i]:.9e}"
            f"    {b_ball[i]:.12e}    {theta_ball[i]:.9e}\n"
        )
    buf.write("gds21 gds22 tgrid\n")
    for i in range(ntheta):
        buf.write(f"    {gds21_ball[i]:.9e}    {gds22_ball[i]:.9e}    {theta_ball[i]:.9e}\n")
    buf.write("cvdrift0 gbdrift0 tgrid\n")
    for i in range(ntheta):
        buf.write(f"    {gbdrift0_ball[i]:.9e}    {gbdrift0_ball[i]:.9e}    {theta_ball[i]:.9e}\n")

    return _load_eik_out_stream(buf.getvalue().splitlines(), cfg, dtype=dtype)


# ---------------------------------------------------------------------------
# Stellarator VMEC wout_*.nc -> .eik.nc Flux-Tube Converter
# Ported from geometry_modules/vmec/src/{solver.cpp,geometric_coefficients.cpp}
# ---------------------------------------------------------------------------


def build_vmec_eik_nc(
    cfg: GXConfig,
    wout_path: str | Path,
    out_nc_path: str | Path,
    *,
    vmec_surface_option: int = 0,
) -> Path:
    """Convert a 3D stellarator VMEC ``wout_*.nc`` equilibrium to a GX ``.eik.nc`` flux-tube file.

    Faithfully ports ``convert_VMEC_to_GX`` (``solver.cpp`` + ``geometric_coefficients.cpp``).
    """
    wout_p = Path(wout_path)
    out_p = Path(out_nc_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    nzgrid = cfg.nz // 2
    n_pts = 2 * nzgrid + 1

    with netCDF4.Dataset(wout_p, "r") as ds:
        ns = int(np.asarray(ds.variables["ns"][:]).item())
        nfp = int(np.asarray(ds.variables["nfp"][:]).item())
        mnmax = int(np.asarray(ds.variables["mnmax"][:]).item())
        mnmax_nyq = int(np.asarray(ds.variables["mnmax_nyq"][:]).item())
        signgs = int(np.asarray(ds.variables["signgs"][:]).item())
        lasym = bool(np.asarray(ds.variables["lasym__logical__"][:]).item())
        aminor_p = float(np.asarray(ds.variables["Aminor_p"][:]).item())

        phi_vmec = np.asarray(ds.variables["phi"][:], dtype=np.float64)
        iotaf = np.asarray(ds.variables["iotaf"][:], dtype=np.float64)
        iotas = np.asarray(ds.variables["iotas"][:], dtype=np.float64)
        presf = np.asarray(ds.variables["presf"][:], dtype=np.float64)

        xm = np.asarray(ds.variables["xm"][:], dtype=np.float64)
        xn = np.asarray(ds.variables["xn"][:], dtype=np.float64)
        xm_nyq = np.asarray(ds.variables["xm_nyq"][:], dtype=np.float64)
        xn_nyq = np.asarray(ds.variables["xn_nyq"][:], dtype=np.float64)

        rmnc = np.asarray(ds.variables["rmnc"][:], dtype=np.float64)  # (ns, mnmax)
        zmns = np.asarray(ds.variables["zmns"][:], dtype=np.float64)  # (ns, mnmax)
        lmns = np.asarray(ds.variables["lmns"][:], dtype=np.float64)  # (ns, mnmax)
        bmnc = np.asarray(ds.variables["bmnc"][:], dtype=np.float64)  # (ns, mnmax_nyq)
        gmnc = np.asarray(ds.variables["gmnc"][:], dtype=np.float64)
        bsupumnc = np.asarray(ds.variables["bsupumnc"][:], dtype=np.float64)
        bsupvmnc = np.asarray(ds.variables["bsupvmnc"][:], dtype=np.float64)
        bsubumnc = np.asarray(ds.variables["bsubumnc"][:], dtype=np.float64)
        bsubvmnc = np.asarray(ds.variables["bsubvmnc"][:], dtype=np.float64)
        bsubsmns = np.asarray(ds.variables["bsubsmns"][:], dtype=np.float64)

        lmnc = np.asarray(ds.variables["lmnc"][:], dtype=np.float64) if lasym else None

    edge_toroidal_flux_over_2pi = signgs * (phi_vmec[-1] / (2.0 * math.pi))
    sign_psi = math.copysign(1.0, phi_vmec[-1])
    l_ref = aminor_p
    b_ref = abs((2.0 * edge_toroidal_flux_over_2pi) / (l_ref * l_ref))

    s_full = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    ds_step = float(s_full[1] - s_full[0])

    desired_s = cfg.torflux
    if vmec_surface_option == 0:
        s_used = desired_s
    elif vmec_surface_option == 1:
        s_used = float(s_half[int(np.argmin((s_half - desired_s) ** 2))])
    elif vmec_surface_option == 2:
        s_used = float(s_full[int(np.argmin((s_full - desired_s) ** 2))])
    else:
        raise ValueError(f"vmec_surface_option must be 0, 1, or 2; got {vmec_surface_option}.")

    if not (0.0 <= s_used <= 1.0):
        raise ValueError(f"normalized_toroidal_flux_used={s_used} must lie in [0, 1].")

    if s_used == 1.0:
        idx_full = (ns - 2, ns - 1)
        w_full = (0.0, 1.0)
    else:
        i0 = min(int(math.floor(s_used * ns)), ns - 2)
        w0 = i0 - s_used * (ns - 1.0) + 1.0
        idx_full = (i0, i0 + 1)
        w_full = (w0, 1.0 - w0)

    if s_used < s_half[0]:
        idx_half = (1, 2)
        w0 = (s_half[1] - s_used) / (s_half[1] - s_half[0])
    elif s_used > s_half[-1]:
        idx_half = (ns - 2, ns - 1)
        w0 = (s_half[-1] - s_used) / (s_half[-1] - s_half[-2])
    elif s_used == s_half[-1]:
        idx_half = (ns - 2, ns - 1)
        w0 = 0.0
    else:
        i0 = int(math.floor(s_used * (ns - 1) + 0.5))
        if i0 < 2:
            i0 = 2
        idx_half = (i0, i0 + 1)
        w0 = i0 - s_used * (ns - 1.0) + 0.5
    w_half = (w0, 1.0 - w0)

    iota = float(iotas[idx_half[0]] * w_half[0] + iotas[idx_half[1]] * w_half[1])
    safety_factor_q = 1.0 / iota

    d_iota_ds_half = np.zeros(ns, dtype=np.float64)
    d_pres_ds_half = np.zeros(ns, dtype=np.float64)
    d_iota_ds_half[1:] = (iotaf[1:] - iotaf[:-1]) / ds_step
    d_pres_ds_half[1:] = (presf[1:] - presf[:-1]) / ds_step

    d_iota_ds = float(
        d_iota_ds_half[idx_half[0]] * w_half[0] + d_iota_ds_half[idx_half[1]] * w_half[1]
    )
    d_pressure_ds = float(
        d_pres_ds_half[idx_half[0]] * w_half[0] + d_pres_ds_half[idx_half[1]] * w_half[1]
    )
    shat = (-2.0 * s_used / iota) * d_iota_ds

    alpha = cfg.alpha
    zeta_center = -alpha / iota

    # Interpolate Lambda harmonics on half grid for PEST -> VMEC root solve (solver.cpp:L9-98)
    lmns_interp = w_half[0] * lmns[idx_half[0], :] + w_half[1] * lmns[idx_half[1], :]
    lmnc_interp = (
        w_half[0] * lmnc[idx_half[0], :] + w_half[1] * lmnc[idx_half[1], :]
        if (lasym and lmnc is not None)
        else None
    )

    def _pest_to_vmec(th_pest_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        out_th = np.zeros(len(th_pest_arr), dtype=np.float64)
        for iz_idx in range(len(th_pest_arr)):
            z0 = float(z_arr[iz_idx])
            if z0 == 0.0 and alpha == 0.0 and not lasym:
                out_th[iz_idx] = 0.0
                continue
            target = float(alpha + iota * z0)

            def _residual(th_try: float, _t: float = target, _z: float = z0) -> float:
                ang_r = xm * th_try - xn * _z
                res = th_try - _t + float(np.dot(lmns_interp, np.sin(ang_r)))
                if lmnc_interp is not None:
                    res += float(np.dot(lmnc_interp, np.cos(ang_r)))
                return res

            out_th[iz_idx] = brentq(_residual, target - 1.0, target + 1.0, xtol=1e-12, rtol=1e-12)
        return out_th

    # Radial derivatives on full / half meshes (geometric_coefficients.cpp:L489-540)
    db_ds_mnc = np.zeros((ns, mnmax_nyq), dtype=np.float64)
    db_ds_mnc[1:-1, :] = (bmnc[2:, :] - bmnc[1:-1, :]) / ds_step
    db_ds_mnc[0, :] = db_ds_mnc[1, :]
    db_ds_mnc[-1, :] = db_ds_mnc[-2, :]

    dr_ds_mnc = np.zeros((ns, mnmax), dtype=np.float64)
    dr_ds_mnc[1:, :] = (rmnc[1:, :] - rmnc[:-1, :]) / ds_step
    dz_ds_mns = np.zeros((ns, mnmax), dtype=np.float64)
    dz_ds_mns[1:, :] = (zmns[1:, :] - zmns[:-1, :]) / ds_step

    dl_ds_mns = np.zeros((ns, mnmax), dtype=np.float64)
    dl_ds_mns[1:-1, :] = (lmns[2:, :] - lmns[1:-1, :]) / ds_step
    dl_ds_mns[0, :] = dl_ds_mns[1, :]
    dl_ds_mns[-1, :] = dl_ds_mns[-2, :]

    # Interpolated Fourier coefficients at s_used
    bmnc_h = w_half[0] * bmnc[idx_half[0]] + w_half[1] * bmnc[idx_half[1]]
    gmnc_h = w_half[0] * gmnc[idx_half[0]] + w_half[1] * gmnc[idx_half[1]]
    bsupu_h = w_half[0] * bsupumnc[idx_half[0]] + w_half[1] * bsupumnc[idx_half[1]]
    bsupv_h = w_half[0] * bsupvmnc[idx_half[0]] + w_half[1] * bsupvmnc[idx_half[1]]
    bsubu_h = w_half[0] * bsubumnc[idx_half[0]] + w_half[1] * bsubumnc[idx_half[1]]
    bsubv_h = w_half[0] * bsubvmnc[idx_half[0]] + w_half[1] * bsubvmnc[idx_half[1]]
    bsubs_f = w_full[0] * bsubsmns[idx_full[0]] + w_full[1] * bsubsmns[idx_full[1]]
    db_ds_f = w_full[0] * db_ds_mnc[idx_full[0]] + w_full[1] * db_ds_mnc[idx_full[1]]

    rmnc_f = w_full[0] * rmnc[idx_full[0]] + w_full[1] * rmnc[idx_full[1]]
    zmns_f = w_full[0] * zmns[idx_full[0]] + w_full[1] * zmns[idx_full[1]]
    lmns_h = lmns_interp
    dr_ds_h = w_half[0] * dr_ds_mnc[idx_half[0]] + w_half[1] * dr_ds_mnc[idx_half[1]]
    dz_ds_h = w_half[0] * dz_ds_mns[idx_half[0]] + w_half[1] * dz_ds_mns[idx_half[1]]
    dl_ds_f = w_full[0] * dl_ds_mns[idx_full[0]] + w_full[1] * dl_ds_mns[idx_full[1]]

    effective_npol = float(cfg.npol)
    if cfg.boundary.strip().lower() == "fix aspect" and abs(shat) >= cfg.zero_shat_threshold:
        # Match gx_geo_vmec.py:L1131-1240 (`flux_tube_cut == "aspect"`): cut field line
        # where `2*shat*gds21/gds22 * (x0/y0)` crosses a nonzero integer after `npol_min*pi`.
        scan_npol = (
            max(4.0, 2.0 * float(cfg.npol_min))
            if cfg.npol_min is not None
            else max(1.0, float(cfg.npol))
        )
        n_scan = max(4 * nzgrid + 1, 513)
        th_scan = np.linspace(-scan_npol * math.pi, scan_npol * math.pi, n_scan, dtype=np.float64)
        ze_scan = (th_scan - alpha) / iota
        th_v_scan = _pest_to_vmec(th_scan, ze_scan)
        ang_nyq_s = xm_nyq[:, None] * th_v_scan[None, :] - xn_nyq[:, None] * ze_scan[None, :]
        sqrt_g_s = gmnc_h @ np.cos(ang_nyq_s)
        ang_s = xm[:, None] * th_v_scan[None, :] - xn[:, None] * ze_scan[None, :]
        cos_s = np.cos(ang_s)
        sin_s = np.sin(ang_s)
        r_s = rmnc_f @ cos_s
        dr_dth_s = -(rmnc_f * xm) @ sin_s
        dr_dze_s = (rmnc_f * xn) @ sin_s
        dz_dth_s = (zmns_f * xm) @ cos_s
        dz_dze_s = -(zmns_f * xn) @ cos_s
        dl_dth_s = (lmns_h * xm) @ cos_s
        dl_dze_s = -(lmns_h * xn) @ cos_s
        dr_ds_s = dr_ds_h @ cos_s
        dz_ds_s = dz_ds_h @ sin_s
        dl_ds_s = dl_ds_f @ sin_s

        cz_s, sz_s = np.cos(ze_scan), np.sin(ze_scan)
        dx_dth_s, dy_dth_s = dr_dth_s * cz_s, dr_dth_s * sz_s
        dx_dze_s = dr_dze_s * cz_s - r_s * sz_s
        dy_dze_s = dr_dze_s * sz_s + r_s * cz_s
        dx_ds_s, dy_ds_s = dr_ds_s * cz_s, dr_ds_s * sz_s

        gs_x = (dy_dth_s * dz_dze_s - dz_dth_s * dy_dze_s) / sqrt_g_s
        gs_y = (dz_dth_s * dx_dze_s - dx_dth_s * dz_dze_s) / sqrt_g_s
        gs_z = (dx_dth_s * dy_dze_s - dy_dth_s * dx_dze_s) / sqrt_g_s
        gth_x = (dy_dze_s * dz_ds_s - dz_dze_s * dy_ds_s) / sqrt_g_s
        gth_y = (dz_dze_s * dx_ds_s - dx_dze_s * dz_ds_s) / sqrt_g_s
        gth_z = (dx_dze_s * dy_ds_s - dy_dze_s * dx_ds_s) / sqrt_g_s
        gze_x, gze_y = -sz_s / r_s, cz_s / r_s

        gp_x = gs_x * edge_toroidal_flux_over_2pi
        gp_y = gs_y * edge_toroidal_flux_over_2pi
        gp_z = gs_z * edge_toroidal_flux_over_2pi
        cs_s = dl_ds_s - (ze_scan - zeta_center) * d_iota_ds
        cth_s = 1.0 + dl_dth_s
        cze_s = -iota + dl_dze_s
        ga_x = cs_s * gs_x + cth_s * gth_x + cze_s * gze_x
        ga_y = cs_s * gs_y + cth_s * gth_y + cze_s * gze_y
        ga_z = cs_s * gs_z + cth_s * gth_z

        gp_sq = gp_x**2 + gp_y**2 + gp_z**2
        ga_dot_gp = ga_x * gp_x + ga_y * gp_y + ga_z * gp_z
        gds21_s = ga_dot_gp * (shat / b_ref)
        gds22_s = gp_sq * ((shat * shat) / (l_ref * l_ref * b_ref * b_ref * s_used))

        x0_target = float(cfg.x0) if cfg.x0 > 0.0 else float(cfg.y0)
        jtwist_s = (2.0 * shat * gds21_s / gds22_s) / float(cfg.y0) * x0_target
        jtwist_spl = CubicSpline(th_scan, jtwist_s)
        vals = (
            [int(-cfg.jtwist), int(cfg.jtwist)]
            if cfg.jtwist != -1000
            else [k for k in range(-30, 31) if k != 0]
        )
        roots_list = [jtwist_spl.solve(float(v), extrapolate=False) for v in vals]
        crossings = np.sort(np.concatenate(roots_list)) if roots_list else np.array([])
        if cfg.npol_min is not None:
            valid_c = crossings[crossings > float(cfg.npol_min) * math.pi]
            if len(valid_c) > 0:
                effective_npol = float(valid_c[0] / math.pi)
        else:
            valid_c = crossings[(crossings > 0.0) & (crossings <= float(cfg.npol) * math.pi)]
            if len(valid_c) > 0:
                effective_npol = float(valid_c[-1] / math.pi)

    theta_pest = (
        effective_npol * math.pi * (np.arange(n_pts, dtype=np.float64) - nzgrid) / float(nzgrid)
    )
    zeta = (theta_pest - alpha) / iota
    theta_vmec = _pest_to_vmec(theta_pest, zeta)

    # Evaluate Nyquist and non-Nyquist Fourier sums on (n_pts,)
    ang_nyq = xm_nyq[:, None] * theta_vmec[None, :] - xn_nyq[:, None] * zeta[None, :]
    cos_nyq = np.cos(ang_nyq)
    sin_nyq = np.sin(ang_nyq)

    b_mag = bmnc_h @ cos_nyq
    db_dtheta = -(bmnc_h * xm_nyq) @ sin_nyq
    db_dzeta = (bmnc_h * xn_nyq) @ sin_nyq
    sqrt_g = gmnc_h @ cos_nyq
    _b_sup_theta = bsupu_h @ cos_nyq
    b_sup_zeta = bsupv_h @ cos_nyq
    b_sub_theta = bsubu_h @ cos_nyq
    b_sub_zeta = bsubv_h @ cos_nyq
    b_sub_s = bsubs_f @ sin_nyq
    db_ds = db_ds_f @ cos_nyq

    ang = xm[:, None] * theta_vmec[None, :] - xn[:, None] * zeta[None, :]
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    r_maj = rmnc_f @ cos_a
    dr_dtheta = -(rmnc_f * xm) @ sin_a
    dr_dzeta = (rmnc_f * xn) @ sin_a
    dz_dtheta = (zmns_f * xm) @ cos_a
    dz_dzeta = -(zmns_f * xn) @ cos_a
    dlambda_dtheta = (lmns_h * xm) @ cos_a
    dlambda_dzeta = -(lmns_h * xn) @ cos_a
    dr_ds = dr_ds_h @ cos_a
    dz_ds = dz_ds_h @ sin_a
    dlambda_ds = dl_ds_f @ sin_a

    # Dual basis in cylindrical -> Cartesian coordinates (geometric_coefficients.cpp:L802-879)
    cos_z = np.cos(zeta)
    sin_z = np.sin(zeta)
    dx_dtheta = dr_dtheta * cos_z
    dx_dzeta = dr_dzeta * cos_z - r_maj * sin_z
    dx_ds = dr_ds * cos_z
    dy_dtheta = dr_dtheta * sin_z
    dy_dzeta = dr_dzeta * sin_z + r_maj * cos_z
    dy_ds = dr_ds * sin_z

    grad_s_x = (dy_dtheta * dz_dzeta - dz_dtheta * dy_dzeta) / sqrt_g
    grad_s_y = (dz_dtheta * dx_dzeta - dx_dtheta * dz_dzeta) / sqrt_g
    grad_s_z = (dx_dtheta * dy_dzeta - dy_dtheta * dx_dzeta) / sqrt_g

    grad_th_x = (dy_dzeta * dz_ds - dz_dzeta * dy_ds) / sqrt_g
    grad_th_y = (dz_dzeta * dx_ds - dx_dzeta * dz_ds) / sqrt_g
    grad_th_z = (dx_dzeta * dy_ds - dy_dzeta * dx_ds) / sqrt_g

    grad_zeta_x = -sin_z / r_maj
    grad_zeta_y = cos_z / r_maj
    grad_zeta_z = np.zeros_like(r_maj)

    grad_psi_x = grad_s_x * edge_toroidal_flux_over_2pi
    grad_psi_y = grad_s_y * edge_toroidal_flux_over_2pi
    grad_psi_z = grad_s_z * edge_toroidal_flux_over_2pi

    c_s_fac = dlambda_ds - (zeta - zeta_center) * d_iota_ds
    c_th_fac = 1.0 + dlambda_dtheta
    c_ze_fac = -iota + dlambda_dzeta

    grad_alpha_x = c_s_fac * grad_s_x + c_th_fac * grad_th_x + c_ze_fac * grad_zeta_x
    grad_alpha_y = c_s_fac * grad_s_y + c_th_fac * grad_th_y + c_ze_fac * grad_zeta_y
    grad_alpha_z = c_s_fac * grad_s_z + c_th_fac * grad_th_z + c_ze_fac * grad_zeta_z

    sqrt_s = math.sqrt(s_used)
    b_cross_gs_dot_ga = (
        b_sub_zeta * (1.0 + dlambda_dtheta) - b_sub_theta * (dlambda_dzeta - iota)
    ) / sqrt_g
    b_cross_gb_dot_ga = (
        b_sub_s * db_dtheta * (dlambda_dzeta - iota)
        + b_sub_theta * db_dzeta * c_s_fac
        + b_sub_zeta * db_ds * (1.0 + dlambda_dtheta)
        - b_sub_zeta * db_dtheta * c_s_fac
        - b_sub_theta * db_ds * (dlambda_dzeta - iota)
        - b_sub_s * db_dzeta * (1.0 + dlambda_dtheta)
    ) / sqrt_g

    grad_psi_sq = grad_psi_x**2 + grad_psi_y**2 + grad_psi_z**2
    grad_alpha_sq = grad_alpha_x**2 + grad_alpha_y**2 + grad_alpha_z**2
    grad_alpha_dot_psi = (
        grad_alpha_x * grad_psi_x + grad_alpha_y * grad_psi_y + grad_alpha_z * grad_psi_z
    )

    bmag_pest = b_mag / b_ref
    gradpar_pest = (l_ref * b_sup_zeta * iota) / b_mag
    grho_pest = np.sqrt(grad_psi_sq) / (l_ref * b_ref * sqrt_s)
    gds2_pest = grad_alpha_sq * (l_ref * l_ref * s_used)
    gds21_pest = grad_alpha_dot_psi * (shat / b_ref)
    gds22_pest = grad_psi_sq * ((shat * shat) / (l_ref * l_ref * b_ref * b_ref * s_used))

    gbdrift_pest = sign_psi * 2.0 * b_ref * (l_ref**2) * sqrt_s * b_cross_gb_dot_ga / (b_mag**3)
    gbdrift0_pest = (
        sign_psi
        * ((b_sub_theta * db_dzeta - b_sub_zeta * db_dtheta) / sqrt_g)
        * ((edge_toroidal_flux_over_2pi * 2.0 * shat) / ((b_mag**3) * sqrt_s))
    )
    cvdrift_pest = gbdrift_pest + (
        sign_psi
        * 2.0
        * b_ref
        * (l_ref**2)
        * sqrt_s
        * _MU_0
        * d_pressure_ds
        * b_cross_gs_dot_ga
        / (b_mag**4)
    )
    cvdrift0_pest = gbdrift0_pest.copy()

    # Equal-arc reparametrization onto [-pi, pi] (get_GX_geo_arrays, L1537-1604)
    dtheta = float(theta_pest[1] - theta_pest[0])
    dtheta_pi = math.pi / float(nzgrid)
    gradpar_half = np.empty(2 * nzgrid, dtype=np.float64)
    gradpar_half[:-1] = 0.5 * (np.abs(gradpar_pest[:-2]) + np.abs(gradpar_pest[1:-1]))
    gradpar_half[-1] = gradpar_half[0]

    temp_grid = np.zeros(n_pts, dtype=np.float64)
    temp_grid[1:] = np.cumsum(dtheta / np.abs(gradpar_half))
    z_on_theta = temp_grid - temp_grid[nzgrid]
    desired_gradpar = math.pi / abs(float(z_on_theta[0]))
    z_on_theta = z_on_theta * desired_gradpar
    uniform_z = z_on_theta[0] + np.arange(n_pts, dtype=np.float64) * dtheta_pi
    domain_scaling_factor = abs(float(theta_pest[0] / uniform_z[0]))

    def _regrid_2nz(arr_pest: np.ndarray) -> np.ndarray:
        # Matches geometric_coefficients.cpp:L1595-1603 (splines first 2*nzgrid points)
        out = arr_pest.copy()
        spline = CubicSpline(z_on_theta, arr_pest, bc_type="natural")
        out[: 2 * nzgrid] = spline(uniform_z[: 2 * nzgrid])
        return out

    bmag_out = _regrid_2nz(bmag_pest)
    grho_out = _regrid_2nz(grho_pest)
    gds2_out = _regrid_2nz(gds2_pest)
    gds21_out = _regrid_2nz(gds21_pest)
    gds22_out = _regrid_2nz(gds22_pest)
    gbdrift_out = _regrid_2nz(gbdrift_pest)
    gbdrift0_out = _regrid_2nz(gbdrift0_pest)
    cvdrift_out = _regrid_2nz(cvdrift_pest)
    cvdrift0_out = _regrid_2nz(cvdrift0_pest)
    gradpar_out = np.full(n_pts, desired_gradpar, dtype=np.float64)

    with netCDF4.Dataset(out_p, "w", format="NETCDF4") as nc_out:
        nc_out.createDimension("z", n_pts)
        for s_name, s_val in (
            ("drhodpsi", 1.0),
            ("Rmaj", 1.0 / l_ref),
            ("shat", shat),
            ("kxfac", 1.0),
            ("q", safety_factor_q),
            ("scale", domain_scaling_factor),
            ("alpha", alpha),
            ("zeta_center", zeta_center),
            ("B_T", b_ref),
            ("aminor", l_ref),
        ):
            v = nc_out.createVariable(s_name, "f8")
            v.assignValue(float(s_val))
        v_nfp = nc_out.createVariable("nfp", "i4")
        v_nfp.assignValue(int(nfp))
        v_wout = nc_out.createVariable("wout", "i4")
        v_wout.setncattr("filename", wout_p.name)

        for a_name, a_val in (
            ("theta", uniform_z),
            ("bmag", bmag_out),
            ("gradpar", gradpar_out),
            ("grho", grho_out),
            ("gbdrift", gbdrift_out),
            ("gbdrift0", gbdrift0_out),
            ("cvdrift", cvdrift_out),
            ("cvdrift0", cvdrift0_out),
            ("gds2", gds2_out),
            ("gds21", gds21_out),
            ("gds22", gds22_out),
        ):
            var = nc_out.createVariable(a_name, "f8", ("z",))
            var[:] = a_val

    return out_p


def init_geometry(
    cfg: GXConfig,
    *,
    geo_path: str | Path | None = None,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[GXConfig, Grids, Geometry]:
    """Dispatch geometry initialization strictly matching ``init_geo`` (``geometry.cu:L13-164``)."""
    opt = cfg.geo_option.lower()
    if opt in ("s-alpha", "slab", "const-curv") or cfg.igeo == 0:
        return s_alpha_geometry(cfg, dtype=dtype)
    if opt == "miller":
        return miller_geometry(cfg, dtype=dtype)
    if opt == "nc" or cfg.igeo == 2:
        target = Path(geo_path) if geo_path is not None else Path(cfg.geofilename)
        if not target.is_absolute():
            if cfg.input_path is not None and (cfg.input_path.parent / target).is_file():
                target = cfg.input_path.parent / target
        return load_eik_nc(target, cfg, dtype=dtype)
    if opt == "eik" or cfg.igeo == 1:
        target = Path(geo_path) if geo_path is not None else Path(cfg.geofilename)
        if not target.is_absolute():
            if cfg.input_path is not None and (cfg.input_path.parent / target).is_file():
                target = cfg.input_path.parent / target
        return load_eik_out(target, cfg, dtype=dtype)
    if opt in ("vmec", "pyvmec", "vmec_c"):
        if (
            geo_path is not None
            and Path(geo_path).suffix == ".nc"
            and "wout" not in Path(geo_path).name
        ):
            return load_eik_nc(geo_path, cfg, dtype=dtype)
        wout_target = Path(geo_path) if geo_path is not None else Path(cfg.vmec_file)
        if not wout_target.is_absolute() and cfg.input_path is not None:
            candidate = cfg.input_path.parent / wout_target
            if candidate.is_file():
                wout_target = candidate
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            eik_nc = build_vmec_eik_nc(cfg, wout_target, Path(tmpdir) / f"{cfg.run_name}.eik.nc")
            return load_eik_nc(eik_nc, cfg, dtype=dtype)
    raise ValueError(
        f"Unsupported geo_option={cfg.geo_option!r}. "
        "Supported options: {'s-alpha', 'slab', 'const-curv', 'miller', 'nc', 'eik', 'vmec'}."
    )
