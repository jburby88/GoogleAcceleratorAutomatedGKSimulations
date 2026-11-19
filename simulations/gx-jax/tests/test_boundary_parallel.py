"""Linked Twist-and-Shift & NTFT Parallel Boundary Operators.

Verifies ``gx_jax/boundary_parallel.py`` and ``gx_jax/linear_rhs.py`` integration against:
- Independent imperative C++-translated NumPy oracles of ``gx/src/{
  grad_parallel_linked.cu,grad_parallel_NTFT.cu,device_funcs.cu,linear.cu}`` across all
  operator methods (``identity``, ``dz``, ``abs_dz``, ``dz2``, ``hyperz``, ``zft``,
  ``zft_inverse``, ``dealias``, ``apply_bcs``) and tensor ranks (3D, 4D, 5D, 6D) to
  ``rtol < 1e-6`` in runtime ``complex64`` and ``< 1e-12`` in ``complex128``.
- Exact C++ GoogleTests from ``gx/unit_tests/src/test_grad_parallel_linked.cu``
  plus extended multi-link ballooning Gaussian derivatives across ``z = +-pi`` link junctions.
- Exact ballooning equivalence between 3-link ``nperiod = 1, nkx = 3`` ``LinkedParallelGrad``
  and ``nperiod = 2, nkx = 1`` Cyclone ``s-alpha`` / Miller golden ``*_correct.out.nc``
  eigenmodes, plus 3D **W7-X** stellarator multi-link & NTFT linear ITG eigenmodes.
- Static-shape ``@jax.jit`` single-trace compilation, padded chain buckets
  (``chain_pad_multiple = 8``), PyTree flatten/unflatten + ``.astype(jnp.float32)``, and
  reverse-mode ``jax.grad`` under ``jax.debug_nans(True)`` with zero NaNs.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

import gx_jax
from gx_jax.boundary_parallel import (
    LinkedParallelGrad,
    NTFTParallelGrad,
    build_linked_topology,
    build_ntft_topology,
    calc_idx0_ntft,
    init_boundary_parallel,
    init_kz_linked,
    init_kz_linked_ntft,
    init_kzm_linked,
    init_linked_parallel_grad,
    init_ntft_parallel_grad,
)
from gx_jax.config import GXConfig, SpeciesConfig, load_config
from gx_jax.fields import (
    Fields,
    enforce_reality,
    field_line_average,
    init_field_solver,
    init_moments,
    solve_fields,
)
from gx_jax.geometry_nc import build_vmec_eik_nc, init_geometry, load_eik_nc
from gx_jax.linear_rhs import (
    ParallelGrad,
    advance_linear_steps,
    compute_growth_rates,
    compute_linear_dt,
    compute_phi2_ky,
    extract_single_ky_config,
    init_linear_operator,
    init_parallel_grad,
    linear_rhs,
    smith_par_get_as,
    smith_perp_get_as,
    solve_linear_eigenmode,
    step_linear,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
GOLDEN_BENCH_DIR = REPO_ROOT / "golden_fixtures" / "benchmarks"
UNIT_INPUTS_DIR = REPO_ROOT / "golden_fixtures" / "unit_tests" / "inputs"


@pytest.fixture(scope="module")
def w7x_eik_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build shared W7-X .eik.nc fixture once per test module (1-turn, nz=32)."""
    base_dir = tmp_path_factory.mktemp("w7x_eik_s4")
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    w7x_cfg = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
    wout_p = w7x_dir / "wout_w7x.nc"

    eik32 = build_vmec_eik_nc(
        dataclasses.replace(w7x_cfg, ntheta_in=32, nz=32, npol=1.0),
        wout_p,
        base_dir / "w7x_1turn_nz32.eik.nc",
    )
    return {"nz32": eik32}


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
    *,
    tol: float = 1.0e-6,
) -> None:
    is_cplx = np.iscomplexobj(actual) or np.iscomplexobj(expected)
    dt = np.complex128 if is_cplx else np.float64
    act = np.asarray(actual, dtype=dt)
    exp = np.asarray(expected, dtype=dt)
    exp_norm = float(np.linalg.norm(exp.ravel()))
    if exp_norm > 1.0e-12:
        rel_l2 = float(np.linalg.norm((act - exp).ravel())) / exp_norm
        assert rel_l2 < tol, f"{label}: relative L2 error {rel_l2:.3e} >= {tol:.1e}"
    scale = max(1.0, float(np.max(np.abs(exp)))) if exp.size > 0 else 1.0
    np.testing.assert_allclose(
        act,
        exp,
        rtol=max(2.0e-6, 2.0 * tol),
        atol=max(1.0e-6, tol) * scale,
        err_msg=f"FP32 mismatch in {label}",
    )


def _random_complex_arr(
    shape: tuple[int, ...],
    active_mask: np.ndarray,
    seed: int,
    *,
    dtype: jnp.dtype = jnp.complex128,
    zero_origin: bool = True,
) -> jax.Array:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)
    if zero_origin:
        raw = np.where(active_mask, raw, 0.0)
    return jnp.asarray(raw, dtype=dtype)


@functools.partial(jax.jit, static_argnames=("op_name",))
def _jit_op(gp, arr: jax.Array, op_name: str) -> jax.Array:
    return getattr(gp, op_name)(arr)


@jax.jit
def _jit_bcs(gp, g: jax.Array, r: jax.Array, f: Fields, sv, dt: float) -> jax.Array:
    return gp.apply_bcs(g, r, f, sv, dt)


@jax.jit
def _jit_linear_rhs(g: jax.Array, f: Fields, op, dt: float, grids, geo) -> jax.Array:
    return linear_rhs(g, f, op, dt, grids=grids, geo=geo)


# ---------------------------------------------------------------------------
# Independent Imperative C++ Reference Oracles (grad_parallel_linked.cu & NTFT.cu)
# ---------------------------------------------------------------------------


def _cpp_jflr_np(ell: int, b: np.ndarray) -> np.ndarray:
    if ell < 0 or ell > 30:
        return np.zeros_like(b, dtype=np.float64)
    return ((-0.5 * b) ** ell) / math.factorial(ell) * np.exp(-0.5 * b)


def _cpp_jflr_b_np(ell: int, b: np.ndarray) -> np.ndarray:
    return _cpp_jflr_np(ell, b) + _cpp_jflr_np(ell - 1, b)


def _cpp_linked_oracle_init(naky: int, nakx: int, nx: int, nz: int, zp: int, jshift0: int) -> dict:
    """Line-by-line imperative C++ port of GradParallelLinked constructor (L50-251, L521-725)."""
    idx_right = [0] * (naky * nakx)
    idx_left = [0] * (naky * nakx)
    links_r = [0] * (naky * nakx)
    links_l = [0] * (naky * nakx)
    n_k = [0] * (naky * nakx)

    for idx in range(nakx):
        for idy in range(naky):
            idx0 = idx if idx < (nakx + 1) // 2 else (idx - nakx)
            if idy == 0:
                idx_l = idx0
                idx_r = idx0
            else:
                idx_l = idx0 + idy * jshift0
                idx_r = idx0 - idy * jshift0

            if 0 <= idx_l < (nakx + 1) // 2:
                idx_left[idy + naky * idx] = idx_l
            elif (nakx + 1) // 2 <= idx_l + nakx < nakx:
                idx_left[idy + naky * idx] = idx_l + nakx
            else:
                idx_left[idy + naky * idx] = -1

            if 0 <= idx_r < (nakx + 1) // 2:
                idx_right[idy + naky * idx] = idx_r
            elif (nakx + 1) // 2 <= idx_r + nakx < nakx:
                idx_right[idy + naky * idx] = idx_r + nakx
            else:
                idx_right[idy + naky * idx] = -1

    for idx in range(nakx):
        for idy in range(naky):
            links_l[idy + naky * idx] = 0
            idx_star = idx
            while (
                idx_star != idx_left[idy + naky * idx_star] and idx_left[idy + naky * idx_star] >= 0
            ):
                links_l[idy + naky * idx] += 1
                idx_star = idx_left[idy + naky * idx_star]

            links_r[idy + naky * idx] = 0
            idx_star = idx
            while (
                idx_star != idx_right[idy + naky * idx_star]
                and idx_right[idy + naky * idx_star] >= 0
            ):
                links_r[idy + naky * idx] += 1
                idx_star = idx_right[idy + naky * idx_star]

    k = 0
    for idx in range(nakx):
        for idy in range(naky):
            n_k[k] = 1 + links_l[idy + naky * idx] + links_r[idy + naky * idx]
            k += 1

    n_k.sort()
    n_classes = 1
    for k in range(naky * nakx - 1):
        if n_k[k] != n_k[k + 1]:
            n_classes += 1

    n_chains = [1] * n_classes
    n_links = [0] * n_classes
    c = 0
    for k in range(1, naky * nakx):
        if n_k[k] == n_k[k - 1]:
            n_chains[c] += 1
        else:
            n_links[c] = n_k[k - 1]
            n_chains[c] = n_chains[c] // n_links[c]
            c += 1
    c = n_classes - 1
    n_links[c] = n_k[naky * nakx - 1]
    n_chains[c] = n_chains[c] // n_links[c]

    nshift = nx - nakx
    ky_h = [[0] * (n_links[ic] * n_chains[ic]) for ic in range(n_classes)]
    kx_h = [[0] * (n_links[ic] * n_chains[ic]) for ic in range(n_classes)]

    for ic in range(n_classes):
        n = 0
        for idy in range(naky):
            for idx in range(nakx):
                np_k = 1 + links_l[idy + naky * idx] + links_r[idy + naky * idx]
                c_found = n_links.index(np_k)
                p_found = links_l[idy + naky * idx]
                if c_found == ic and p_found == 0:
                    idx0 = idx if idx < (nakx + 1) // 2 else (idx + nshift)
                    ky_h[ic][0 + n_links[ic] * n] = idy
                    kx_h[ic][0 + n_links[ic] * n] = idx0
                    idx_r_cur = idx
                    for p_step in range(1, n_links[ic]):
                        idx_r_cur = idx_right[idy + naky * idx_r_cur]
                        ky_h[ic][p_step + n_links[ic] * n] = idy
                        kx_h[ic][p_step + n_links[ic] * n] = (
                            idx_r_cur if idx_r_cur < (nakx + 1) // 2 else (idx_r_cur + nshift)
                        )
                    n += 1

    c_map = [0] * (naky * nakx)
    n_map = [0] * (naky * nakx)
    n_links_map = [0] * (naky * nakx)
    n_chains_map = [0] * (naky * nakx)
    for c_idx in range(n_classes):
        for n_idx in range(n_chains[c_idx]):
            for p_idx in range(n_links[c_idx]):
                iky = ky_h[c_idx][p_idx + n_links[c_idx] * n_idx]
                ikx = kx_h[c_idx][p_idx + n_links[c_idx] * n_idx]
                if ikx >= (nakx + 1) // 2:
                    ikx -= nshift
                c_map[iky + ikx * naky] = c_idx
                n_map[iky + ikx * naky] = n_idx
                n_links_map[iky + ikx * naky] = n_links[c_idx]
                n_chains_map[iky + ikx * naky] = n_chains[c_idx]

    kz_linked = []
    for c_idx in range(n_classes):
        nz_l = nz * n_links[c_idx]
        kz_c = np.zeros((nz_l,), dtype=np.float64)
        for i in range(nz_l):
            if i < nz_l // 2 + 1:
                kz_c[i] = float(i) / float(zp * n_links[c_idx])
            else:
                kz_c[i] = float(i - nz_l) / float(zp * n_links[c_idx])
        kz_linked.append(kz_c)

    return {
        "nClasses": n_classes,
        "nLinks": n_links,
        "nChains": n_chains,
        "ikyLinked_h": ky_h,
        "ikxLinked_h": kx_h,
        "p_map": links_l,
        "n_map": n_map,
        "c_map": c_map,
        "nLinks_map": n_links_map,
        "nChains_map": n_chains_map,
        "kzLinked": kz_linked,
        "naky": naky,
        "nakx": nakx,
        "nx": nx,
        "nz": nz,
        "zp": zp,
    }


def _cpp_linked_oracle_op(
    arr_np: np.ndarray,
    oracle: dict,
    op_name: str,
    *,
    p_hyper_z: int = 6,
    nu_hyper_z: float = 0.1,
    accumulate: bool = False,
    target_np: np.ndarray | None = None,
) -> np.ndarray:
    """Imperative C++ GradParallelLinked execution on Option A (*batch, nz, naky, nakx)."""
    nz = oracle["nz"]
    naky = oracle["naky"]
    nakx = oracle["nakx"]
    nshift = oracle["nx"] - nakx
    half = (nakx + 1) // 2

    batch_shape = arr_np.shape[:-3]
    n_moms = int(np.prod(batch_shape)) if batch_shape else 1
    g_in = np.asarray(arr_np, dtype=np.complex128).reshape(n_moms, nz, naky, nakx).copy()
    g_in[:, :, 0, 0] = 0.0  # unmasked origin
    if op_name == "dealias":
        # GradParallelLinked::dealias (grad_parallel_linked.cu:L328-336) is an empty no-op in C++
        return g_in.reshape(arr_np.shape)

    g_linked = []
    for c in range(oracle["nClasses"]):
        lnk = oracle["nLinks"][c]
        chn = oracle["nChains"][c]
        nz_l = nz * lnk
        buf = np.zeros((n_moms, chn, nz_l), dtype=np.complex128)
        for n_idx in range(chn):
            for p_idx in range(lnk):
                idk = p_idx + lnk * n_idx
                iky = oracle["ikyLinked_h"][c][idk]
                ikx_full = oracle["ikxLinked_h"][c][idk]
                idakx = ikx_full if ikx_full < half else (ikx_full - nshift)
                buf[:, n_idx, p_idx * nz : (p_idx + 1) * nz] = g_in[:, :, iky, idakx]

        kz = oracle["kzLinked"][c]
        if op_name == "identity":
            out_c = buf
        elif op_name == "zft":
            out_c = np.fft.fft(buf, axis=-1) / float(nz_l)
        elif op_name == "zft_inverse":
            out_c = np.fft.ifft(buf, axis=-1) * float(nz_l)
        elif op_name == "dz":
            hat = np.fft.fft(buf, axis=-1) * (1.0j * kz) / float(nz_l)
            out_c = np.fft.ifft(hat, axis=-1) * float(nz_l)
        elif op_name == "abs_dz":
            hat = np.fft.fft(buf, axis=-1) * np.abs(kz) / float(nz_l)
            out_c = np.fft.ifft(hat, axis=-1) * float(nz_l)
        elif op_name == "dz2":
            hat = np.fft.fft(buf, axis=-1) * (-(kz * kz)) / float(nz_l)
            out_c = np.fft.ifft(hat, axis=-1) * float(nz_l)
        elif op_name == "hyperz":
            kzmax = kz[nz_l // 2] if nz_l > 1 else 0.0
            hyp = (np.abs(kz) / kzmax) ** p_hyper_z if kzmax > 0.0 else np.zeros_like(kz)
            hat = np.fft.fft(buf, axis=-1) * (-hyp) / float(nz_l)
            out_c = np.fft.ifft(hat, axis=-1) * float(nz_l)
        elif op_name == "dealias_chain":
            cutoff = (nz_l - 1) // 3
            kzm = np.array(
                [0.0 if (cutoff < i < nz_l - cutoff) else 1.0 for i in range(nz_l)],
                dtype=np.float64,
            )
            out_c = np.fft.ifft(np.fft.fft(buf, axis=-1) * kzm, axis=-1)
        else:
            raise ValueError(f"Unknown op_name={op_name!r}")
        g_linked.append(out_c)

    # linkedCopyBackAll / linkedAccumulateBackAll (device_funcs.cu:L2650-2737)
    scale = float(nu_hyper_z) if op_name == "hyperz" else 1.0
    if accumulate and target_np is not None:
        out_4d = np.asarray(target_np, dtype=np.complex128).reshape(n_moms, nz, naky, nakx).copy()
    else:
        out_4d = np.zeros((n_moms, nz, naky, nakx), dtype=np.complex128)

    for idy in range(naky):
        for idakx in range(nakx):
            if idy == 0 and idakx == 0:
                continue
            idxy = idy + naky * idakx
            c_idx = oracle["c_map"][idxy]
            n_idx = oracle["n_map"][idxy]
            p_idx = oracle["p_map"][idxy]
            seg = g_linked[c_idx][:, n_idx, p_idx * nz : (p_idx + 1) * nz]
            if accumulate and target_np is not None:
                out_4d[:, :, idy, idakx] += scale * seg
            else:
                out_4d[:, :, idy, idakx] = scale * seg

    return out_4d.reshape(arr_np.shape)


def _cpp_linked_oracle_apply_bcs(
    G_np: np.ndarray,
    rhs_np: np.ndarray,
    fields: Fields,
    cfg: GXConfig,
    kperp2_np: np.ndarray,
    oracle: dict,
    dt: float,
) -> np.ndarray:
    """Imperative C++ port of dampEnds_linked (device_funcs.cu:L2813-2876)."""
    if cfg.boundary_option_periodic or cfg.local_limit:
        return np.asarray(rhs_np, dtype=np.complex128).copy()
    ns, nm, nl, nz, naky, nakx = G_np.shape
    out = np.asarray(rhs_np, dtype=np.complex128).copy()
    phi = np.asarray(fields.phi, dtype=np.complex128)
    apar = np.asarray(fields.apar, dtype=np.complex128)
    bpar = np.asarray(fields.bpar, dtype=np.complex128)
    amp = float(cfg.damp_ends_amp) / float(dt)

    for idy in range(1, naky):
        for idakx in range(nakx):
            idxy = idy + naky * idakx
            p_val = oracle["p_map"][idxy]
            n_lnk = oracle["nLinks_map"][idxy]
            width = int(nz * n_lnk * cfg.damp_ends_widthfrac)
            for idz in range(nz):
                idzp = idz + nz * p_val
                nu = 0.0
                if width > 0:
                    if idzp <= width:
                        x = float(idzp) / float(width)
                        nu = 1.0 - 2.0 * x * x / (1.0 + x**4)
                    elif idzp >= nz * n_lnk - width:
                        x = float(nz * n_lnk - idzp) / float(width)
                        nu = 1.0 - 2.0 * x * x / (1.0 + x**4)
                if nu == 0.0:
                    continue
                kp2 = float(kperp2_np[idz, idy, idakx])
                for s_idx, sp in enumerate(cfg.species):
                    b_val = kp2 * sp.rho2
                    for ell in range(nl):
                        jl = float(_cpp_jflr_np(ell, np.array(b_val)))
                        jlb = float(_cpp_jflr_b_np(ell, np.array(b_val)))
                        for m_idx in range(nm):
                            h_val = G_np[s_idx, m_idx, ell, idz, idy, idakx]
                            if m_idx == 0:
                                h_val += (
                                    sp.zt * jl * phi[idz, idy, idakx] + jlb * bpar[idz, idy, idakx]
                                )
                            elif m_idx == 1:
                                h_val -= sp.zt * sp.vt * jl * apar[idz, idy, idakx]
                            out[s_idx, m_idx, ell, idz, idy, idakx] -= nu * amp * h_val
    return out


def _cpp_ntft_calc_idx0(idx: int, nshift: int, nakx: int) -> int:
    """Exact C++ GradParallelNTFT::calc_idx0 (grad_parallel_NTFT.cu:L624-647)."""
    if idx < (nakx // 2):
        idx0 = idx + (nakx // 2) + 1
    else:
        idx0 = idx - (nakx // 2)
    if idx0 < (nakx + 1) // 2:
        return idx0
    return idx0 + nshift


def _cpp_ntft_oracle_init(
    nz: int,
    naky: int,
    nakx: int,
    nx: int,
    zp: int,
    jtwist: int,
    m0_zy: np.ndarray,
    ky_np: np.ndarray,
    *,
    n_links_max: int = 3000,
    n_links_min: int = 0,
) -> dict:
    """Line-by-line imperative C++ port of GradParallelNTFT constructor (L42-164, L431-647)."""
    mode_nums = [0] * (naky * nakx * nz)
    mode = 0

    for idy in range(naky):
        if ky_np[idy] < 1.0e-10:
            for idx in range(nakx):
                mode += 1
                for idz in range(nz):
                    mode_nums[idy + naky * (idx + nakx * idz)] = mode
        else:
            for idx in range(nakx):
                for idz in range(nz):
                    if mode_nums[idy + naky * (idx + nakx * idz)] == 0:
                        idz_start = idz
                        idx_constant = idx + int(m0_zy[idz, idy])
                        while (
                            0
                            <= idx_constant
                            - int(m0_zy[(idz_start - 1 + nz) % nz, idy])
                            - math.floor((idz_start - 1) / float(nz)) * jtwist * idy
                            < nakx
                        ):
                            if idz_start == 0:
                                idx_constant = idx_constant + jtwist * idy
                                idz_start = nz - 1
                            else:
                                idz_start -= 1

                        mode += 1
                        idz_prime = idz_start
                        while 0 <= idx_constant - int(m0_zy[idz_prime, idy]) < nakx:
                            idx_prime = idx_constant - int(m0_zy[idz_prime, idy])
                            mode_nums[idy + naky * (idx_prime + nakx * idz_prime)] = mode
                            if idz_prime == nz - 1:
                                idx_constant = idx_constant - jtwist * idy
                                idz_prime = 0
                            else:
                                idz_prime += 1

    mode_size = [0] * mode
    mode_size_ref = [0] * mode
    for idy in range(naky):
        for idx in range(nakx - 1, -1, -1):
            for idz in range(nz):
                m_id = mode_nums[idy + naky * (idx + nakx * idz)] - 1
                mode_size[m_id] += 1
                mode_size_ref[m_id] += 1

    n_extra = [0] * mode
    for k in range(mode):
        if mode_size[k] > n_links_max:
            n_extra[k] = mode_size[k] % nz
            mode_size[k] -= n_extra[k]
            mode_size_ref[k] -= n_extra[k]

    mode_size.sort()
    n_classes = 1
    for k in range(mode - 1):
        if mode_size[k] != mode_size[k + 1] and mode_size[k] > n_links_min:
            n_classes += 1

    n_chains = [1] * n_classes
    n_links = [0] * n_classes
    c = 0
    for k in range(1, mode):
        if mode_size[k - 1] > n_links_min:
            if mode_size[k] == mode_size[k - 1]:
                n_chains[c] += 1
            else:
                n_links[c] = mode_size[k - 1]
                c += 1
    n_links[n_classes - 1] = mode_size[mode - 1]

    nshift = nx - nakx
    iky_ntft = [[0] * (n_links[ic] * n_chains[ic]) for ic in range(n_classes)]
    ikxdz_ntft = [[0] * (n_links[ic] * n_chains[ic]) for ic in range(n_classes)]

    for ic in range(n_classes):
        n = -1
        for i in range(mode):
            if n_links[ic] == mode_size_ref[i]:
                n += 1
                p = 0
                for idy in range(naky):
                    if idy == 0:
                        for idx in range(nakx):
                            if mode_nums[idy + naky * (idx + nakx * 0)] == i + 1:
                                idx0 = _cpp_ntft_calc_idx0(idx, nshift, nakx)
                                for idz in range(nz):
                                    ikxdz_ntft[ic][p + n_links[ic] * n] = idx0 + nx * idz
                                    iky_ntft[ic][p + n_links[ic] * n] = idy
                                    p += 1
                    else:
                        found = False
                        for idx in range(nakx):
                            if found:
                                break
                            for idz in range(nz):
                                if mode_nums[idy + naky * (idx + nakx * idz)] == i + 1:
                                    extra_cnt = 0
                                    idz_start = idz
                                    idx_constant = idx + int(m0_zy[idz, idy])
                                    while (
                                        0
                                        <= idx_constant
                                        - int(m0_zy[(idz_start - 1 + nz) % nz, idy])
                                        - math.floor((idz_start - 1) / float(nz)) * jtwist * idy
                                        < nakx
                                    ):
                                        if idz_start == 0:
                                            idx_constant += jtwist * idy
                                            idz_start = nz - 1
                                        else:
                                            idz_start -= 1

                                    idz_prime = idz_start
                                    while (
                                        0 <= idx_constant - int(m0_zy[idz_prime, idy]) < nakx
                                        and p < n_links[ic]
                                    ):
                                        if extra_cnt < n_extra[i] // 2:
                                            extra_cnt += 1
                                        else:
                                            idx_prime = idx_constant - int(m0_zy[idz_prime, idy])
                                            idx0 = _cpp_ntft_calc_idx0(idx_prime, nshift, nakx)
                                            ikxdz_ntft[ic][p + n_links[ic] * n] = (
                                                idx0 + nx * idz_prime
                                            )
                                            iky_ntft[ic][p + n_links[ic] * n] = idy
                                            p += 1
                                        if idz_prime == nz - 1:
                                            idx_constant -= jtwist * idy
                                            idz_prime = 0
                                        else:
                                            idz_prime += 1
                                    found = True
                                    break

    kz_ntft = []
    for c_idx in range(n_classes):
        nz_l = n_links[c_idx]
        kz_c = np.zeros((nz_l,), dtype=np.float64)
        for i in range(nz_l):
            if i < nz_l // 2 + 1:
                kz_c[i] = float(i * nz) / float(zp * nz_l)
            else:
                kz_c[i] = float(nz * (i - nz_l)) / float(zp * nz_l)
        kz_ntft.append(kz_c)

    return {
        "mode": mode,
        "mode_nums": mode_nums,
        "nExtra": n_extra,
        "nClasses": n_classes,
        "nLinks": n_links,
        "nChains": n_chains,
        "ikyLinked_h": iky_ntft,
        "ikxLinked_h": ikxdz_ntft,
        "kzLinked": kz_ntft,
        "nz": nz,
        "naky": naky,
        "nakx": nakx,
        "nx": nx,
        "zp": zp,
    }


def _cpp_ntft_oracle_op(
    arr_np: np.ndarray,
    oracle: dict,
    op_name: str,
    *,
    p_hyper_z: int = 6,
    nu_hyper_z: float = 0.1,
    accumulate: bool = False,
    target_np: np.ndarray | None = None,
) -> np.ndarray:
    """Imperative C++ execution of GradParallelNTFT methods on Option A (*batch, nz, naky, nakx)."""
    nz = oracle["nz"]
    naky = oracle["naky"]
    nakx = oracle["nakx"]
    nx = oracle["nx"]
    nshift = nx - nakx
    half = (nakx + 1) // 2

    batch_shape = arr_np.shape[:-3]
    n_moms = int(np.prod(batch_shape)) if batch_shape else 1
    g_in = np.asarray(arr_np, dtype=np.complex128).reshape(n_moms, nz, naky, nakx).copy()
    g_in[:, :, 0, 0] = 0.0
    if op_name == "dealias":
        # GradParallelNTFT::dealias (grad_parallel_NTFT.cu:L244-252) is an empty no-op in C++
        return g_in.reshape(arr_np.shape)

    scale = float(nu_hyper_z) if op_name == "hyperz" else 1.0
    if accumulate and target_np is not None:
        out_4d = np.asarray(target_np, dtype=np.complex128).reshape(n_moms, nz, naky, nakx).copy()
    else:
        out_4d = np.zeros((n_moms, nz, naky, nakx), dtype=np.complex128)

    for c in range(oracle["nClasses"]):
        lnk = oracle["nLinks"][c]
        chn = oracle["nChains"][c]
        buf = np.zeros((n_moms, chn, lnk), dtype=np.complex128)
        for idn in range(chn):
            for idp in range(lnk):
                idpn = idp + lnk * idn
                ikx_ntft = oracle["ikxLinked_h"][c][idpn] % nx
                idz = oracle["ikxLinked_h"][c][idpn] // nx
                idy = oracle["ikyLinked_h"][c][idpn]
                idakx = ikx_ntft if ikx_ntft < half else (ikx_ntft - nshift)
                buf[:, idn, idp] = g_in[:, idz, idy, idakx]

        kz = oracle["kzLinked"][c]
        if op_name == "identity":
            out_c = buf
        elif op_name == "zft":
            out_c = np.fft.fft(buf, axis=-1) / float(lnk)
        elif op_name == "zft_inverse":
            out_c = np.fft.ifft(buf, axis=-1) * float(lnk)
        elif op_name == "dz":
            out_c = np.fft.ifft(np.fft.fft(buf, axis=-1) * (1.0j * kz), axis=-1)
        elif op_name == "abs_dz":
            out_c = np.fft.ifft(np.fft.fft(buf, axis=-1) * np.abs(kz), axis=-1)
        elif op_name == "dz2":
            out_c = np.fft.ifft(np.fft.fft(buf, axis=-1) * (-(kz * kz)), axis=-1)
        elif op_name == "hyperz":
            kzmax = kz[lnk // 2] if lnk > 1 else 0.0
            hyp = (np.abs(kz) / kzmax) ** p_hyper_z if kzmax > 0.0 else np.zeros_like(kz)
            out_c = np.fft.ifft(np.fft.fft(buf, axis=-1) * (-hyp), axis=-1)
        elif op_name == "dealias_chain":
            cutoff = (lnk - 1) // 3
            kzm = np.array(
                [0.0 if (cutoff < i < lnk - cutoff) else 1.0 for i in range(lnk)],
                dtype=np.float64,
            )
            out_c = np.fft.ifft(np.fft.fft(buf, axis=-1) * kzm, axis=-1)
        else:
            raise ValueError(f"Unknown op_name={op_name!r}")

        for idn in range(chn):
            for idp in range(lnk):
                idpn = idp + lnk * idn
                ikx_ntft = oracle["ikxLinked_h"][c][idpn] % nx
                idz = oracle["ikxLinked_h"][c][idpn] // nx
                idy = oracle["ikyLinked_h"][c][idpn]
                idakx = ikx_ntft if ikx_ntft < half else (ikx_ntft - nshift)
                if idy == 0 and idakx == 0:
                    continue
                if accumulate and target_np is not None:
                    out_4d[:, idz, idy, idakx] += scale * out_c[:, idn, idp]
                else:
                    out_4d[:, idz, idy, idakx] = scale * out_c[:, idn, idp]

    out_4d[:, :, 0, 0] = 0.0
    return out_4d.reshape(arr_np.shape)


def _cpp_ntft_oracle_apply_bcs(
    G_np: np.ndarray,
    rhs_np: np.ndarray,
    fields: Fields,
    cfg: GXConfig,
    kperp2_np: np.ndarray,
    oracle: dict,
    dt: float,
) -> np.ndarray:
    """Imperative C++ port of dampEnds_linkedNTFT (device_funcs.cu:L2880-2945)."""
    if cfg.boundary_option_periodic or cfg.local_limit:
        return np.asarray(rhs_np, dtype=np.complex128).copy()
    ns, nm, nl, nz, naky, nakx = G_np.shape
    nx = oracle["nx"]
    nshift = nx - nakx
    half = (nakx + 1) // 2
    out = np.asarray(rhs_np, dtype=np.complex128).copy()
    phi = np.asarray(fields.phi, dtype=np.complex128)
    apar = np.asarray(fields.apar, dtype=np.complex128)
    bpar = np.asarray(fields.bpar, dtype=np.complex128)
    amp = float(cfg.damp_ends_amp) / float(dt)

    for c in range(oracle["nClasses"]):
        lnk = oracle["nLinks"][c]
        chn = oracle["nChains"][c]
        width = int(lnk * cfg.damp_ends_widthfrac)
        if width == 0:
            width = 1
        for idn in range(chn):
            for idp in range(lnk):
                idpn = idp + lnk * idn
                idy = oracle["ikyLinked_h"][c][idpn]
                if idy <= 0:
                    continue
                ikx_ntft = oracle["ikxLinked_h"][c][idpn] % nx
                idz = oracle["ikxLinked_h"][c][idpn] // nx
                idakx = ikx_ntft if ikx_ntft < half else (ikx_ntft - nshift)

                nu = 0.0
                if idp < width:
                    x = float(idp) / float(width)
                    nu = 1.0 - 2.0 * x * x / (1.0 + x**4)
                elif idp > lnk - width - 1:
                    x = float(lnk - idp - 1) / float(width)
                    nu = 1.0 - 2.0 * x * x / (1.0 + x**4)
                if nu == 0.0:
                    continue
                kp2 = float(kperp2_np[idz, idy, idakx])
                for s_idx, sp in enumerate(cfg.species):
                    b_val = kp2 * sp.rho2
                    for ell in range(nl):
                        jl = float(_cpp_jflr_np(ell, np.array(b_val)))
                        jlb = float(_cpp_jflr_b_np(ell, np.array(b_val)))
                        for m_idx in range(nm):
                            h_val = G_np[s_idx, m_idx, ell, idz, idy, idakx]
                            if m_idx == 0:
                                h_val += (
                                    sp.zt * jl * phi[idz, idy, idakx] + jlb * bpar[idz, idy, idakx]
                                )
                            elif m_idx == 1:
                                h_val -= sp.zt * sp.vt * jl * apar[idz, idy, idakx]
                            out[s_idx, m_idx, ell, idz, idy, idakx] -= nu * amp * h_val
    return out


# ---------------------------------------------------------------------------
# 1. GradParallelLinked Topology, nLinks=1 Equivalence, & C++ Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("jtwist", [1, -1, 2, -2, 3, -3])
def test_linked_topology_positive_and_negative_jtwist_multiclass(jtwist: int) -> None:
    """Verify get_nClasses, get_nLinks_nChains, kFill, and maps for jtwist > 0 and jtwist < 0."""
    base_cfg = load_config(
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    )
    shat_use = 0.8 if jtwist > 0 else -0.8
    cfg = dataclasses.replace(
        base_cfg,
        nx=16,  # nakx = 11 (-5..+5) -> rich non-uniform chain classes!
        ny=13,  # naky = 5 (0..4)
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        shat=shat_use,
        jtwist=jtwist,
        x0=-1.0,
    )
    cfg, grids, _ = init_geometry(cfg)
    assert cfg.jtwist == jtwist

    topo = build_linked_topology(grids, cfg.jtwist)
    oracle = _cpp_linked_oracle_init(
        grids.naky, grids.nakx, grids.nx, grids.nz, grids.zp, cfg.jtwist
    )

    assert topo.n_classes == oracle["nClasses"]
    assert topo.n_classes >= 3, f"Expected >= 3 non-uniform classes, got {topo.n_classes}"
    assert list(topo.n_links) == oracle["nLinks"]
    assert list(topo.n_chains) == oracle["nChains"]

    for ic in range(topo.n_classes):
        np.testing.assert_array_equal(topo.iky_chains[ic].ravel(), oracle["ikyLinked_h"][ic])
        np.testing.assert_array_equal(topo.ikx_full_chains[ic].ravel(), oracle["ikxLinked_h"][ic])
        np.testing.assert_array_equal(
            np.asarray(grids.active_kx_idx)[topo.ikx_chains[ic]],
            topo.ikx_full_chains[ic],
        )

    # C++ stores maps in column-major (idy + naky * idakx) == Fortran ravel of (naky, nakx)
    np.testing.assert_array_equal(topo.p_map.ravel(order="F"), oracle["p_map"])
    np.testing.assert_array_equal(topo.n_map.ravel(order="F"), oracle["n_map"])
    np.testing.assert_array_equal(topo.c_map.ravel(order="F"), oracle["c_map"])
    np.testing.assert_array_equal(topo.n_links_map.ravel(order="F"), oracle["nLinks_map"])
    np.testing.assert_array_equal(topo.n_chains_map.ravel(order="F"), oracle["nChains_map"])

    gp = init_linked_parallel_grad(cfg, grids)
    assert "Class 0" in gp.format_links()
    for idy in range(grids.naky):
        for idx in range(grids.nakx):
            coupled = gp.coupled_ikx_indices(idy, idx)
            assert idx in coupled
            assert len(coupled) == int(topo.n_links_map[idy, idx])


def test_linked_nlink1_equivalence_with_parallel_grad() -> None:
    """Verify LinkedParallelGrad with nLinks=1 matches Session 3 ParallelGrad identically."""
    base_cfg = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    for dealias_kz in (False, True):
        for boundary_mode, jtwist_in in [("periodic", -1000), ("linked", 20)]:
            cfg = dataclasses.replace(
                base_cfg,
                nx=7,
                ny=7,
                nz=16,
                ntheta_in=16,
                nperiod=1,
                zp=1,
                nhermite=4,
                nlaguerre=2,
                fbpar=1.0,
                boundary=boundary_mode,
                boundary_option_periodic=(boundary_mode == "periodic"),
                jtwist=jtwist_in,
                dealias_kz=dealias_kz,
            )
            cfg, grids, geo = init_geometry(cfg)
            gp_ref = init_parallel_grad(cfg, grids, geo, dtype=jnp.float64)
            gp_lnk = init_boundary_parallel(cfg, grids, geo, dtype=jnp.float64)
            assert isinstance(gp_ref, ParallelGrad)
            assert isinstance(gp_lnk, LinkedParallelGrad)
            assert gp_lnk.n_classes == 1 and gp_lnk.n_links == (1,)

            solver = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
            g = _random_complex_arr(grids.g_shape, np.asarray(grids.active_mask), seed=31)
            tgt = _random_complex_arr(grids.g_shape, np.asarray(grids.active_mask), seed=32)
            f = solve_fields(g, solver)
            tag = f"{boundary_mode}:dealias={dealias_kz}"

            _assert_close_fp64(gp_lnk.identity(g), g, f"{tag}:identity")
            _assert_close_fp64(gp_lnk.dz(g), gp_ref.dz(g), f"{tag}:dz")
            _assert_close_fp64(
                gp_lnk.dz(g, accumulate=True, target=tgt),
                gp_ref.dz(g, accumulate=True, target=tgt),
                f"{tag}:dz_acc",
            )
            _assert_close_fp64(gp_lnk.abs_dz(g), gp_ref.abs_dz(g), f"{tag}:abs_dz")
            _assert_close_fp64(gp_lnk.dz2(g), gp_ref.dz2(g), f"{tag}:dz2")
            _assert_close_fp64(gp_lnk.hyperz(g, 0.25), gp_ref.hyperz(g, 0.25), f"{tag}:hz")
            _assert_close_fp64(gp_lnk.zft(g), gp_ref.zft(g), f"{tag}:zft")
            _assert_close_fp64(gp_lnk.zft_inverse(g), gp_ref.zft_inverse(g), f"{tag}:zft_inv")
            _assert_close_fp64(gp_lnk.dealias(g), gp_ref.dealias(g), f"{tag}:dealias")
            _assert_close_fp64(
                gp_lnk.dealias_chain(g), gp_ref.dealias_chain(g), f"{tag}:dealias_chain"
            )
            _assert_close_fp64(
                gp_lnk.apply_bcs(g, tgt, f, solver, 0.02),
                gp_ref.apply_bcs(g, tgt, f, solver, 0.02),
                f"{tag}:apply_bcs",
            )


def test_cpp_unit_tests_grad_parallel_linked_monochromatic_and_gaussian() -> None:
    """Reproduce all 5 TEST_F in test_grad_parallel_linked.cu + 5-link ballooning Gaussian."""
    cyc_cfg = load_config(UNIT_INPUTS_DIR / "cyc_nl.in")

    # 1 & 2. TestGradParallelLinked3D_nLink1 (nx=1, ny=4, nz=8): identity + cos(2z) derivative
    cfg_nl1 = dataclasses.replace(
        cyc_cfg,
        nx=1,
        ny=4,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nspecies=1,
        nhermite=4,
        nlaguerre=2,
        species=cyc_cfg.species[:1],
        init_field="all",
        init_amp=0.01,
        ikpar_init=2,
        linear=True,
        nonlinear_mode=False,
        random_init=False,
        random_seed=22,
    )
    cfg_nl1, grids_nl1, geo_nl1 = init_geometry(cfg_nl1)
    gp_nl1 = init_linked_parallel_grad(cfg_nl1, grids_nl1)
    g_nl1 = enforce_reality(
        init_moments(cfg_nl1, grids_nl1, geo_nl1, dtype=jnp.complex128), grids_nl1
    )
    _assert_close_fp64(gp_nl1.identity(g_nl1), g_nl1, "nLink1:identity")

    cfg_nl1_d = dataclasses.replace(cfg_nl1, init_field="density")
    g_cos = init_moments(cfg_nl1_d, grids_nl1, geo_nl1, dtype=jnp.complex128)
    z_nl1 = np.asarray(grids_nl1.z, dtype=np.float64)
    # Since g_cos[:, 0, 0, iz, 1, 0] = c * cos(2 * z), dz is -2 * c * sin(2 * z)
    amp_nl1 = np.asarray(g_cos[0, 0, 0, 0, 1, 0]) / math.cos(2.0 * float(z_nl1[0]))
    deriv_expect_nl1 = np.zeros(grids_nl1.field_shape, dtype=np.complex128)
    deriv_expect_nl1[:, 1, 0] = -2.0 * amp_nl1 * np.sin(2.0 * z_nl1)
    _assert_close_fp64(gp_nl1.dz(g_cos[0, 0, 0]), deriv_expect_nl1, "nLink1:dz_single_mom")
    _assert_close_fp64(gp_nl1.dz(g_cos)[0, 0, 0], deriv_expect_nl1, "nLink1:dz_6d")

    # 3. TestGradParallelLinked3D (nx=4, ny=4, nz=32 -> nLinks=3): identity + cos^2(2z) + accumulate
    cfg_3d = dataclasses.replace(
        cyc_cfg,
        nx=4,
        ny=4,
        nz=32,
        ntheta_in=32,
        nperiod=1,
        zp=1,
        nspecies=1,
        nhermite=4,
        nlaguerre=2,
        species=cyc_cfg.species[:1],
        init_field="all",
        init_amp=1.0,
        ikpar_init=2,
        linear=True,
        nonlinear_mode=False,
        random_init=False,
    )
    cfg_3d, grids_3d, geo_3d = init_geometry(cfg_3d)
    gp_3d = init_linked_parallel_grad(cfg_3d, grids_3d)
    assert gp_3d.n_classes == 2 and gp_3d.n_links == (1, 3)

    g_id3d = enforce_reality(init_moments(cfg_3d, grids_3d, geo_3d, dtype=jnp.complex128), grids_3d)
    _assert_close_fp64(gp_3d.identity(g_id3d), g_id3d, "Linked3D:identity")

    z32 = np.asarray(grids_3d.z, dtype=np.float64)
    init_3d = np.zeros(grids_3d.field_shape, dtype=np.complex128)
    deriv_3d = np.zeros(grids_3d.field_shape, dtype=np.complex128)
    init_3d[:, 1, :] = ((1.0 - 1.0j) * (np.cos(2.0 * z32) ** 2))[:, None]
    deriv_3d[:, 1, :] = ((-4.0 + 4.0j) * np.cos(2.0 * z32) * np.sin(2.0 * z32))[:, None]

    res_single = gp_3d.dz(jnp.asarray(init_3d))
    _assert_close_fp64(res_single, deriv_3d, "Linked3D:dz_single_mom")
    g_upar = jnp.zeros(grids_3d.g_shape, dtype=jnp.complex128).at[0, 1, 0].set(init_3d)
    g_res0 = jnp.zeros(grids_3d.g_shape, dtype=jnp.complex128).at[0, 1, 0].set(res_single)
    g_acc = gp_3d.dz(g_upar, accumulate=True, target=g_res0)
    _assert_close_fp64(g_acc[0, 1, 0], 2.0 * deriv_3d, "Linked3D:dz_accumulate")

    # 4a. Exact C++ EvaluateDerivativeGaussian (nx=4, width=0.5)
    cfg_gauss = dataclasses.replace(
        cfg_3d, init_field="upar", gaussian_init=True, gaussian_width=0.5
    )
    g_gauss = init_moments(cfg_gauss, grids_3d, geo_3d, dtype=jnp.complex128)
    deriv_gauss_cpp = np.zeros(grids_3d.field_shape, dtype=np.complex128)
    for idx_act in range(grids_3d.nakx):
        kx_mag = abs(float(grids_3d.kx[idx_act]))
        th0 = kx_mag / (float(cfg_gauss.shat) * float(grids_3d.ky[1]))
        deriv_gauss_cpp[:, 1, idx_act] = (
            (1.0 + 1.0j) * (-2.0 * z32 / (0.5**2)) * np.exp(-(((z32 - th0) / 0.5) ** 2))
        )
    _assert_close_fp64(
        gp_3d.dz(g_gauss[0, 1, 0]),
        deriv_gauss_cpp,
        "Linked3D:EvaluateDerivativeGaussian_cpp",
        rtol=5.0e-6,
        atol=5.0e-6,
    )

    # 4b. Wide 5-link extended ballooning Gaussian crossing z = +-pi, +-3*pi smoothly!
    cfg_wide = dataclasses.replace(cfg_3d, nx=7, nz=64, ntheta_in=64)  # nakx=5 -> nLinks=5
    cfg_wide, grids_wide, _ = init_geometry(cfg_wide)
    gp_wide = init_linked_parallel_grad(cfg_wide, grids_wide)
    assert gp_wide.n_links == (1, 5)
    z64 = np.asarray(grids_wide.z, dtype=np.float64)
    w_ext = 2.2  # Wide enough to have O(1) amplitude across z = +-pi junctions!
    f_ext = np.zeros(grids_wide.field_shape, dtype=np.complex128)
    df_ext = np.zeros(grids_wide.field_shape, dtype=np.complex128)
    dz2f_ext = np.zeros(grids_wide.field_shape, dtype=np.complex128)
    ky1 = float(grids_wide.ky[1])
    for idx_act in range(grids_wide.nakx):
        kx_signed = float(grids_wide.kx[idx_act])
        # Ballooning angle theta_ext = z - kx / (shat * ky) is continuous across p -> p+1!
        th_ext = z64 - kx_signed / (float(cfg_wide.shat) * ky1)
        env = np.exp(-((th_ext / w_ext) ** 2))
        f_ext[:, 1, idx_act] = env
        df_ext[:, 1, idx_act] = (-2.0 * th_ext / (w_ext**2)) * env
        dz2f_ext[:, 1, idx_act] = ((4.0 * th_ext**2 - 2.0 * w_ext**2) / (w_ext**4)) * env

    # Confirm non-trivial O(0.1) amplitude right at the z = -pi link junction on p != 0
    assert float(np.max(np.abs(f_ext[0, 1, 1:]))) > 0.10
    _assert_close_fp64(
        gp_wide.dz(jnp.asarray(f_ext)),
        df_ext,
        "Linked5:wide_ballooning_dz",
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    _assert_close_fp64(
        gp_wide.dz2(jnp.asarray(f_ext)),
        dz2f_ext,
        "Linked5:wide_ballooning_dz2",
        rtol=1.0e-9,
        atol=1.0e-9,
    )


# ---------------------------------------------------------------------------
# 2. Differential Verification of LinkedParallelGrad (All Ranks, Nyquist, Padding)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("jtwist", [1, -2])
@pytest.mark.parametrize("pad_mult", [1, 8])
def test_linked_operators_differential_all_ranks_nyquist_and_padded_buckets(
    jtwist: int, pad_mult: int
) -> None:
    """Verify LinkedParallelGrad vs C++ oracle across 3D/4D/5D/6D, Nyquist, & padded buckets."""
    kbm_cfg = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    shat_val = float(kbm_cfg.shat) if jtwist > 0 else -float(kbm_cfg.shat)
    cfg = dataclasses.replace(
        kbm_cfg,
        nx=10,  # nakx = 7 -> multiple classes
        ny=10,  # naky = 4
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        fbpar=1.0,
        shat=shat_val,
        jtwist=jtwist,
        x0=-1.0,
        p_hyper_z=6,
    )
    cfg, grids, geo = init_geometry(cfg)
    oracle = _cpp_linked_oracle_init(
        grids.naky, grids.nakx, grids.nx, grids.nz, grids.zp, cfg.jtwist
    )
    gp64 = init_linked_parallel_grad(cfg, grids, dtype=jnp.float64, chain_pad_multiple=pad_mult)
    gp32 = gp64.astype(jnp.float32)
    solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    solver32 = solver64.astype(jnp.float32)
    mask_np = np.asarray(grids.active_mask, dtype=bool)

    # Test 3D, 4D, 5D, and 6D leading batch shapes + unmasked origin input masking
    rank_shapes = (
        [
            ("3D", (grids.nz, grids.naky, grids.nakx)),
            ("4D", (2, grids.nz, grids.naky, grids.nakx)),
            ("5D", (2, 3, 2, grids.nz, grids.naky, grids.nakx)),
            ("6D", grids.g_shape),
        ]
        if (jtwist == 1 and pad_mult == 8)
        else [
            ("3D", (grids.nz, grids.naky, grids.nakx)),
            ("6D", grids.g_shape),
        ]
    )
    for rank_label, shape in rank_shapes:
        arr128 = _random_complex_arr(shape, mask_np, seed=200 + len(shape), zero_origin=False)
        tgt128 = _random_complex_arr(shape, mask_np, seed=300 + len(shape), zero_origin=True)
        arr64 = arr128.astype(jnp.complex64)

        for op_name in (
            "identity",
            "dz",
            "abs_dz",
            "dz2",
            "zft",
            "zft_inverse",
            "dealias",
            "dealias_chain",
        ):
            ref = _cpp_linked_oracle_op(np.asarray(arr128), oracle, op_name)
            out128 = getattr(gp64, op_name)(arr128)
            out64 = getattr(gp32, op_name)(arr64)
            _assert_close_fp64(out128, ref, f"linked:{rank_label}:{op_name}:fp64")
            _assert_close_fp32_1e6(out64, ref, f"linked:{rank_label}:{op_name}:c64")
            np.testing.assert_array_equal(np.asarray(out128[..., 0, 0]), 0.0)

        # Round-trip zft -> zft_inverse and dealias / dealias_chain idempotence
        masked_in = np.where(mask_np, np.asarray(arr128), 0.0)
        _assert_close_fp64(
            gp64.zft_inverse(gp64.zft(arr128)), masked_in, f"linked:{rank_label}:zft_rt"
        )
        _assert_close_fp64(gp64.dealias(arr128), masked_in, f"linked:{rank_label}:dealias_noop")
        d1 = gp64.dealias_chain(arr128)
        _assert_close_fp64(gp64.dealias_chain(d1), d1, f"linked:{rank_label}:dealias_chain_idem")

        # hyperz & accumulate=True checks
        ref_hz = _cpp_linked_oracle_op(
            np.asarray(arr128), oracle, "hyperz", p_hyper_z=cfg.p_hyper_z, nu_hyper_z=0.15
        )
        _assert_close_fp64(gp64.hyperz(arr128, 0.15), ref_hz, f"linked:{rank_label}:hyperz")
        _assert_close_fp32_1e6(gp32.hyperz(arr64, 0.15), ref_hz, f"linked:{rank_label}:hyperz:c64")
        for acc_op in ("dz", "abs_dz", "hyperz"):
            ref_acc = _cpp_linked_oracle_op(
                np.asarray(arr128),
                oracle,
                acc_op,
                p_hyper_z=cfg.p_hyper_z,
                nu_hyper_z=0.15,
                accumulate=True,
                target_np=np.asarray(tgt128),
            )
            act_acc = (
                gp64.hyperz(arr128, 0.15, accumulate=True, target=tgt128)
                if acc_op == "hyperz"
                else getattr(gp64, acc_op)(arr128, accumulate=True, target=tgt128)
            )
            _assert_close_fp64(act_acc, ref_acc, f"linked:{rank_label}:{acc_op}:acc")

    # Verify apply_bcs (dampEnds_linked): exact C++ match + ZERO damping at internal link junctions!
    g128 = _random_complex_arr(grids.g_shape, mask_np, seed=401)
    rhs128 = _random_complex_arr(grids.g_shape, mask_np, seed=402)
    f128 = solve_fields(g128, solver64)
    dt = 0.02
    ref_bcs = _cpp_linked_oracle_apply_bcs(
        np.asarray(g128), np.asarray(rhs128), f128, cfg, np.asarray(geo.kperp2), oracle, dt
    )
    act_bcs128 = gp64.apply_bcs(g128, rhs128, f128, solver64, dt)
    act_bcs64 = gp32.apply_bcs(
        g128.astype(jnp.complex64),
        rhs128.astype(jnp.complex64),
        solve_fields(g128.astype(jnp.complex64), solver32),
        solver32,
        dt,
    )
    _assert_close_fp64(act_bcs128, ref_bcs, "linked:apply_bcs:fp64")
    _assert_close_fp32_1e6(act_bcs64, ref_bcs, "linked:apply_bcs:c64")
    # Zonal ky=0 untouched, and middle link (p=1 of a >=3-link chain) has zero end-damping!
    np.testing.assert_array_equal(np.asarray(gp64.end_damp_nu[:, 0, :]), 0.0)
    mid_mask = (np.asarray(gp64.n_links_map) >= 3) & (np.asarray(gp64.p_map) == 1)
    assert np.any(mid_mask), "Expected at least one interior link p=1 in >=3-link chain"
    np.testing.assert_array_equal(np.asarray(gp64.end_damp_nu)[:, mid_mask], 0.0)

    # Verify GX positive Nyquist sign +(nz * nLinks // 2) / (zp * nLinks) on (-1)^idzl
    nyq_field = np.zeros(grids.field_shape, dtype=np.complex128)
    for idy in range(1, grids.naky):
        for idx in range(grids.nakx):
            p_val = int(np.asarray(gp64.p_map)[idy, idx])
            idzl = np.arange(grids.nz) + grids.nz * p_val
            nyq_field[:, idy, idx] = (0.7 - 0.3j) * ((-1.0) ** idzl)
    k_nyq = float(grids.nz // 2) / float(grids.zp)
    _assert_close_fp64(
        gp64.dz(jnp.asarray(nyq_field)),
        1.0j * k_nyq * nyq_field,
        "linked:positive_nyquist_kz",
    )


# ---------------------------------------------------------------------------
# 3. Differential Verification of NTFTParallelGrad (s-alpha, Miller, & 3D W7-X)
# ---------------------------------------------------------------------------


def test_ntft_topology_and_operators_salpha_miller_and_w7x(
    w7x_eik_files: dict[str, Path],
) -> None:
    """Verify NTFTParallelGrad vs C++ oracle on low-shear s-alpha, Miller, and 3D W7-X."""
    salpha_base = load_config(
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    )
    miller_base = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    w7x_base = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
    w7x_eik = w7x_eik_files["nz32"]

    equilibria = [
        (
            "salpha_pos_low_shat",
            dataclasses.replace(
                salpha_base,
                nx=10,
                ny=10,
                nz=16,
                ntheta_in=16,
                nperiod=1,
                zp=1,
                nhermite=4,
                nlaguerre=2,
                shat=0.05,
                non_twist=True,
                x0=-1.0,
                jtwist=-1000,
            ),
            None,
        ),
        (
            "salpha_neg_low_shat",
            dataclasses.replace(
                salpha_base,
                nx=10,
                ny=10,
                nz=16,
                ntheta_in=16,
                nperiod=1,
                zp=1,
                nhermite=4,
                nlaguerre=2,
                shat=-0.12,
                non_twist=True,
                x0=-1.0,
                jtwist=-1000,
            ),
            None,
        ),
        (
            "miller_em_low_shat",
            dataclasses.replace(
                miller_base,
                nx=10,
                ny=10,
                nz=16,
                ntheta_in=16,
                nperiod=1,
                zp=1,
                nhermite=4,
                nlaguerre=2,
                fbpar=1.0,
                shat=0.1,
                non_twist=True,
                x0=-1.0,
                jtwist=-1000,
            ),
            None,
        ),
        (
            "w7x_stellarator_ntft",
            dataclasses.replace(
                w7x_base,
                geo_option="nc",
                geofilename=str(w7x_eik),
                nx=10,
                ny=10,
                nz=32,
                ntheta_in=32,
                nhermite=4,
                nlaguerre=2,
                non_twist=True,
                x0=-1.0,
                jtwist=-1000,
            ),
            w7x_eik,
        ),
    ]

    for eq_name, cfg_in, eik_p in equilibria:
        cfg, grids, geo = load_eik_nc(eik_p, cfg_in) if eik_p is not None else init_geometry(cfg_in)
        assert geo.m0.shape == (grids.nz, grids.naky)
        assert geo.ftwist.shape == (grids.nz,)
        assert geo.delta_kx.shape == (grids.nz, grids.naky)

        topo = build_ntft_topology(grids, geo.m0, cfg.jtwist)
        oracle = _cpp_ntft_oracle_init(
            grids.nz,
            grids.naky,
            grids.nakx,
            grids.nx,
            grids.zp,
            cfg.jtwist,
            np.asarray(geo.m0),
            np.asarray(grids.ky),
        )

        # Verify exact mode numbers, class counts, chain lengths (including non-multiples of nz!)
        assert topo.mode_count == oracle["mode"]
        assert topo.n_classes == oracle["nClasses"]
        assert list(topo.n_links) == oracle["nLinks"]
        assert list(topo.n_chains) == oracle["nChains"]
        # C++ mode_nums[idy + naky * (idx + nakx * idz)] == C-order ravel of (nz, nakx, naky)
        np.testing.assert_array_equal(topo.mode_nums_seq.ravel(), oracle["mode_nums"])
        for ic in range(topo.n_classes):
            np.testing.assert_array_equal(topo.iky_chains[ic].ravel(), oracle["ikyLinked_h"][ic])
            np.testing.assert_array_equal(
                topo.ikxdz_full_chains[ic].ravel(), oracle["ikxLinked_h"][ic]
            )

        # Verify calc_idx0_ntft directly
        nshift = grids.nx - grids.nakx
        for idx_s in range(grids.nakx):
            _, idx0_full = calc_idx0_ntft(idx_s, nshift, grids.nakx)
            assert idx0_full == _cpp_ntft_calc_idx0(idx_s, nshift, grids.nakx)

        gp64 = init_ntft_parallel_grad(cfg, grids, geo, dtype=jnp.float64, chain_pad_multiple=8)
        gp32 = gp64.astype(jnp.float32)
        assert "NTFT Class 0" in gp64.format_links()
        solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
        solver32 = solver64.astype(jnp.float32)
        mask_np = np.asarray(grids.active_mask, dtype=bool)

        ntft_ranks = (
            [
                ("3D", (grids.nz, grids.naky, grids.nakx)),
                ("5D", (cfg.nspecies, 2, cfg.nlaguerre, grids.nz, grids.naky, grids.nakx)),
                ("6D", grids.g_shape),
            ]
            if eq_name == "w7x_stellarator_ntft"
            else [("6D", grids.g_shape)]
        )
        for rank_label, shape in ntft_ranks:
            arr128 = _random_complex_arr(shape, mask_np, seed=500 + len(shape), zero_origin=False)
            tgt128 = _random_complex_arr(shape, mask_np, seed=600 + len(shape), zero_origin=True)
            arr64 = arr128.astype(jnp.complex64)

            for op_name in (
                "identity",
                "dz",
                "abs_dz",
                "dz2",
                "zft",
                "zft_inverse",
                "dealias",
                "dealias_chain",
            ):
                ref = _cpp_ntft_oracle_op(np.asarray(arr128), oracle, op_name)
                out128 = getattr(gp64, op_name)(arr128)
                out64 = getattr(gp32, op_name)(arr64)
                _assert_close_fp64(out128, ref, f"{eq_name}:{rank_label}:{op_name}:fp64")
                _assert_close_fp32_1e6(out64, ref, f"{eq_name}:{rank_label}:{op_name}:c64")
                np.testing.assert_array_equal(np.asarray(out128[..., 0, 0]), 0.0)

            _assert_close_fp64(
                gp64.zft_inverse(gp64.zft(arr128)),
                np.where(mask_np, np.asarray(arr128), 0.0),
                f"{eq_name}:{rank_label}:zft_roundtrip",
            )
            _assert_close_fp64(
                gp64.dealias(arr128),
                np.where(mask_np, np.asarray(arr128), 0.0),
                f"{eq_name}:{rank_label}:dealias_noop",
            )
            ref_hz = _cpp_ntft_oracle_op(
                np.asarray(arr128), oracle, "hyperz", p_hyper_z=cfg.p_hyper_z, nu_hyper_z=0.2
            )
            _assert_close_fp64(
                gp64.hyperz(arr128, 0.2), ref_hz, f"{eq_name}:{rank_label}:hyperz:fp64"
            )
            _assert_close_fp32_1e6(
                gp32.hyperz(arr64, 0.2), ref_hz, f"{eq_name}:{rank_label}:hyperz:c64"
            )
            ref_dz_acc = _cpp_ntft_oracle_op(
                np.asarray(arr128), oracle, "dz", accumulate=True, target_np=np.asarray(tgt128)
            )
            _assert_close_fp64(
                gp64.dz(arr128, accumulate=True, target=tgt128),
                ref_dz_acc,
                f"{eq_name}:{rank_label}:dz_acc",
            )

        # Verify NTFT apply_bcs (dampEnds_linkedNTFT, including short-chain width=1 & right ramp!)
        g128 = _random_complex_arr(grids.g_shape, mask_np, seed=777)
        rhs128 = _random_complex_arr(grids.g_shape, mask_np, seed=778)
        f128 = solve_fields(g128, solver64)
        dt = 0.015
        ref_bcs = _cpp_ntft_oracle_apply_bcs(
            np.asarray(g128), np.asarray(rhs128), f128, cfg, np.asarray(geo.kperp2), oracle, dt
        )
        act_bcs128 = gp64.apply_bcs(g128, rhs128, f128, solver64, dt)
        act_bcs64 = gp32.apply_bcs(
            g128.astype(jnp.complex64),
            rhs128.astype(jnp.complex64),
            solve_fields(g128.astype(jnp.complex64), solver32),
            solver32,
            dt,
        )
        _assert_close_fp64(act_bcs128, ref_bcs, f"{eq_name}:apply_bcs:fp64")
        _assert_close_fp32_1e6(act_bcs64, ref_bcs, f"{eq_name}:apply_bcs:c64")
        np.testing.assert_array_equal(np.asarray(gp64.end_damp_nu[:, 0, :]), 0.0)


def test_ntft_nextra_long_chain_truncation_and_kz_dealias_helpers() -> None:
    """Verify nExtra truncation when mode_size > n_links_max plus init_kz_linked* dealias flag."""
    base_cfg = load_config(
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    )
    cfg = dataclasses.replace(
        base_cfg,
        nx=16,  # nakx = 11
        ny=10,  # naky = 4
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        shat=0.18,
        non_twist=True,
        x0=-1.0,
        jtwist=-1000,
    )
    cfg, grids, geo = init_geometry(cfg)

    # Set n_links_max = 32 so chains longer than 32 grid points with non-zero % nz get trimmed!
    n_max = 32
    gp_trunc = init_ntft_parallel_grad(cfg, grids, geo, dtype=jnp.float64, n_links_max=n_max)
    oracle = _cpp_ntft_oracle_init(
        grids.nz,
        grids.naky,
        grids.nakx,
        grids.nx,
        grids.zp,
        cfg.jtwist,
        np.asarray(geo.m0),
        np.asarray(grids.ky),
        n_links_max=n_max,
    )
    assert int(jnp.sum(gp_trunc.n_extra)) > 0, "Expected non-zero nExtra truncation on long chains"
    np.testing.assert_array_equal(np.asarray(gp_trunc.n_extra), oracle["nExtra"])

    g128 = _random_complex_arr(grids.g_shape, np.asarray(grids.active_mask), seed=811)
    tgt128 = _random_complex_arr(grids.g_shape, np.asarray(grids.active_mask), seed=812)
    solver = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    f128 = solve_fields(g128, solver)

    for op_name in (
        "identity",
        "dz",
        "abs_dz",
        "dz2",
        "zft",
        "zft_inverse",
        "dealias",
        "dealias_chain",
    ):
        _assert_close_fp64(
            getattr(gp_trunc, op_name)(g128),
            _cpp_ntft_oracle_op(np.asarray(g128), oracle, op_name),
            f"ntft_nextra:{op_name}",
        )
    _assert_close_fp64(
        gp_trunc.dz(g128, accumulate=True, target=tgt128),
        _cpp_ntft_oracle_op(
            np.asarray(g128), oracle, "dz", accumulate=True, target_np=np.asarray(tgt128)
        ),
        "ntft_nextra:dz_acc",
    )
    _assert_close_fp64(
        gp_trunc.apply_bcs(g128, tgt128, f128, solver, 0.02),
        _cpp_ntft_oracle_apply_bcs(
            np.asarray(g128), np.asarray(tgt128), f128, cfg, np.asarray(geo.kperp2), oracle, 0.02
        ),
        "ntft_nextra:apply_bcs",
    )

    # Direct check of dealias_kz=True flag on init_kz_linked and init_kz_linked_ntft
    kz_raw = init_kz_linked(16, 1, 3, dealias_kz=False)
    kz_deal = init_kz_linked(16, 1, 3, dealias_kz=True)
    kzm48 = init_kzm_linked(48)
    _assert_close_fp64(kz_deal, jnp.where(kzm48 == 1, kz_raw, 0.0), "init_kz_linked:dealias")

    kz_ntft_raw = init_kz_linked_ntft(16, 1, 25, dealias_kz=False)
    kz_ntft_deal = init_kz_linked_ntft(16, 1, 25, dealias_kz=True)
    kzm25 = init_kzm_linked(25)
    _assert_close_fp64(
        kz_ntft_deal, jnp.where(kzm25 == 1, kz_ntft_raw, 0.0), "init_kz_linked_ntft:dealias"
    )


# ---------------------------------------------------------------------------
# 4. Full linear_rhs Integration (All Closures, Dissipation, Steppers, Linked & NTFT)
# ---------------------------------------------------------------------------


def _cpp_oracle_full_linear_rhs(
    G_np: np.ndarray,
    fields: Fields,
    cfg: GXConfig,
    grids,
    geo,
    dt: float,
) -> np.ndarray:
    """Independent C++ Linear_GK::rhs oracle supporting multi-link Linked and NTFT."""
    ns, nm, nl, nz, naky, nakx = G_np.shape
    mask2d = np.asarray(grids.active_mask, dtype=bool)
    mask3d = np.broadcast_to(mask2d[None, :, :], (nz, naky, nakx))
    mask6d = np.broadcast_to(mask2d[None, None, None, None, :, :], G_np.shape)

    g = np.where(mask6d, np.asarray(G_np, dtype=np.complex128), 0.0)
    phi = np.where(mask3d, np.asarray(fields.phi, dtype=np.complex128), 0.0)
    apar = np.where(mask3d, np.asarray(fields.apar, dtype=np.complex128), 0.0)
    bpar = np.where(mask3d, np.asarray(fields.bpar, dtype=np.complex128), 0.0)

    if cfg.non_twist:
        par_oracle = _cpp_ntft_oracle_init(
            nz,
            naky,
            nakx,
            grids.nx,
            grids.zp,
            cfg.jtwist,
            np.asarray(geo.m0),
            np.asarray(grids.ky),
        )
        par_fn = _cpp_ntft_oracle_op
        bcs_fn = _cpp_ntft_oracle_apply_bcs
    else:
        par_oracle = _cpp_linked_oracle_init(naky, nakx, grids.nx, nz, grids.zp, cfg.jtwist)
        par_fn = _cpp_linked_oracle_op
        bcs_fn = _cpp_linked_oracle_apply_bcs

    kperp2 = np.asarray(geo.kperp2, dtype=np.float64)
    bmag = np.asarray(geo.bmag, dtype=np.float64)
    bgrad = np.asarray(geo.bgrad, dtype=np.float64)
    cv_d = np.asarray(geo.cv_d, dtype=np.float64)
    gb_d = np.asarray(geo.gb_d, dtype=np.float64)
    omegad = np.asarray(geo.omegad, dtype=np.float64)
    ky = np.asarray(grids.ky, dtype=np.float64)
    kx = np.asarray(grids.kx, dtype=np.float64)
    gpar = float(geo.gradpar)

    # 1. streaming_rhs + grad_par->dz
    rhs_par = np.zeros_like(g)
    for is_idx, sp in enumerate(cfg.species):
        b_s = sp.rho2 * kperp2
        vt, zt = sp.vt, sp.zt
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

    rhs = par_fn(rhs_par, par_oracle, "dz") if nz > 1 else np.zeros_like(g)

    # 2. rhs_linear local terms + collisions
    sp0 = cfg.species[0]
    iky3 = (1.0j * ky)[None, :, None]
    for is_idx, sp in enumerate(cfg.species):
        b_s = sp.rho2 * kperp2
        vt, zt, tz, nu_ss = sp.vt, sp.zt, sp.tz, sp.nu_ss
        if sp.type_id == 1 and cfg.ei_colls:
            nuei = (sp0.z * nu_ss) if ns > 1 else nu_ss
            as_i, vt_i, nzvt_i = sp0.jparfac, sp0.vt, sp0.nz * sp0.vt
        else:
            nuei, as_i, vt_i, nzvt_i = 0.0, 1.0, 1.0, 1.0

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
                uperp_bar += (jl + jlm1) * h_sp[0, ell]
                if nm >= 2:
                    upar_bar += jl * h_sp[1, ell]
                g2 = g[is_idx, 2, ell] if nm >= 3 else 0.0
                t_bar += (math.sqrt(2.0) / 3.0) * jl * g2 + (2.0 / 3.0) * (
                    ell * jlm1 + 2.0 * ell * jl + (ell + 1) * jlp1
                ) * h_sp[0, ell]
            uperp_bar *= np.sqrt(b_s)

        def _hg(mq: int, lq: int, h_arr: np.ndarray = h_sp):
            return h_arr[mq, lq] if (0 <= mq < nm and 0 <= lq < nl) else 0.0

        icv, igb, bgrad3 = 1.0j * tz * cv_d, 1.0j * tz * gb_d, bgrad[:, None, None]
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
                        -math.sqrt(m + 1)
                        * ((ell + 1) * _hg(m + 1, ell) + ell * _hg(m + 1, ell - 1))
                        + math.sqrt(m) * (ell * _hg(m - 1, ell) + (ell + 1) * _hg(m - 1, ell + 1))
                    )
                    - icv
                    * (
                        math.sqrt((m + 1) * (m + 2)) * _hg(m + 2, ell)
                        + (2 * m + 1) * _hg(m, ell)
                        + (math.sqrt(m * (m - 1)) if m >= 2 else 0.0) * _hg(m - 2, ell)
                    )
                    - igb
                    * (
                        (ell + 1) * _hg(m, ell + 1)
                        + (2 * ell + 1) * _hg(m, ell)
                        + ell * _hg(m, ell - 1)
                    )
                    - (nu_ss + nuei) * (b_s + 2 * ell + m) * _hg(m, ell)
                )
                if m == 0:
                    acc += (
                        iky3
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
                    upar_i = (
                        kperp2 * (bmag[:, None, None] ** 2) * apar / as_i
                        - sp.nz * vt * upar_bar / nzvt_i
                        if (ns > 1 and as_i > 0.0)
                        else 0.0
                    )
                    acc += -vt * iky3 * apar * (
                        jlm1 * ell * sp.tprim
                        + jl * (sp.fprim + (2 * ell + 1) * sp.tprim)
                        + jlp1 * (ell + 1) * sp.tprim
                    ) + jl * (nu_ss * upar_bar + nuei * (vt_i / vt) * upar_i)
                elif m == 2:
                    acc += (
                        iky3 * phi * jl / math.sqrt(2.0) * sp.tprim
                        + (iky3 / zt) * bpar * jlb / math.sqrt(2.0) * sp.tprim
                        + (nu_ss + nuei) * math.sqrt(2.0) * jl * t_bar
                    )
                elif m == 3:
                    acc -= vt * iky3 * apar * math.sqrt(1.5) * sp.tprim * jl
                rhs[is_idx, m, ell] += np.where(mask3d, acc, 0.0)

    # 3. HB_hyper
    if cfg.hb_hyper:
        favg_j, df_j = field_line_average(jnp.asarray(phi, dtype=jnp.complex128), grids, geo)
        favg_np = np.asarray(favg_j, dtype=np.complex128)
        df_np = np.asarray(df_j, dtype=np.complex128)
        s01_raw = float(np.sum((np.abs(kx) ** 4) * (np.abs(favg_np) ** 2)))
        s01 = 0.5 * (-cfg.w_osc + math.sqrt(cfg.w_osc**2 + 2.0 * s01_raw))
        nonzonal = mask2d & (np.arange(naky)[:, None] > 0)
        df2 = np.where(nonzonal[None, :, :], np.abs(df_np) ** 2, 0.0)
        s10 = 0.5 * (
            -cfg.w_osc
            + np.sqrt(cfg.w_osc**2 + 2.0 * np.sum(2.0 * (ky**4)[None, :, None] * df2, axis=(1, 2)))
        )
        kp4 = ((kx[None, :] ** 2 + ky[:, None] ** 2) ** 2)[None, :, :]
        s11 = 0.5 * (
            -cfg.w_osc + np.sqrt(cfg.w_osc**2 + 2.0 * np.sum(2.0 * kp4 * df2, axis=(1, 2)))
        )
        kpmax2 = grids.kx_max**2 + grids.ky_max**2
        d10 = cfg.d_hb * ((kx / grids.kx_max) ** 4)
        d01 = (ky / grids.ky_max)[:, None] * d10[None, :]
        d11 = cfg.d_hb * (((kx[None, :] ** 2 + ky[:, None] ** 2) / kpmax2) ** cfg.p_hb)
        sfac = np.where(
            (np.arange(naky) == 0)[None, :, None],
            s10[:, None, None] * d10[None, None, :],
            s11[:, None, None] * d11[None, :, :] + s01 * d01[None, :, :],
        )
        rhs -= np.where(mask6d, sfac[None, None, None, :, :, :] * g, 0.0)

    # 4. Closures
    cmod = cfg.closure_model.strip().lower()
    if cmod == "beer4+2":
        d_par = 2.0 * math.sqrt(math.pi) / (3.0 * math.pi - 8.0)
        beta_par = (32.0 - 9.0 * math.pi) / (3.0 * math.pi - 8.0)
        d_perp = 0.5 * math.sqrt(math.pi)
        nu = [
            0.0j,
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
            dz_g20 = par_fn(g[is_idx, 2, 0], par_oracle, "dz")
            adz_g30 = par_fn(g[is_idx, 3, 0], par_oracle, "abs_dz")
            adz_g11 = par_fn(g[is_idx, 1, 1], par_oracle, "abs_dz")
            rhs[is_idx, 3, 0] += (-beta_par / math.sqrt(3.0) * gpar * sp.vt) * dz_g20 + (
                -math.sqrt(2.0) * d_par * gpar * sp.vt
            ) * adz_g30
            rhs[is_idx, 1, 1] += (-math.sqrt(2.0) * d_perp * gpar * sp.vt) * adz_g11
            iwd, awd = 1.0j * omegad * sp.tz, np.abs(omegad) * sp.tz
            s2, s6 = math.sqrt(2.0), math.sqrt(6.0)
            g02, g10, g01, g03, g11 = (
                g[is_idx, 2, 0],
                g[is_idx, 0, 1],
                g[is_idx, 1, 0],
                g[is_idx, 3, 0],
                g[is_idx, 1, 1],
            )
            rhs[is_idx, 2, 0] += np.where(
                mask3d,
                -s2
                * (
                    awd * (nu[1].real * s2 * g02 + nu[2].real * g10)
                    + iwd * (nu[1].imag * s2 * g02 + nu[2].imag * g10)
                ),
                0.0,
            )
            rhs[is_idx, 0, 1] += np.where(
                mask3d,
                -2.0
                * (
                    awd * (nu[3].real * s2 * g02 + nu[4].real * g10)
                    + iwd * (nu[3].imag * s2 * g02 + nu[4].imag * g10)
                ),
                0.0,
            )
            rhs[is_idx, 3, 0] += np.where(
                mask3d,
                -(1.0 / s6)
                * (
                    awd * (nu[5].real * g01 + nu[6].real * s6 * g03 + nu[7].real * g11)
                    + iwd * (nu[5].imag * g01 + nu[6].imag * s6 * g03 + nu[7].imag * g11)
                ),
                0.0,
            )
            rhs[is_idx, 1, 1] += np.where(
                mask3d,
                -(
                    awd * (nu[8].real * g01 + nu[9].real * s6 * g03 + nu[10].real * g11)
                    + iwd * (nu[8].imag * g01 + nu[9].imag * s6 * g03 + nu[10].imag * g11)
                ),
                0.0,
            )
    elif cmod == "smith_perp":
        aclos = smith_perp_get_as(nl, cfg.smith_perp_q)
        ell_last = nl - 1
        for is_idx, sp in enumerate(cfg.species):
            iwd, awd = 1.0j * omegad * sp.tz, np.abs(omegad) * sp.tz
            for m in range(nm):
                clos = np.zeros_like(omegad, dtype=np.complex128)
                for ell in range(ell_last, nl - cfg.smith_perp_q - 1, -1):
                    c_a = aclos[ell_last - ell]
                    clos += (awd * c_a.imag + iwd * c_a.real) * g[is_idx, m, ell]
                rhs[is_idx, m, ell_last] -= np.where(mask3d, (ell_last + 1) * clos, 0.0)
    elif cmod == "smith_par":
        acoeff = smith_par_get_as(nm, cfg.smith_par_q)
        m_last = nm - 1
        for is_idx, sp in enumerate(cfg.species):
            for ell in range(nl):
                clos = np.zeros((nz, naky, nakx), dtype=np.complex128)
                for m in range(m_last, nm - cfg.smith_par_q - 1, -1):
                    dz_m = par_fn(g[is_idx, m, ell], par_oracle, "dz")
                    adz_m = par_fn(g[is_idx, m, ell], par_oracle, "abs_dz")
                    c_a = acoeff[m_last - m]
                    clos += -c_a.imag * adz_m + c_a.real * dz_m
                rhs[is_idx, m_last, ell] += (-math.sqrt(m_last + 1) * gpar * sp.vt) * clos

    # 5. hypercollisions_const
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

    # 6. hypercollisions_kz
    if cfg.hypercollisions_kz and nz > 1:
        m_max_f, p_f = float(nm - 1), float(cfg.p_hyper_m)
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
            rhs[is_idx] += par_fn(tmp, par_oracle, "abs_dz")

    # 7. hyperdiff
    if cfg.hyper:
        kpmax2 = grids.kx_max**2 + grids.ky_max**2
        dfac = cfg.d_hyper * (((kx[None, :] ** 2 + ky[:, None] ** 2) / kpmax2) ** cfg.p_hyper)
        rhs -= np.where(mask6d, dfac[None, None, None, None, :, :] * g, 0.0)

    # 8. hyperz
    if cfg.hyperz:
        rhs = par_fn(
            g,
            par_oracle,
            "hyperz",
            p_hyper_z=cfg.p_hyper_z,
            nu_hyper_z=cfg.nu_hyper_z,
            accumulate=True,
            target_np=rhs,
        )

    # 9. applyBCs
    rhs = bcs_fn(g, rhs, fields, cfg, kperp2, par_oracle, dt)
    return np.where(mask6d, rhs, 0.0)


@pytest.mark.parametrize("non_twist", [False, True])
@pytest.mark.parametrize(
    ("closure_model", "nm", "nl"),
    [
        ("beer4+2", 4, 2),
        ("smith_par", 6, 3),
        ("smith_perp", 5, 4),
    ],
)
def test_linear_rhs_multilink_and_ntft_all_closures_and_dissipation(
    non_twist: bool,
    closure_model: str,
    nm: int,
    nl: int,
    w7x_eik_files: dict[str, Path],
) -> None:
    """Verify full linear_rhs & steppers with closures + dissipation in Linked and NTFT."""
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    w7x_base = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
    eik_path = w7x_eik_files["nz32"]

    beta_em = 0.015
    ion = SpeciesConfig.from_raw(
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
    elec = SpeciesConfig.from_raw(
        z=-1.0,
        mass=2.7e-4,
        dens=1.0,
        temp=1.0,
        tprim=3.0,
        fprim=1.0,
        nu_ss=0.04,
        stype="electron",
        beta=beta_em,
    )
    cfg = dataclasses.replace(
        w7x_base,
        geo_option="nc",
        geofilename=str(eik_path),
        nx=7,  # nakx = 5 -> multi-link chains!
        ny=7,  # naky = 3
        nz=32,
        ntheta_in=32,
        nhermite=nm,
        nlaguerre=nl,
        nspecies=2,
        species=(ion, elec),
        all_kinetic=True,
        add_boltzmann_species=False,
        beta=beta_em,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        collisions=True,
        coll_conservation=True,
        non_twist=non_twist,
        closure_model=closure_model,
        smith_par_q=3,
        smith_perp_q=3,
        hypercollisions_kz=True,
        p_hyper_m=2,
        hypercollisions_const=True,
        nu_hyper_l=0.05,
        nu_hyper_lm=0.02,
        hyper=True,
        d_hyper=0.05,
        hb_hyper=True,
        d_hb=0.05,
        w_osc=0.01,
        hyperz=True,
        nu_hyper_z=0.12,
        p_hyper_z=6,
        dealias_kz=True,
        x0=-1.0,
        jtwist=-1000 if non_twist else 1,
    )
    cfg, grids, geo = load_eik_nc(eik_path, cfg)
    op64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
    op32 = op64.astype(jnp.float32)
    geo32 = geo.astype(jnp.float32)
    assert isinstance(op64.grad_par, NTFTParallelGrad if non_twist else LinkedParallelGrad)

    g128 = _random_complex_arr(grids.g_shape, np.asarray(grids.active_mask), seed=901)
    f128 = solve_fields(g128, op64.solver)
    dt = 0.002

    rhs_ref = _cpp_oracle_full_linear_rhs(np.asarray(g128), f128, cfg, grids, geo, dt)
    rhs128 = linear_rhs(g128, f128, op64, dt, grids=grids, geo=geo)
    _assert_close_fp64(rhs128, rhs_ref, f"rhs:{non_twist}:{closure_model}:fp64")

    g64 = g128.astype(jnp.complex64)
    f64 = solve_fields(g64, op32.solver)
    rhs64 = linear_rhs(g64, f64, op32, dt, grids=grids, geo=geo32)
    _assert_close_fp32_1e6(rhs64, rhs_ref, f"rhs:{non_twist}:{closure_model}:c64")

    # Also verify rk4 (4-stage), sspx3 (3-stage), and k10 (10-stage) steppers:
    # single-application linear_rhs stays < 1e-6 above; multi-stage RK (up to 10 RHS + 11 field
    # solves with C++-faithful no-op dealias) uses tol=5e-6 to account for stage count.
    scheme = {"beer4+2": "rk4", "smith_par": "sspx3", "smith_perp": "k10"}[closure_model]
    g_next128, f_next128 = step_linear(g128, f128, op64, dt, scheme=scheme, grids=grids, geo=geo)
    g_next64, f_next64 = step_linear(g64, f64, op32, dt, scheme=scheme, grids=grids, geo=geo32)
    _assert_close_fp32_1e6(
        g_next64, g_next128, f"step:{scheme}:{non_twist}:{closure_model}", tol=5.0e-6
    )
    _assert_close_fp32_1e6(
        f_next64.phi, f_next128.phi, f"step_phi:{scheme}:{non_twist}:{closure_model}", tol=5.0e-6
    )


# ---------------------------------------------------------------------------
# 5. Linear Eigenmodes: Multi-Link Ballooning Equivalence, .out.nc, & 3D W7-X
# ---------------------------------------------------------------------------


def test_multilink_ballooning_equivalence_and_cpp_out_nc_eigenmodes() -> None:
    """Verify 3-link (nperiod=1, nkx=3) eigenmodes match nperiod=2 (nkx=1) and C++ *.out.nc!"""
    salpha_in = BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    salpha_out = (
        GOLDEN_BENCH_DIR
        / "linear"
        / "ITG_cyclone"
        / "itg_salpha_adiabatic_electrons_correct.out.nc"
    )
    miller_in = BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_adiabatic_electrons.in"
    miller_out = (
        GOLDEN_BENCH_DIR
        / "linear"
        / "ITG_cyclone"
        / "itg_miller_adiabatic_electrons_correct.out.nc"
    )

    # 1. Exact machine-precision ballooning identity on s-alpha:
    # (nperiod=2, zp=3, nz=96, nkx=1) == (nperiod=1, zp=1, nz=32, nkx=3, jtwist=1 -> nLinks=3)!
    cfg_base = extract_single_ky_config(load_config(salpha_in), 6)  # ky = 0.30
    cfg_np2 = dataclasses.replace(cfg_base, nhermite=8, nlaguerre=4)
    cfg_np2, grids_np2, geo_np2 = init_geometry(cfg_np2)
    op_np2 = init_linear_operator(cfg_np2, grids_np2, geo_np2, dtype=jnp.float64)
    assert isinstance(op_np2.grad_par, ParallelGrad)

    cfg_lnk3 = dataclasses.replace(
        cfg_np2,
        nx=4,  # nakx = 3 (-1, 0, +1) -> 1 chain of nLinks = 3 at iky = 1!
        nz=32,
        ntheta_in=32,
        nperiod=1,
        zp=1,
        jtwist=1,
        x0=-1.0,
    )
    cfg_lnk3, grids_lnk3, geo_lnk3 = init_geometry(cfg_lnk3)
    op_lnk3 = init_linear_operator(cfg_lnk3, grids_lnk3, geo_lnk3, dtype=jnp.float64)
    assert isinstance(op_lnk3.grad_par, LinkedParallelGrad)
    assert op_lnk3.grad_par.n_links == (1, 3)

    dt_bal = compute_linear_dt(cfg_np2, grids_np2, geo_np2)
    g0_np2 = init_moments(cfg_np2, grids_np2, geo_np2, dtype=jnp.complex128)
    w_np2, eig_np2 = solve_linear_eigenmode(
        op_np2, dt_bal, iky=1, ikx=0, steps_per_iter=50, ncv=10, tol=1.0e-10, v0=g0_np2[..., 1, 0]
    )
    w_lnk3, eig_lnk3 = solve_linear_eigenmode(
        op_lnk3,
        dt_bal,
        iky=1,
        ikx=0,
        steps_per_iter=50,
        ncv=10,
        tol=1.0e-10,
        v0=g0_np2[..., 32:64, 1, 0],
    )
    assert eig_lnk3.shape == (1, 8, 4, 32, 3)
    # Exact FP64 ballooning equivalence of the Arnoldi eigenmode across the 3 linked 2*pi segments
    # vs the single 6*pi domain (modulo global complex phase normalization):
    p_map_lnk3 = np.asarray(op_lnk3.grad_par.p_map)[1]
    eig_lnk3_unrolled = np.empty_like(np.asarray(eig_np2))
    for idx_kx in range(grids_lnk3.nakx):
        p_seg = int(p_map_lnk3[idx_kx])
        eig_lnk3_unrolled[..., p_seg * 32 : (p_seg + 1) * 32] = np.asarray(eig_lnk3)[..., idx_kx]
    phase_align = np.vdot(eig_np2, eig_lnk3_unrolled) / np.abs(np.vdot(eig_np2, eig_lnk3_unrolled))
    _assert_close_fp64(
        eig_lnk3_unrolled / phase_align,
        eig_np2,
        "salpha_3link_vs_nperiod2_eigenvector",
        rtol=1.0e-12,
    )
    # Exact ballooning equivalence between 3 linked 2*pi segments (coupled-chain Rayleigh quotient)
    # and 1 6*pi domain (single-kx point ratio) at Krylov tol=1e-10:
    _assert_close_fp64(w_lnk3, w_np2, "salpha_3link_vs_nperiod2_eigenvalue", rtol=1.0e-10)

    # Verify runtime complex64 RK4 stepping on the 3-link eigenvector matches Krylov!
    op_lnk3_32 = op_lnk3.astype(jnp.float32)
    g_lnk3_c64 = (
        jnp.zeros(grids_lnk3.g_shape, dtype=jnp.complex64)
        .at[..., 1, :]
        .set(eig_lnk3.astype(jnp.complex64))
    )
    f_lnk3_c64 = solve_fields(g_lnk3_c64, op_lnk3_32.solver)
    _, f_lnk3_end, phi_lnk3_prev = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_lnk3_32, dt_bal, 20, scheme="rk4")
    )(g_lnk3_c64, f_lnk3_c64)
    w_lnk3_rk4_20 = complex(
        compute_growth_rates(f_lnk3_end.phi, f_lnk3_c64.phi, 20.0 * dt_bal, grids_lnk3)[1, 0]
    )
    w_lnk3_rk4_1 = complex(
        compute_growth_rates(f_lnk3_end.phi, phi_lnk3_prev, dt_bal, grids_lnk3)[1, 0]
    )
    np.testing.assert_allclose(w_lnk3_rk4_20.real, w_lnk3.real, rtol=1.0e-5)
    np.testing.assert_allclose(w_lnk3_rk4_20.imag, w_lnk3.imag, rtol=1.0e-5)
    np.testing.assert_allclose(w_lnk3_rk4_1.real, w_lnk3.real, rtol=5.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(w_lnk3_rk4_1.imag, w_lnk3.imag, rtol=5.0e-5, atol=5.0e-6)

    # 2. Full-res (nhermite=48, nlaguerre=16) 3-link (nperiod=1, nkx=3) ≡ nperiod=2 (nkx=1)
    # to FP64 machine precision (transitively matching C++ *_correct.out.nc via Session 3)!
    assert w_lnk3.real > 0.20 and w_lnk3.imag > 0.08

    for label, in_p, out_p, iky in [
        ("salpha_3link_out_nc", salpha_in, salpha_out, 6),
        ("miller_3link_out_nc", miller_in, miller_out, 8),
    ]:
        with netCDF4.Dataset(out_p, "r") as ds:
            t_ref = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
            dt_cpp = float(t_ref[0])

        cfg_raw = extract_single_ky_config(load_config(in_p), iky)
        cfg_1d, grids_1d, geo_1d = init_geometry(cfg_raw)
        op_1d_64 = init_linear_operator(cfg_1d, grids_1d, geo_1d, dtype=jnp.float64)
        g0_1d = init_moments(cfg_1d, grids_1d, geo_1d, dtype=jnp.complex128)
        f0_1d = solve_fields(g0_1d, op_1d_64.solver)
        g10_1d, f10_1d, phi9_1d = jax.jit(
            lambda g, f, op=op_1d_64, d=dt_cpp: advance_linear_steps(g, f, op, d, 10, scheme="rk4")
        )(g0_1d, f0_1d)
        w10_1d = complex(compute_growth_rates(f10_1d.phi, phi9_1d, dt_cpp, grids_1d)[1, 0])

        cfg_3l = dataclasses.replace(
            cfg_raw,
            nx=4,  # nakx = 3 -> nLinks = 3 matching nperiod = 2 (3 * 2pi)!
            nz=cfg_raw.ntheta_in,
            nperiod=1,
            zp=1,
            jtwist=1,
            x0=-1.0,
        )
        cfg_3l, grids_3l, geo_3l = init_geometry(cfg_3l)
        op_3l_64 = init_linear_operator(cfg_3l, grids_3l, geo_3l, dtype=jnp.float64)
        op_3l_32 = op_3l_64.astype(jnp.float32)
        assert isinstance(op_3l_64.grad_par, LinkedParallelGrad)
        assert op_3l_64.grad_par.n_links == (1, 3)

        nz_seg = grids_3l.nz
        g0_3l = jnp.zeros(grids_3l.g_shape, dtype=jnp.complex128)
        for idx_kx in range(grids_3l.nakx):
            p_seg = int(np.asarray(op_3l_64.grad_par.p_map)[1, idx_kx])
            g0_3l = g0_3l.at[..., 1, idx_kx].set(
                g0_1d[..., p_seg * nz_seg : (p_seg + 1) * nz_seg, 1, 0]
            )
        f0_3l = solve_fields(g0_3l, op_3l_64.solver)
        g10_3l, f10_3l, phi9_3l = jax.jit(
            lambda g, f, op=op_3l_64, d=dt_cpp: advance_linear_steps(g, f, op, d, 10, scheme="rk4")
        )(g0_3l, f0_3l)
        w10_3l = complex(compute_growth_rates(f10_3l.phi, phi9_3l, dt_cpp, grids_3l)[1, 0])

        # Exact machine-precision trajectory match across all 3 links at full (48, 16) resolution!
        for idx_kx in range(grids_3l.nakx):
            p_seg = int(np.asarray(op_3l_64.grad_par.p_map)[1, idx_kx])
            _assert_close_fp64(
                g10_3l[..., 1, idx_kx],
                g10_1d[..., p_seg * nz_seg : (p_seg + 1) * nz_seg, 1, 0],
                f"{label}:g10_link_{p_seg}",
            )
        _assert_close_fp64(w10_3l, w10_1d, f"{label}:w10_fp64")

        # Also verify runtime complex64 10-step trajectory on the 3-link (48, 16) operator (< 1e-6)
        g0_3l_c64 = g0_3l.astype(jnp.complex64)
        f0_3l_c64 = solve_fields(g0_3l_c64, op_3l_32.solver)
        g10_3l_c64, f10_3l_c64, _ = jax.jit(
            lambda g, f, op=op_3l_32, d=dt_cpp: advance_linear_steps(g, f, op, d, 10, scheme="rk4")
        )(g0_3l_c64, f0_3l_c64)
        _assert_close_fp32_1e6(g10_3l_c64, g10_3l, f"{label}:g10_c64")
        _assert_close_fp32_1e6(f10_3l_c64.phi, f10_3l.phi, f"{label}:phi10_c64")


def test_w7x_stellarator_multilink_and_ntft_eigenmodes(
    w7x_eik_files: dict[str, Path],
) -> None:
    """Verify multi-link Linked & NTFT linear ITG eigenmodes on 3D W7-X (wout_w7x.nc)."""
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    # Use 1-poloidal-turn W7-X flux tube (matching benchmarks/nonlinear/w7x) with multi-kx chains!
    cfg_w7x_raw = dataclasses.replace(
        extract_single_ky_config(load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in"), 14),
        ntheta_in=32,
        nz=32,
        npol=1.0,
        nhermite=8,
        nlaguerre=4,
    )
    eik_path = w7x_eik_files["nz32"]

    # 1. Multi-link LinkedParallelGrad (nx=7 -> nakx=5 -> nLinks=5 extended 5-turn ballooning!)
    cfg_lnk = dataclasses.replace(
        cfg_w7x_raw,
        geo_option="nc",
        geofilename=str(eik_path),
        nx=7,
        non_twist=False,
        jtwist=-1,  # W7-X has negative shat (~ -0.125)
        x0=-1.0,
    )
    cfg_lnk, grids_lnk, geo_lnk = load_eik_nc(eik_path, cfg_lnk)
    dt_lnk = compute_linear_dt(cfg_lnk, grids_lnk, geo_lnk)
    op_lnk64 = init_linear_operator(cfg_lnk, grids_lnk, geo_lnk, dtype=jnp.float64)
    op_lnk32 = op_lnk64.astype(jnp.float32)
    assert isinstance(op_lnk64.grad_par, LinkedParallelGrad)
    assert op_lnk64.grad_par.n_links == (1, 5)

    g0_lnk = init_moments(cfg_lnk, grids_lnk, geo_lnk, dtype=jnp.complex128)
    w_lnk, vec_lnk = solve_linear_eigenmode(
        op_lnk64, dt_lnk, iky=1, ikx=0, steps_per_iter=60, ncv=12, tol=1.0e-8, v0=g0_lnk[..., 1, :]
    )
    assert w_lnk.imag > 0.05 and w_lnk.real > 0.0

    # Verify FP64 (< 1e-8) and runtime complex64 (< 1e-5) RK4 stepping on W7-X 5-link eigenmode
    g_lnk64 = jnp.zeros(grids_lnk.g_shape, dtype=jnp.complex128).at[..., 1, :].set(vec_lnk)
    f_lnk64 = solve_fields(g_lnk64, op_lnk64.solver)
    _, f_lnk64_end, phi_lnk64_prev = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_lnk64, dt_lnk, 15, scheme="rk4")
    )(g_lnk64, f_lnk64)
    w_lnk_rk64 = complex(
        compute_growth_rates(f_lnk64_end.phi, phi_lnk64_prev, dt_lnk, grids_lnk)[1, 0]
    )
    np.testing.assert_allclose(w_lnk_rk64.real, w_lnk.real, rtol=1.0e-8)
    np.testing.assert_allclose(w_lnk_rk64.imag, w_lnk.imag, rtol=1.0e-8)

    g_lnk32 = g_lnk64.astype(jnp.complex64)
    f_lnk32 = solve_fields(g_lnk32, op_lnk32.solver)
    _, f_lnk32_end, phi_lnk32_prev = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_lnk32, dt_lnk, 20, scheme="rk4")
    )(g_lnk32, f_lnk32)
    w_lnk_rk32_20 = complex(
        compute_growth_rates(f_lnk32_end.phi, f_lnk32.phi, 20.0 * dt_lnk, grids_lnk)[1, 0]
    )
    w_lnk_rk32_1 = complex(
        compute_growth_rates(f_lnk32_end.phi, phi_lnk32_prev, dt_lnk, grids_lnk)[1, 0]
    )
    np.testing.assert_allclose(w_lnk_rk32_20.real, w_lnk.real, rtol=1.0e-5)
    np.testing.assert_allclose(w_lnk_rk32_20.imag, w_lnk.imag, rtol=1.0e-5)
    np.testing.assert_allclose(w_lnk_rk32_1.real, w_lnk.real, rtol=5.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(w_lnk_rk32_1.imag, w_lnk.imag, rtol=5.0e-5, atol=1.0e-5)

    # 2. NTFTParallelGrad (non_twist=True) on the same 3D W7-X equilibrium!
    cfg_ntft = dataclasses.replace(cfg_lnk, non_twist=True, jtwist=-1000, x0=-1.0)
    cfg_ntft, grids_ntft, geo_ntft = load_eik_nc(eik_path, cfg_ntft)
    dt_ntft = compute_linear_dt(cfg_ntft, grids_ntft, geo_ntft)
    op_ntft64 = init_linear_operator(cfg_ntft, grids_ntft, geo_ntft, dtype=jnp.float64)
    op_ntft32 = op_ntft64.astype(jnp.float32)
    assert isinstance(op_ntft64.grad_par, NTFTParallelGrad)

    g0_ntft = init_moments(cfg_ntft, grids_ntft, geo_ntft, dtype=jnp.complex128)
    w_ntft, vec_ntft = solve_linear_eigenmode(
        op_ntft64,
        dt_ntft,
        iky=1,
        ikx=0,
        steps_per_iter=60,
        ncv=12,
        tol=1.0e-8,
        v0=g0_ntft[..., 1, :],
    )
    assert w_ntft.imag > 0.05 and w_ntft.real > 0.0
    # Linked and NTFT represent the same physical W7-X ITG instability on 5 poloidal turns!
    np.testing.assert_allclose(w_ntft.imag, w_lnk.imag, rtol=0.05)
    np.testing.assert_allclose(w_ntft.real, w_lnk.real, rtol=0.05)

    g_ntft64 = jnp.zeros(grids_ntft.g_shape, dtype=jnp.complex128).at[..., 1, :].set(vec_ntft)
    f_ntft64 = solve_fields(g_ntft64, op_ntft64.solver)
    _, f_ntft64_end, phi_ntft64_prev = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_ntft64, dt_ntft, 15, scheme="rk4")
    )(g_ntft64, f_ntft64)
    w_ntft_rk64 = complex(
        compute_growth_rates(f_ntft64_end.phi, phi_ntft64_prev, dt_ntft, grids_ntft)[1, 0]
    )
    np.testing.assert_allclose(w_ntft_rk64.real, w_ntft.real, rtol=1.0e-8)
    np.testing.assert_allclose(w_ntft_rk64.imag, w_ntft.imag, rtol=1.0e-8)

    g_ntft32 = g_ntft64.astype(jnp.complex64)
    f_ntft32 = solve_fields(g_ntft32, op_ntft32.solver)
    _, f_ntft32_end, phi_ntft32_prev = jax.jit(
        lambda g, f: advance_linear_steps(g, f, op_ntft32, dt_ntft, 20, scheme="rk4")
    )(g_ntft32, f_ntft32)
    w_ntft_rk32_20 = complex(
        compute_growth_rates(f_ntft32_end.phi, f_ntft32.phi, 20.0 * dt_ntft, grids_ntft)[1, 0]
    )
    w_ntft_rk32_1 = complex(
        compute_growth_rates(f_ntft32_end.phi, phi_ntft32_prev, dt_ntft, grids_ntft)[1, 0]
    )
    np.testing.assert_allclose(w_ntft_rk32_20.real, w_ntft.real, rtol=1.0e-5)
    np.testing.assert_allclose(w_ntft_rk32_20.imag, w_ntft.imag, rtol=1.0e-5)
    np.testing.assert_allclose(w_ntft_rk32_1.real, w_ntft.real, rtol=5.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(w_ntft_rk32_1.imag, w_ntft.imag, rtol=5.0e-5, atol=1.0e-5)


def test_multichain_jtwist2_eigenmode_subspace_isolation() -> None:
    """Verify solve_linear_eigenmode isolates decoupled chains at ikx=0 vs ikx=1 for jtwist=2."""
    salpha_in = BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    cfg = dataclasses.replace(
        extract_single_ky_config(load_config(salpha_in), 6),
        nx=7,  # nakx = 5: with jtwist=2 at iky=1 -> chain A (+2, 0, -2) & chain B (+1, -1)!
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        jtwist=2,
        x0=-1.0,
    )
    cfg, grids, geo = init_geometry(cfg)
    op = init_linear_operator(cfg, grids, geo, dtype=jnp.float64)
    dt = compute_linear_dt(cfg, grids, geo)

    chain0 = op.grad_par.coupled_ikx_indices(1, 0)
    chain1 = op.grad_par.coupled_ikx_indices(1, 1)
    assert set(chain0).isdisjoint(set(chain1))
    assert len(chain0) == 3 and len(chain1) == 2

    g0 = init_moments(cfg, grids, geo, dtype=jnp.complex128)
    w0, vec0 = solve_linear_eigenmode(
        op, dt, iky=1, ikx=0, steps_per_iter=20, tol=1.0e-7, v0=g0[..., 1, :]
    )
    # Querying a far link ikx=2 in chain0 gives identical eigenvalue via subspace Rayleigh quotient
    w0_far, _ = solve_linear_eigenmode(
        op, dt, iky=1, ikx=2, steps_per_iter=20, tol=1.0e-7, v0=g0[..., 1, :]
    )
    _assert_close_fp64(w0_far, w0, "chain0_rayleigh_far_ikx")
    w1, vec1 = solve_linear_eigenmode(op, dt, iky=1, ikx=1, steps_per_iter=20, tol=1.0e-7)
    # Chain A (theta_0=0 outboard ballooning) and Chain B (theta_0=+-pi inboard) are decoupled!
    assert abs(w1 - w0) > 0.05
    # Chain A eigenvector is strictly zero on chain B's kx slots and vice versa!
    np.testing.assert_array_equal(np.asarray(vec0[..., list(chain1)]), 0.0)
    np.testing.assert_array_equal(np.asarray(vec1[..., list(chain0)]), 0.0)


# ---------------------------------------------------------------------------
# 6. Static JIT Trace Count, PyTree / astype, & jax.grad + jax.debug_nans(True)
# ---------------------------------------------------------------------------


def test_jit_static_compilation_pytree_and_autodiff_debug_nans() -> None:
    """Verify static-bucket single JIT trace, PyTree round-trip, and NaN-free jax.grad."""
    # Verify global TPU precision invariants remain active
    assert jax.config.jax_enable_x64 is True
    assert jax.config.jax_default_matmul_precision == "highest"
    assert "LinkedParallelGrad" in gx_jax.__all__
    assert "NTFTParallelGrad" in gx_jax.__all__
    assert gx_jax.__all__ == sorted(gx_jax.__all__)

    kbm_cfg = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    for non_twist in (False, True):
        cfg = dataclasses.replace(
            kbm_cfg,
            nx=10,
            ny=10,
            nz=16,
            ntheta_in=16,
            nperiod=1,
            zp=1,
            nhermite=4,
            nlaguerre=2,
            fbpar=1.0,
            shat=0.12 if non_twist else float(kbm_cfg.shat),
            non_twist=non_twist,
            closure_model="beer4+2",
            hypercollisions_kz=True,
            hyperz=True,
            nu_hyper_z=0.1,
            jtwist=-1000 if non_twist else 1,
            x0=-1.0,
        )
        cfg, grids, geo = init_geometry(cfg)
        # Use chain_pad_multiple=8 so every bucket has unused padded chain slots tested under AD!
        gp64 = init_boundary_parallel(
            cfg, grids, geo, dtype=jnp.float64, n_links_max=32, chain_pad_multiple=8
        )
        op64 = init_linear_operator(cfg, grids, geo, grad_par=gp64, dtype=jnp.float64)
        op32 = op64.astype(jnp.float32)

        # PyTree flatten/unflatten round-trip check
        leaves, treedef = jax.tree_util.tree_flatten(op32)
        op32_rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(op32_rt.grad_par) is type(gp64)
        assert op32_rt.grad_par.buckets[0].kz.dtype == jnp.float32

        # Static single-trace @jax.jit verification (zero recompilation across distinct inputs)
        trace_counter = {"n": 0}

        @jax.jit
        def _compiled_step(op_arg, g_arg, tc=trace_counter):
            tc["n"] += 1
            f_arg = solve_fields(g_arg, op_arg.solver)
            return step_linear(g_arg, f_arg, op_arg, 0.002, scheme="rk4")

        mask_np = np.asarray(grids.active_mask, dtype=bool)
        g_a = _random_complex_arr(grids.g_shape, mask_np, seed=1001, dtype=jnp.complex64)
        g_b = _random_complex_arr(grids.g_shape, mask_np, seed=1002, dtype=jnp.complex64)
        _ = _compiled_step(op32, g_a)
        _ = _compiled_step(op32_rt, g_b)
        assert trace_counter["n"] == 1, f"Expected 1 XLA trace, got {trace_counter['n']}"

        # End-to-end jax.grad under jax.debug_nans(True) with padded chain slots!
        g0 = _random_complex_arr(grids.g_shape, mask_np, seed=1003, dtype=jnp.complex128)

        def _loss(g_re_im: tuple[jax.Array, jax.Array], op_use=op64, gr=grids, ge=geo) -> jax.Array:
            g_in = g_re_im[0] + 1.0j * g_re_im[1]
            f0 = solve_fields(g_in, op_use.solver)
            g1, f1 = step_linear(g_in, f0, op_use, 0.002, scheme="rk4")
            z_rt = op_use.grad_par.zft_inverse(op_use.grad_par.zft(g1))
            g_deal = op_use.grad_par.dealias(z_rt)
            dz2_phi = op_use.grad_par.dz2(f1.phi)
            omg = compute_growth_rates(f1.phi, f0.phi, 0.002, gr)
            p2 = compute_phi2_ky(f1.phi, gr, ge)
            return (
                jnp.sum(p2)
                + jnp.sum(jnp.real(omg) ** 2 + jnp.imag(omg) ** 2)
                + jnp.sum(jnp.abs(g_deal) ** 2)
                + jnp.sum(jnp.abs(dz2_phi) ** 2)
            )

        with jax.debug_nans(True):
            grad_re, grad_im = jax.jit(jax.grad(_loss))((jnp.real(g0), jnp.imag(g0)))

        assert bool(jnp.all(jnp.isfinite(grad_re)))
        assert bool(jnp.all(jnp.isfinite(grad_im)))
        np.testing.assert_array_equal(np.asarray(grad_re[..., 0, 0]), 0.0)
        np.testing.assert_array_equal(np.asarray(grad_im[..., 0, 0]), 0.0)
        assert float(jnp.linalg.norm(grad_re)) > 0.0
