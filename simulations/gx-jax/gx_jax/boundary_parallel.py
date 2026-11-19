"""Linked twist-and-shift and Non-Twisting Flux-Tube (NTFT) parallel operators for GX-JAX.

Ported strictly from:
- ``gx/include/grad_parallel.h`` (``GradParallelLinked``, ``GradParallelNTFT``)
- ``gx/src/grad_parallel_linked.cu`` (``get_nClasses``, ``get_nLinks_nChains``,
  ``kt2ki``, ``fill``, ``kFill``, ``dz``, ``dz2``, ``abs_dz``, ``hyperz``, ``zft``,
  ``zft_inverse``, ``applyBCs``, ``identity``, ``linkPrint``)
- ``gx/src/grad_parallel_NTFT.cu`` (``get_mode_nums_ntft``,
  ``get_nClasses_ntft``, ``get_nChains_nLinks_ntft``, ``kFill_ntft``, ``calc_idx0``,
  ``dz``, ``dz2``, ``abs_dz``, ``hyperz``, ``zft``, ``zft_inverse``, ``applyBCs``,
  ``linkPrint``)
- ``gx/src/device_funcs.cu`` (``init_kzLinked``, ``init_kzLinkedNTFT``,
  ``i_kzLinked``, ``i_kzLinkedNTFT``, ``mkz2_Linked``, ``mkz2_LinkedNTFT``,
  ``abs_kzLinked``, ``abs_kzLinkedNTFT``, ``zfts_Linked``, ``zfts_LinkedNTFT``,
  ``hyperkzLinked``, ``hyperkzLinkedNTFT``, ``ikzLinked_kernel``, ``abskzLinked_kernel``,
  ``mkz2Linked_kernel``, ``hyperkzLinked_kernel``, ``linkedCopy``, ``linkedCopyBackAll``,
  ``linkedAccumulateBackAll``, ``linkedCopyNTFT``, ``linkedCopyBackNTFT``,
  ``linkedAccumulateBackNTFT``, ``dampEnds_linked``, ``dampEnds_linkedNTFT``)

Static Bucket Architecture:
- Both ``LinkedParallelGrad`` and ``NTFTParallelGrad`` partition extended field-line chains into
  fixed-shape buckets grouped by exact chain FFT length ``L`` (``nz * n_links`` for standard
  twist-and-shift, ``n_links`` grid points for NTFT) so XLA compiles static 1D FFT shapes with
  zero recompilation and zero Python loops over individual chains.
- Optional ``chain_pad_multiple >= 1`` pads each bucket's ``n_chains`` axis to a fixed tile
  multiple while masking unused chain slots cleanly in both forward and reverse-mode AD.
- Scatter-back uses a precomputed single-gather inverse permutation (vectorizing C++
  ``linkedCopyBackAll`` / ``linkedCopyBackNTFT`` with zero XLA scatter ops), preserving Option A
  trailing ``(..., nz, naky, nakx)`` layout across 3D, 4D, 5D, and 6D inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gx_jax.config import GXConfig
from gx_jax.fields import Fields, FieldSolver, _complex_and_real_dtypes
from gx_jax.geometry_nc import Geometry
from gx_jax.grids import Grids
from gx_jax.linear_rhs import build_nonadiabatic_h
from gx_jax.mxu_fft import (
    MXU1DPlan,
    init_mxu_1d_plan,
    mxu_fft,
    mxu_ifft,
    resolve_fft_backend,
)


@dataclass(frozen=True)
class LinkedChainTopology:
    """Host-side linked chain topology matching ``GradParallelLinked`` (``L521-725``)."""

    n_classes: int
    n_links: tuple[int, ...]
    n_chains: tuple[int, ...]
    iky_chains: tuple[np.ndarray, ...]  # (n_chains[c], n_links[c]) int32 in [0..naky-1]
    ikx_chains: tuple[np.ndarray, ...]  # (n_chains[c], n_links[c]) int32 Option A [0..nakx-1]
    ikx_full_chains: tuple[np.ndarray, ...]  # (n_chains[c], n_links[c]) int32 full [0..nx-1]
    idx_left: np.ndarray  # (naky, nakx) int32
    idx_right: np.ndarray  # (naky, nakx) int32
    links_l: np.ndarray  # (naky, nakx) int32
    links_r: np.ndarray  # (naky, nakx) int32
    p_map: np.ndarray  # (naky, nakx) int32
    n_map: np.ndarray  # (naky, nakx) int32
    c_map: np.ndarray  # (naky, nakx) int32
    n_links_map: np.ndarray  # (naky, nakx) int32
    n_chains_map: np.ndarray  # (naky, nakx) int32


@dataclass(frozen=True)
class NTFTChainTopology:
    """Host-side NTFT ballooning chain topology matching ``GradParallelNTFT`` (``L431-647``)."""

    mode_count: int
    mode_nums_seq: np.ndarray  # (nz, nakx, naky) int32 in sequential [-K..+K] idx order
    mode_nums_active: np.ndarray  # (nz, naky, nakx) int32 in Option A active ikx order
    mode_size_raw: np.ndarray  # (mode_count,) int32 pre-nExtra sizes
    mode_size_ref: np.ndarray  # (mode_count,) int32 post-nExtra sizes
    n_extra: np.ndarray  # (mode_count,) int32 trimmed endpoint counts (> n_links_max)
    n_classes: int
    n_links: tuple[int, ...]  # Grid points per chain in class c
    n_chains: tuple[int, ...]  # Number of chains in class c
    idz_chains: tuple[np.ndarray, ...]  # (n_chains[c], n_links[c]) int32 in [0..nz-1]
    iky_chains: tuple[np.ndarray, ...]  # (n_chains[c], n_links[c]) int32 in [0..naky-1]
    ikx_chains: tuple[np.ndarray, ...]  # (n_chains[c], n_links[c]) int32 Option A [0..nakx-1]
    ikxdz_full_chains: tuple[np.ndarray, ...]  # (n_chains[c], n_links[c]) idx0_full + nx * idz


def _kzm_linked_np(fft_len: int) -> np.ndarray:
    """Compute 2/3 parallel dealiasing mask on a chain of length ``fft_len`` in NumPy."""
    if fft_len <= 0:
        raise ValueError(f"fft_len={fft_len} must be positive.")
    idx = np.arange(int(fft_len), dtype=np.int32)
    cutoff = (int(fft_len) - 1) // 3
    zero_band = (idx > cutoff) & (idx < (int(fft_len) - cutoff))
    return np.where(zero_band, 0, 1).astype(np.int32)


def _kz_linked_np(nz: int, zp: int, n_links: int, *, dealias_kz: bool = False) -> np.ndarray:
    if nz <= 0 or n_links <= 0:
        raise ValueError(f"nz={nz} and n_links={n_links} must be positive.")
    if zp == 0:
        raise ValueError("zp must be non-zero.")
    nz_l = int(nz) * int(n_links)
    idx = np.arange(nz_l, dtype=np.int32)
    ikz = np.where(idx < (nz_l // 2 + 1), idx, idx - nz_l)
    kz = ikz.astype(np.float64) / float(int(zp) * int(n_links))
    if dealias_kz:
        kzm = _kzm_linked_np(nz_l)
        kz = np.where(kzm == 1, kz, 0.0)
    return kz


def _kz_linked_ntft_np(nz: int, zp: int, n_links: int, *, dealias_kz: bool = False) -> np.ndarray:
    if nz <= 0 or n_links <= 0:
        raise ValueError(f"nz={nz} and n_links={n_links} must be positive.")
    if zp == 0:
        raise ValueError("zp must be non-zero.")
    nz_l = int(n_links)
    idx = np.arange(nz_l, dtype=np.int32)
    ikz = np.where(idx < (nz_l // 2 + 1), idx, idx - nz_l)
    kz = (ikz.astype(np.float64) * float(nz)) / float(int(zp) * nz_l)
    if dealias_kz:
        kzm = _kzm_linked_np(nz_l)
        kz = np.where(kzm == 1, kz, 0.0)
    return kz


def _hyperz_weight_np(kz_np: np.ndarray, p_hyper_z: int) -> np.ndarray:
    nz_l = int(kz_np.shape[0])
    if nz_l <= 1:
        return np.zeros((nz_l,), dtype=np.float64)
    kz_max = float(abs(kz_np[nz_l // 2]))
    if kz_max <= 0.0:
        return np.zeros((nz_l,), dtype=np.float64)
    return (np.abs(kz_np) / kz_max) ** int(p_hyper_z)


def init_kz_linked(
    nz: int,
    zp: int,
    n_links: int,
    *,
    dealias_kz: bool = False,
) -> jax.Array:
    """Compute linked-chain parallel wavenumbers ``kzLinked`` in IEEE ``float64``.

    Ported strictly from ``init_kzLinked`` (``device_funcs.cu:L2475-2488``):
    assigns GX's positive Nyquist mode ``+(nzL // 2) / (zp * n_links)`` at ``i = nzL // 2``
    when ``nzL = nz * n_links`` is even.
    """
    return jnp.asarray(_kz_linked_np(nz, zp, n_links, dealias_kz=dealias_kz), dtype=jnp.float64)


def init_kz_linked_ntft(
    nz: int,
    zp: int,
    n_links: int,
    *,
    dealias_kz: bool = False,
) -> jax.Array:
    """Compute NTFT chain parallel wavenumbers ``kzLinked`` in IEEE ``float64``.

    Ported strictly from ``init_kzLinkedNTFT`` (``device_funcs.cu:L2490-2503``), where
    ``n_links`` is the number of grid points along the ballooning chain (not ``2*pi`` segments).
    """
    return jnp.asarray(
        _kz_linked_ntft_np(nz, zp, n_links, dealias_kz=dealias_kz), dtype=jnp.float64
    )


def init_kzm_linked(fft_len: int) -> jax.Array:
    """Compute 2/3 parallel dealiasing mask on a chain of length ``fft_len``.

    Matches the ``dealias_kz`` filter in ``init_kzLinked`` / ``init_kzLinkedNTFT``
    (``device_funcs.cu:L2484-2486, L2499-2501``) and ``kInit`` (``grids.py:L260-261``):
    zeros modes ``i`` satisfying ``(fft_len - 1) // 3 < i < fft_len - (fft_len - 1) // 3``.
    """
    return jnp.asarray(_kzm_linked_np(fft_len), dtype=jnp.int32)


def _compute_hyperz_weight(kz: jax.Array | np.ndarray, p_hyper_z: int) -> jax.Array:
    """Compute ``(|kz| / kz_max)^p_hyper_z`` matching ``hyperkzLinked`` / ``hyperkzLinkedNTFT``.

    Ported from ``device_funcs.cu:L2414-2454``: ``kz_max`` is ``kz[nzL // 2]`` for ``nzL > 1``
    (which equals ``grids.kz_max = (nz // 2) / zp`` for all even-``nz`` linked chains), and
    safely returns ``0.0`` when ``nzL <= 1`` so single-point edge NTFT chains never divide ``0/0``.
    """
    return jnp.asarray(
        _hyperz_weight_np(np.asarray(kz, dtype=np.float64), p_hyper_z), dtype=jnp.float64
    )


def calc_idx0_ntft(idx: int, nshift: int, nakx: int) -> tuple[int, int]:
    """Convert sequential NTFT ``idx`` in ``[0..nakx-1]`` to ``(idx0_active, idx0_full)``.

    Ported strictly from ``GradParallelNTFT::calc_idx0`` (``grad_parallel_NTFT.cu:L624-647``):
    - ``idx`` indexes ``kx`` sequentially from negative to positive ``[-K, ..., 0, ..., +K]``
      (``K = nakx // 2``).
    - ``idx0_active`` is Option A's FFT-active index in ``[0..nakx-1]``
      (``[0, +1, ..., +K, -K, ..., -1]``).
    - ``idx0_full`` is C++'s full padded FFT index in ``[0..nx-1]`` (``nshift = nx - nakx``).
    """
    half_floor = nakx // 2
    if idx < half_floor:
        idx0_active = idx + half_floor + 1
    else:
        idx0_active = idx - half_floor

    if idx0_active < (nakx + 1) // 2:
        idx0_full = idx0_active
    else:
        idx0_full = idx0_active + nshift
    return int(idx0_active), int(idx0_full)


def build_linked_topology(grids: Grids, jtwist: int | None = None) -> LinkedChainTopology:
    """Build exact ``GradParallelLinked`` chain topology (``grad_parallel_linked.cu:L521-725``)."""
    naky = int(grids.naky)
    nakx = int(grids.nakx)
    nx = int(grids.nx)
    nshift = nx - nakx
    jshift0 = int(grids.jtwist if jtwist is None else jtwist)
    if jshift0 == 0:
        raise ValueError("jtwist = 0 is not allowed in GX.")

    half = (nakx + 1) // 2
    idx_left = np.full((naky, nakx), -1, dtype=np.int32)
    idx_right = np.full((naky, nakx), -1, dtype=np.int32)

    # get_nClasses step 1: left/right neighbor active indices (L527-563)
    for idx in range(nakx):
        idx0 = idx if idx < half else (idx - nakx)
        for idy in range(naky):
            if idy == 0:
                idx_l = idx0
                idx_r = idx0
            else:
                idx_l = idx0 + idy * jshift0
                idx_r = idx0 - idy * jshift0

            if 0 <= idx_l < half:
                idx_left[idy, idx] = idx_l
            elif half <= idx_l + nakx < nakx:
                idx_left[idy, idx] = idx_l + nakx
            else:
                idx_left[idy, idx] = -1

            if 0 <= idx_r < half:
                idx_right[idy, idx] = idx_r
            elif half <= idx_r + nakx < nakx:
                idx_right[idy, idx] = idx_r + nakx
            else:
                idx_right[idy, idx] = -1

    # get_nClasses step 2: count linksL and linksR (L566-592)
    links_l = np.zeros((naky, nakx), dtype=np.int32)
    links_r = np.zeros((naky, nakx), dtype=np.int32)
    for idx in range(nakx):
        for idy in range(naky):
            idx_star = idx
            while idx_left[idy, idx_star] >= 0 and idx_star != idx_left[idy, idx_star]:
                links_l[idy, idx] += 1
                idx_star = int(idx_left[idy, idx_star])

            idx_star = idx
            while idx_right[idy, idx_star] >= 0 and idx_star != idx_right[idy, idx_star]:
                links_r[idy, idx] += 1
                idx_star = int(idx_right[idy, idx_star])

    # get_nClasses step 3 & get_nLinks_nChains (L599-645)
    n_k_grid = 1 + links_l + links_r  # (naky, nakx)
    unique_links, counts = np.unique(n_k_grid, return_counts=True)
    n_classes = int(unique_links.size)
    n_links = tuple(int(v) for v in unique_links)
    n_chains = tuple(int(cnt // lnk) for cnt, lnk in zip(counts, unique_links, strict=True))
    link_to_class = {lnk: ic for ic, lnk in enumerate(n_links)}

    # kFill (L691-725): iterate idy outer, idx inner for each class ic when p == 0
    iky_chains_list: list[np.ndarray] = []
    ikx_chains_list: list[np.ndarray] = []
    ikx_full_chains_list: list[np.ndarray] = []

    p_map = links_l.copy()
    n_map = np.zeros((naky, nakx), dtype=np.int32)
    c_map = np.zeros((naky, nakx), dtype=np.int32)
    n_links_map = np.zeros((naky, nakx), dtype=np.int32)
    n_chains_map = np.zeros((naky, nakx), dtype=np.int32)

    for ic in range(n_classes):
        lnk = n_links[ic]
        chn = n_chains[ic]
        ky_arr = np.zeros((chn, lnk), dtype=np.int32)
        kx_arr = np.zeros((chn, lnk), dtype=np.int32)
        kx_full_arr = np.zeros((chn, lnk), dtype=np.int32)

        n_idx = 0
        for idy in range(naky):
            for idx in range(nakx):
                c_here = link_to_class[int(n_k_grid[idy, idx])]
                p_here = int(links_l[idy, idx])
                if c_here == ic and p_here == 0:
                    idx_cur = idx
                    for p_step in range(lnk):
                        if p_step > 0:
                            idx_cur = int(idx_right[idy, idx_cur])
                        ky_arr[n_idx, p_step] = idy
                        kx_arr[n_idx, p_step] = idx_cur
                        kx_full_arr[n_idx, p_step] = (
                            idx_cur if idx_cur < half else (idx_cur + nshift)
                        )
                        c_map[idy, idx_cur] = ic
                        n_map[idy, idx_cur] = n_idx
                        n_links_map[idy, idx_cur] = lnk
                        n_chains_map[idy, idx_cur] = chn
                    n_idx += 1

        if n_idx != chn:
            raise RuntimeError(f"Expected {chn} chains in class {ic}, filled {n_idx}.")
        iky_chains_list.append(ky_arr)
        ikx_chains_list.append(kx_arr)
        ikx_full_chains_list.append(kx_full_arr)

    return LinkedChainTopology(
        n_classes=n_classes,
        n_links=n_links,
        n_chains=n_chains,
        iky_chains=tuple(iky_chains_list),
        ikx_chains=tuple(ikx_chains_list),
        ikx_full_chains=tuple(ikx_full_chains_list),
        idx_left=idx_left,
        idx_right=idx_right,
        links_l=links_l,
        links_r=links_r,
        p_map=p_map,
        n_map=n_map,
        c_map=c_map,
        n_links_map=n_links_map,
        n_chains_map=n_chains_map,
    )


def _ntft_find_chain_start(
    idy: int,
    idx: int,
    idz: int,
    nz: int,
    nakx: int,
    jtwist: int,
    m0_zy: np.ndarray,
) -> tuple[int, int]:
    """Replicate the ``while`` backward walk in ``get_mode_nums_ntft`` / ``kFill_ntft``.

    Matches ``grad_parallel_NTFT.cu:L454-463, L582-591`` identically.
    """
    idz_start = int(idz)
    idx_constant = int(idx) + int(m0_zy[idz, idy])
    while True:
        idz_prev = (idz_start - 1 + nz) % nz
        wrap = math.floor((idz_start - 1) / float(nz))
        candidate = idx_constant - int(m0_zy[idz_prev, idy]) - int(wrap) * jtwist * idy
        if 0 <= candidate < nakx:
            if idz_start == 0:
                idx_constant = idx_constant + jtwist * idy
                idz_start = nz - 1
            else:
                idz_start -= 1
        else:
            break
    return idz_start, idx_constant


def build_ntft_topology(
    grids: Grids,
    m0: jax.Array | np.ndarray,
    jtwist: int | None = None,
    *,
    n_links_max: int = 3000,
    n_links_min: int = 0,
) -> NTFTChainTopology:
    """Build exact ``GradParallelNTFT`` ballooning chain topology (``L431-647``)."""
    nz = int(grids.nz)
    naky = int(grids.naky)
    nakx = int(grids.nakx)
    nx = int(grids.nx)
    nshift = nx - nakx
    jtwist_val = int(grids.jtwist if jtwist is None else jtwist)
    if jtwist_val == 0:
        raise ValueError("jtwist = 0 is not allowed in GX.")

    m0_zy = np.asarray(m0, dtype=np.int32)
    if m0_zy.shape != (nz, naky):
        raise ValueError(f"Expected geo.m0 of shape {(nz, naky)}, got {m0_zy.shape}.")
    ky_np = np.asarray(grids.ky, dtype=np.float64)

    # 1. get_mode_nums_ntft (L431-483)
    # mode_nums_seq[idz, idx_seq, idy] matches C++ mode_nums[idy + naky * (idx + nakx * idz)]
    mode_nums_seq = np.zeros((nz, nakx, naky), dtype=np.int32)
    mode_starts: list[tuple[int, int, int]] = []  # (idy, idz_start, idx_constant_or_seq_idx)
    mode = 0

    for idy in range(naky):
        if ky_np[idy] < 1.0e-10:
            for idx in range(nakx):
                mode += 1
                mode_nums_seq[:, idx, idy] = mode
                mode_starts.append((idy, 0, idx))
        else:
            for idx in range(nakx):
                for idz in range(nz):
                    if mode_nums_seq[idz, idx, idy] == 0:
                        idz_start, idx_constant = _ntft_find_chain_start(
                            idy, idx, idz, nz, nakx, jtwist_val, m0_zy
                        )
                        mode += 1
                        mode_starts.append((idy, idz_start, idx_constant))
                        idz_prime = idz_start
                        idx_const_cur = idx_constant
                        while 0 <= idx_const_cur - int(m0_zy[idz_prime, idy]) < nakx:
                            idx_prime = idx_const_cur - int(m0_zy[idz_prime, idy])
                            mode_nums_seq[idz_prime, idx_prime, idy] = mode
                            if idz_prime == nz - 1:
                                idx_const_cur -= jtwist_val * idy
                                idz_prime = 0
                            else:
                                idz_prime += 1

    # Convert mode_nums_seq (nz, nakx_seq, naky) -> Option A mode_nums_active (nz, naky, nakx)
    seq_to_active = np.array(
        [calc_idx0_ntft(idx_s, nshift, nakx)[0] for idx_s in range(nakx)], dtype=np.int32
    )
    mode_nums_active = np.zeros((nz, naky, nakx), dtype=np.int32)
    for idx_s in range(nakx):
        mode_nums_active[:, :, seq_to_active[idx_s]] = mode_nums_seq[:, idx_s, :]

    # 2. get_nClasses_ntft & get_nChains_nLinks_ntft (L484-544)
    flat_modes = mode_nums_seq.ravel()
    counts_1based = np.bincount(flat_modes, minlength=mode + 1)
    mode_size_raw = counts_1based[1 : mode + 1].astype(np.int32)
    n_extra = np.where(mode_size_raw > n_links_max, mode_size_raw % nz, 0).astype(np.int32)
    mode_size_ref = (mode_size_raw - n_extra).astype(np.int32)

    valid_sizes = mode_size_ref[mode_size_ref > n_links_min]
    unique_sizes, class_counts = np.unique(valid_sizes, return_counts=True)
    n_classes = int(unique_sizes.size)
    n_links = tuple(int(v) for v in unique_sizes)
    n_chains = tuple(int(c) for c in class_counts)

    # 3. kFill_ntft (L546-622)
    idz_chains_list: list[np.ndarray] = []
    iky_chains_list: list[np.ndarray] = []
    ikx_chains_list: list[np.ndarray] = []
    ikxdz_full_chains_list: list[np.ndarray] = []

    for ic in range(n_classes):
        lnk = n_links[ic]
        chn = n_chains[ic]
        idz_arr = np.zeros((chn, lnk), dtype=np.int32)
        iky_arr = np.zeros((chn, lnk), dtype=np.int32)
        ikx_arr = np.zeros((chn, lnk), dtype=np.int32)
        ikxdz_full_arr = np.zeros((chn, lnk), dtype=np.int32)

        n_idx = 0
        for i_mode in range(mode):
            if int(mode_size_ref[i_mode]) != lnk:
                continue
            idy, idz_start, idx_info = mode_starts[i_mode]
            if ky_np[idy] < 1.0e-10:
                idx0_act, idx0_full = calc_idx0_ntft(idx_info, nshift, nakx)
                idz_seq = np.arange(nz, dtype=np.int32)[:lnk]
                idz_arr[n_idx, :] = idz_seq
                iky_arr[n_idx, :] = idy
                ikx_arr[n_idx, :] = idx0_act
                ikxdz_full_arr[n_idx, :] = idx0_full + nx * idz_seq
            else:
                extra_skip = int(n_extra[i_mode]) // 2
                extra_counter = 0
                p_step = 0
                idz_prime = idz_start
                idx_const_cur = idx_info
                while 0 <= idx_const_cur - int(m0_zy[idz_prime, idy]) < nakx and p_step < lnk:
                    if extra_counter < extra_skip:
                        extra_counter += 1
                    else:
                        idx_prime = idx_const_cur - int(m0_zy[idz_prime, idy])
                        idx0_act, idx0_full = calc_idx0_ntft(idx_prime, nshift, nakx)
                        idz_arr[n_idx, p_step] = idz_prime
                        iky_arr[n_idx, p_step] = idy
                        ikx_arr[n_idx, p_step] = idx0_act
                        ikxdz_full_arr[n_idx, p_step] = idx0_full + nx * idz_prime
                        p_step += 1

                    if idz_prime == nz - 1:
                        idx_const_cur -= jtwist_val * idy
                        idz_prime = 0
                    else:
                        idz_prime += 1

                if p_step != lnk:
                    raise RuntimeError(
                        f"NTFT mode {i_mode + 1} filled {p_step} points, expected {lnk}."
                    )
            n_idx += 1

        if n_idx != chn:
            raise RuntimeError(f"NTFT class {ic} filled {n_idx} chains, expected {chn}.")
        idz_chains_list.append(idz_arr)
        iky_chains_list.append(iky_arr)
        ikx_chains_list.append(ikx_arr)
        ikxdz_full_chains_list.append(ikxdz_full_arr)

    return NTFTChainTopology(
        mode_count=mode,
        mode_nums_seq=mode_nums_seq,
        mode_nums_active=mode_nums_active,
        mode_size_raw=mode_size_raw,
        mode_size_ref=mode_size_ref,
        n_extra=n_extra,
        n_classes=n_classes,
        n_links=n_links,
        n_chains=n_chains,
        idz_chains=tuple(idz_chains_list),
        iky_chains=tuple(iky_chains_list),
        ikx_chains=tuple(ikx_chains_list),
        ikxdz_full_chains=tuple(ikxdz_full_chains_list),
    )


def _cast_float_leaf(x: jax.Array, r_dtype: jnp.dtype) -> jax.Array:
    if not (jnp.issubdtype(x.dtype, jnp.floating) or jnp.issubdtype(x.dtype, jnp.complexfloating)):
        return x
    c_dtype = jnp.complex64 if r_dtype == jnp.float32 else jnp.complex128
    target = c_dtype if jnp.issubdtype(x.dtype, jnp.complexfloating) else r_dtype
    if isinstance(x, jax.core.Tracer):
        return x.astype(target)
    return jnp.asarray(np.asarray(x), dtype=target)


def _pad_chain_count(n_chains: int, multiple: int) -> int:
    if multiple < 1:
        raise ValueError(f"chain_pad_multiple={multiple} must be >= 1.")
    if n_chains <= 0:
        return 0
    return ((n_chains + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Static-Shape Linked Chain Bucket & LinkedParallelGrad Operator
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinkedBucket:
    """Static-shape bucket of ``n_chains`` linked chains with FFT length ``nz * n_links``."""

    yx_idx: jax.Array  # (n_chains_padded, n_links) int32 into flattened (naky * nakx)
    chain_mask: jax.Array  # (n_chains_padded,) bool
    kz: jax.Array  # (fft_len,) float64/float32
    kzm: jax.Array  # (fft_len,) int32
    hyperz_weight: jax.Array  # (fft_len,) float64/float32

    n_links: int
    n_chains: int
    n_chains_padded: int
    fft_len: int
    mxu_plan: MXU1DPlan | None = None

    def tree_flatten(self):
        children = (
            self.yx_idx,
            self.chain_mask,
            self.kz,
            self.kzm,
            self.hyperz_weight,
            self.mxu_plan,
        )
        aux_data = (self.n_links, self.n_chains, self.n_chains_padded, self.fft_len)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> LinkedBucket:
        n_links, n_chains, n_chains_padded, fft_len = aux_data
        yx_idx, chain_mask, kz, kzm, hyperz_weight, mxu_plan = children
        return cls(
            yx_idx=yx_idx,
            chain_mask=chain_mask,
            kz=kz,
            kzm=kzm,
            hyperz_weight=hyperz_weight,
            n_links=n_links,
            n_chains=n_chains,
            n_chains_padded=n_chains_padded,
            fft_len=fft_len,
            mxu_plan=mxu_plan,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinkedParallelGrad:
    """Bucketed multi-link twist-and-shift parallel operator (``GradParallelLinked``).

    Ported strictly from ``gx/src/grad_parallel_linked.cu`` and
    ``gx/src/device_funcs.cu`` (``L2350-2876``).
    Supports arbitrary leading batch axes ``(*batch, nz, naky, nakx)`` (3D fields, 4D/5D
    closures, and 6D ``G``) with zero Python per-chain loops and zero XLA scatter ops.
    """

    buckets: tuple[LinkedBucket, ...]
    inv_yx_perm: jax.Array  # (naky, nakx) int32 inverse bijection (linkedCopyBackAll)
    end_damp_nu: jax.Array  # (nz, naky, nakx) float64/float32 dampEnds_linked profile
    active_mask: jax.Array  # (naky, nakx) bool
    p_map: jax.Array  # (naky, nakx) int32
    n_map: jax.Array  # (naky, nakx) int32
    c_map: jax.Array  # (naky, nakx) int32
    n_links_map: jax.Array  # (naky, nakx) int32
    n_chains_map: jax.Array  # (naky, nakx) int32

    nz: int
    naky: int
    nakx: int
    zp: int
    jtwist: int
    n_classes: int
    n_links: tuple[int, ...]
    n_chains: tuple[int, ...]
    chain_pad_multiple: int
    local_limit: bool
    boundary_option_periodic: bool
    dealias_kz: bool
    damp_ends_widthfrac: float
    damp_ends_amp: float
    p_hyper_z: int
    fft_backend: str = "fft"

    @property
    def supports_fused_z_fft(self) -> bool:
        """False because linked chains couple distinct ``kx`` segments across ``z = +-pi*zp``."""
        return False

    def tree_flatten(self):
        children = (
            self.buckets,
            self.inv_yx_perm,
            self.end_damp_nu,
            self.active_mask,
            self.p_map,
            self.n_map,
            self.c_map,
            self.n_links_map,
            self.n_chains_map,
        )
        aux_data = (
            self.nz,
            self.naky,
            self.nakx,
            self.zp,
            self.jtwist,
            self.n_classes,
            self.n_links,
            self.n_chains,
            self.chain_pad_multiple,
            self.local_limit,
            self.boundary_option_periodic,
            self.dealias_kz,
            self.damp_ends_widthfrac,
            self.damp_ends_amp,
            self.p_hyper_z,
            self.fft_backend,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> LinkedParallelGrad:
        (
            nz,
            naky,
            nakx,
            zp,
            jtwist,
            n_classes,
            n_links,
            n_chains,
            chain_pad_multiple,
            local_limit,
            boundary_option_periodic,
            dealias_kz,
            damp_ends_widthfrac,
            damp_ends_amp,
            p_hyper_z,
            fft_backend,
        ) = aux_data
        (
            buckets,
            inv_yx_perm,
            end_damp_nu,
            active_mask,
            p_map,
            n_map,
            c_map,
            n_links_map,
            n_chains_map,
        ) = children
        return cls(
            buckets=buckets,
            inv_yx_perm=inv_yx_perm,
            end_damp_nu=end_damp_nu,
            active_mask=active_mask,
            p_map=p_map,
            n_map=n_map,
            c_map=c_map,
            n_links_map=n_links_map,
            n_chains_map=n_chains_map,
            nz=nz,
            naky=naky,
            nakx=nakx,
            zp=zp,
            jtwist=jtwist,
            n_classes=n_classes,
            n_links=n_links,
            n_chains=n_chains,
            chain_pad_multiple=chain_pad_multiple,
            local_limit=local_limit,
            boundary_option_periodic=boundary_option_periodic,
            dealias_kz=dealias_kz,
            damp_ends_widthfrac=damp_ends_widthfrac,
            damp_ends_amp=damp_ends_amp,
            p_hyper_z=p_hyper_z,
            fft_backend=fft_backend,
        )

    def astype(self, dtype: jnp.dtype) -> LinkedParallelGrad:
        """Cast floating-point leaves (including bucket MXU plans) to ``dtype``."""
        target = jnp.dtype(dtype)
        r_dtype = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        return jax.tree_util.tree_map(lambda x: _cast_float_leaf(x, r_dtype), self)

    def _check_shape(self, arr: jax.Array) -> None:
        if arr.ndim < 3 or arr.shape[-3:] != (self.nz, self.naky, self.nakx):
            raise ValueError(
                f"Expected trailing shape {(self.nz, self.naky, self.nakx)}, got {arr.shape}."
            )

    def _mask_out(self, arr: jax.Array) -> jax.Array:
        return jnp.where(self.active_mask, arr, jnp.zeros_like(arr))

    def _transform_buckets(
        self,
        arr: jax.Array,
        kind: str,
    ) -> jax.Array:
        """Gather chains per bucket, apply 1D chain operator, and inverse-permute back."""
        self._check_shape(arr)
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = self._mask_out(arr.astype(c_dtype))
        batch_shape = arr_c.shape[:-3]
        n_moms = math.prod(batch_shape) if batch_shape else 1
        arr_yx = arr_c.reshape(n_moms, self.nz, self.naky * self.nakx)

        use_mxu = self.fft_backend == "mxu"
        seg_outs: list[jax.Array] = []
        for b in self.buckets:
            # Gather (n_moms, nz, C_pad, P) -> move nz to last -> (n_moms, C_pad, P * nz)
            gathered = arr_yx[:, :, b.yx_idx]
            chains = jnp.moveaxis(gathered, 1, -1).reshape(n_moms, b.n_chains_padded, b.fft_len)
            if b.n_chains_padded > b.n_chains:
                chains = jnp.where(b.chain_mask[:, None], chains, jnp.zeros_like(chains))

            def _fwd_1d(x_in: jax.Array, *, _b: LinkedBucket = b) -> jax.Array:
                if use_mxu:
                    return mxu_fft(x_in, n=_b.fft_len, axis=-1, plan=_b.mxu_plan)
                return jnp.fft.fft(x_in, axis=-1)

            def _inv_1d(x_hat: jax.Array, *, _b: LinkedBucket = b) -> jax.Array:
                if use_mxu:
                    return mxu_ifft(x_hat, n=_b.fft_len, axis=-1, plan=_b.mxu_plan).astype(c_dtype)
                return jnp.fft.ifft(x_hat, axis=-1).astype(c_dtype)

            if kind == "identity":
                chains_out = chains
            elif kind == "zft":
                if use_mxu:
                    chains_out = mxu_fft(
                        chains, n=b.fft_len, axis=-1, norm="forward", plan=b.mxu_plan
                    ).astype(c_dtype)
                else:
                    norm = jnp.asarray(1.0 / float(b.fft_len), dtype=r_dtype)
                    chains_out = jnp.fft.fft(chains, axis=-1).astype(c_dtype) * norm
            elif kind == "zft_inverse":
                if use_mxu:
                    chains_out = mxu_ifft(
                        chains, n=b.fft_len, axis=-1, norm="forward", plan=b.mxu_plan
                    ).astype(c_dtype)
                else:
                    scale = jnp.asarray(float(b.fft_len), dtype=r_dtype)
                    chains_out = jnp.fft.ifft(chains, axis=-1).astype(c_dtype) * scale
            elif kind == "dz":
                kz_r = b.kz.astype(r_dtype)
                mult = jax.lax.complex(jnp.zeros_like(kz_r), kz_r).astype(c_dtype)
                chains_out = _inv_1d(_fwd_1d(chains) * mult)
            elif kind == "abs_dz":
                mult_r = jnp.abs(b.kz.astype(r_dtype))
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            elif kind == "dz2":
                kz_r = b.kz.astype(r_dtype)
                mult_r = -(kz_r * kz_r)
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            elif kind == "hyperz":
                mult_r = -b.hyperz_weight.astype(r_dtype)
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            elif kind == "dealias_chain":
                mult_r = b.kzm.astype(r_dtype)
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            else:  # pragma: no cover
                raise ValueError(f"Unknown bucket operator kind={kind!r}.")

            active = chains_out[:, : b.n_chains, :].reshape(n_moms, b.n_chains * b.n_links, self.nz)
            seg_outs.append(jnp.swapaxes(active, -2, -1))

        all_yx = jnp.concatenate(seg_outs, axis=-1)
        res = all_yx[:, :, self.inv_yx_perm].reshape(*batch_shape, self.nz, self.naky, self.nakx)
        return self._mask_out(res)

    def identity(self, arr: jax.Array) -> jax.Array:
        """Linked gather + scatter round-trip (``GradParallelLinked::identity``, ``L339-345``)."""
        return self._transform_buckets(arr, "identity")

    def dz(
        self, arr: jax.Array, *, accumulate: bool = False, target: jax.Array | None = None
    ) -> jax.Array:
        """Linked parallel derivative ``d/dz`` (``GradParallelLinked::dz``, ``L385-403``)."""
        c_dtype, _ = _complex_and_real_dtypes(arr)
        res = self._transform_buckets(arr, "dz")
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def abs_dz(
        self, arr: jax.Array, *, accumulate: bool = False, target: jax.Array | None = None
    ) -> jax.Array:
        """Linked Landau operator ``|d/dz|`` (``GradParallelLinked::abs_dz``, ``L461-478``)."""
        c_dtype, _ = _complex_and_real_dtypes(arr)
        res = self._transform_buckets(arr, "abs_dz")
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def dz2(self, arr: jax.Array) -> jax.Array:
        """Linked second parallel derivative ``d^2/dz^2`` (``GradParallelLinked::dz2``)."""
        return self._transform_buckets(arr, "dz2")

    def hyperz(
        self,
        arr: jax.Array,
        nu_hyper_z: float,
        *,
        accumulate: bool = False,
        target: jax.Array | None = None,
    ) -> jax.Array:
        """Linked parallel hyperviscosity ``-nu_hyper_z * (|kz| / kz_max)^p_hyper_z``."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        res = jnp.asarray(nu_hyper_z, dtype=r_dtype) * self._transform_buckets(arr, "hyperz")
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def zft(self, arr: jax.Array) -> jax.Array:
        """Normalized forward linked chain FFT (``GradParallelLinked::zft``, ``L347-355``)."""
        return self._transform_buckets(arr, "zft")

    def zft_inverse(self, arr: jax.Array) -> jax.Array:
        """Unnormalized inverse linked chain FFT (``GradParallelLinked::zft_inverse``)."""
        return self._transform_buckets(arr, "zft_inverse")

    def dealias(self, arr: jax.Array) -> jax.Array:
        """Apply periodic 2/3 ``kzm`` mask if ``boundary_option_periodic``, else C++ no-op.

        Matches C++ timestepper ``dealias_kz`` dispatch (``ts_sspx3.cu:L38-46``,
        ``ts_k10.cu:L31-32``): ``GradParallelPeriodic::dealias`` (``grad_parallel.cu:L83-97``)
        applies ``kzm`` when ``boundary_option_periodic=True``, whereas
        ``GradParallelLinked::dealias`` (``grad_parallel_linked.cu:L328-336``) is a no-op.
        """
        self._check_shape(arr)
        c_dtype, _ = _complex_and_real_dtypes(arr)
        if self.local_limit or self.nz <= 1 or not self.boundary_option_periodic:
            return self._mask_out(arr.astype(c_dtype))
        return self.dealias_chain(arr)

    def dealias_chain(self, arr: jax.Array) -> jax.Array:
        """Opt-in helper applying the 2/3 parallel ``kz`` mask ``kzm`` on each linked chain."""
        return self._transform_buckets(arr, "dealias_chain")

    def apply_bcs(
        self,
        G: jax.Array,
        rhs: jax.Array,
        fields: Fields,
        solver: FieldSolver,
        dt: float | jax.Array,
        *,
        H: jax.Array | None = None,
    ) -> jax.Array:
        """Apply linked extended-chain end damping on ``H`` (``dampEnds_linked``, ``L2813-2876``).

        Damps only the physical ends ``idzp <= width`` and ``idzp >= nz * nLinks - width`` of
        each extended ballooning chain (never internal ``z = +-pi*zp`` link junctions) and
        exclusively on non-zonal ``iky > 0`` modes.
        """
        c_dtype, r_dtype = _complex_and_real_dtypes(G)
        if self.boundary_option_periodic or self.local_limit:
            return rhs.astype(c_dtype)
        if H is None:
            H = build_nonadiabatic_h(G, fields, solver)
        h_c = H.astype(c_dtype)
        amp = jnp.asarray(self.damp_ends_amp, dtype=r_dtype) / jnp.asarray(dt, dtype=r_dtype)
        nu6 = self.end_damp_nu.astype(r_dtype)[None, None, None, :, :, :]
        mask6 = self.active_mask[None, None, None, None, :, :]
        damp_term = jnp.where(mask6, (nu6 * amp) * h_c, jnp.zeros_like(h_c))
        return rhs.astype(c_dtype) - damp_term

    def coupled_ikx_indices(self, iky: int, ikx: int) -> tuple[int, ...]:
        """Return active ``ikx`` indices forming the linked chain containing ``(iky, ikx)``."""
        if not (0 <= iky < self.naky and 0 <= ikx < self.nakx):
            raise ValueError(f"(iky={iky}, ikx={ikx}) out of bounds for {(self.naky, self.nakx)}.")
        c_idx = int(np.asarray(self.c_map)[iky, ikx])
        n_idx = int(np.asarray(self.n_map)[iky, ikx])
        b = self.buckets[c_idx]
        yx_row = np.asarray(b.yx_idx)[n_idx]
        return tuple(int(v % self.nakx) for v in yx_row)

    def format_links(self) -> str:
        """Return human-readable chain table mirroring ``GradParallelLinked::linkPrint``."""
        half = (self.nakx + 1) // 2
        lines: list[str] = []
        for ic, b in enumerate(self.buckets):
            lines.append(f"Class {ic} (nLinks={b.n_links}, nChains={b.n_chains}):")
            yx_np = np.asarray(b.yx_idx)[: b.n_chains]
            for row in yx_np:
                pairs = []
                for yx in row:
                    idy = int(yx) // self.nakx
                    idx = int(yx) % self.nakx
                    signed_kx = idx if idx < half else (idx - self.nakx)
                    pairs.append(f"({idy},{signed_kx})")
                lines.append("  " + " ".join(pairs))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Static-Shape NTFT Chain Bucket & NTFTParallelGrad Operator
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NTFTBucket:
    """Static-shape bucket of ``n_chains`` NTFT ballooning chains of ``n_links`` grid points."""

    zyx_idx: jax.Array  # (n_chains_padded, n_links) int32 into flattened (nz * naky * nakx)
    chain_mask: jax.Array  # (n_chains_padded,) bool
    kz: jax.Array  # (n_links,) float64/float32
    kzm: jax.Array  # (n_links,) int32
    hyperz_weight: jax.Array  # (n_links,) float64/float32

    n_links: int
    n_chains: int
    n_chains_padded: int
    fft_len: int
    mxu_plan: MXU1DPlan | None = None

    def tree_flatten(self):
        children = (
            self.zyx_idx,
            self.chain_mask,
            self.kz,
            self.kzm,
            self.hyperz_weight,
            self.mxu_plan,
        )
        aux_data = (self.n_links, self.n_chains, self.n_chains_padded, self.fft_len)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> NTFTBucket:
        n_links, n_chains, n_chains_padded, fft_len = aux_data
        zyx_idx, chain_mask, kz, kzm, hyperz_weight, mxu_plan = children
        return cls(
            zyx_idx=zyx_idx,
            chain_mask=chain_mask,
            kz=kz,
            kzm=kzm,
            hyperz_weight=hyperz_weight,
            n_links=n_links,
            n_chains=n_chains,
            n_chains_padded=n_chains_padded,
            fft_len=fft_len,
            mxu_plan=mxu_plan,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NTFTParallelGrad:
    """Bucketed Non-Twisting Flux-Tube parallel operator (``GradParallelNTFT``).

    Ported strictly from ``gx/src/grad_parallel_NTFT.cu`` and
    ``gx/src/device_funcs.cu`` (``L2365-2545, L2739-2811, L2880-2945``).
    In NTFT (Ball et al. 2020), ``n_links`` is the exact number of ``z`` grid points along each
    variable-length ballooning chain connected via discrete mode shift ``geo.m0(z, ky)``.
    """

    buckets: tuple[NTFTBucket, ...]
    inv_zyx_map: jax.Array  # (nz, naky, nakx) int32 (0 = masked/trimmed, >=1 = flat chain slot)
    end_damp_nu: jax.Array  # (nz, naky, nakx) float64/float32 dampEnds_linkedNTFT profile
    active_mask: jax.Array  # (naky, nakx) bool
    mode_nums: jax.Array  # (nz, naky, nakx) int32 Option A active ikx order
    n_extra: jax.Array  # (mode_count,) int32

    nz: int
    naky: int
    nakx: int
    zp: int
    jtwist: int
    mode_count: int
    n_classes: int
    n_links: tuple[int, ...]
    n_chains: tuple[int, ...]
    n_links_max: int
    chain_pad_multiple: int
    local_limit: bool
    boundary_option_periodic: bool
    dealias_kz: bool
    damp_ends_widthfrac: float
    damp_ends_amp: float
    p_hyper_z: int
    fft_backend: str = "fft"

    @property
    def supports_fused_z_fft(self) -> bool:
        """False because NTFT chains follow ``m0(z, ky)`` shifts across ``(z, kx)``."""
        return False

    def tree_flatten(self):
        children = (
            self.buckets,
            self.inv_zyx_map,
            self.end_damp_nu,
            self.active_mask,
            self.mode_nums,
            self.n_extra,
        )
        aux_data = (
            self.nz,
            self.naky,
            self.nakx,
            self.zp,
            self.jtwist,
            self.mode_count,
            self.n_classes,
            self.n_links,
            self.n_chains,
            self.n_links_max,
            self.chain_pad_multiple,
            self.local_limit,
            self.boundary_option_periodic,
            self.dealias_kz,
            self.damp_ends_widthfrac,
            self.damp_ends_amp,
            self.p_hyper_z,
            self.fft_backend,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> NTFTParallelGrad:
        (
            nz,
            naky,
            nakx,
            zp,
            jtwist,
            mode_count,
            n_classes,
            n_links,
            n_chains,
            n_links_max,
            chain_pad_multiple,
            local_limit,
            boundary_option_periodic,
            dealias_kz,
            damp_ends_widthfrac,
            damp_ends_amp,
            p_hyper_z,
            fft_backend,
        ) = aux_data
        buckets, inv_zyx_map, end_damp_nu, active_mask, mode_nums, n_extra = children
        return cls(
            buckets=buckets,
            inv_zyx_map=inv_zyx_map,
            end_damp_nu=end_damp_nu,
            active_mask=active_mask,
            mode_nums=mode_nums,
            n_extra=n_extra,
            nz=nz,
            naky=naky,
            nakx=nakx,
            zp=zp,
            jtwist=jtwist,
            mode_count=mode_count,
            n_classes=n_classes,
            n_links=n_links,
            n_chains=n_chains,
            n_links_max=n_links_max,
            chain_pad_multiple=chain_pad_multiple,
            local_limit=local_limit,
            boundary_option_periodic=boundary_option_periodic,
            dealias_kz=dealias_kz,
            damp_ends_widthfrac=damp_ends_widthfrac,
            damp_ends_amp=damp_ends_amp,
            p_hyper_z=p_hyper_z,
            fft_backend=fft_backend,
        )

    def astype(self, dtype: jnp.dtype) -> NTFTParallelGrad:
        """Cast floating-point leaves (including bucket MXU plans) to ``dtype``."""
        target = jnp.dtype(dtype)
        r_dtype = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        return jax.tree_util.tree_map(lambda x: _cast_float_leaf(x, r_dtype), self)

    def _check_shape(self, arr: jax.Array) -> None:
        if arr.ndim < 3 or arr.shape[-3:] != (self.nz, self.naky, self.nakx):
            raise ValueError(
                f"Expected trailing shape {(self.nz, self.naky, self.nakx)}, got {arr.shape}."
            )

    def _mask_out(self, arr: jax.Array) -> jax.Array:
        return jnp.where(self.active_mask, arr, jnp.zeros_like(arr))

    def _transform_buckets(self, arr: jax.Array, kind: str) -> jax.Array:
        """Gather NTFT chains directly along ``axis=-1``, apply 1D FFT op, and map back."""
        self._check_shape(arr)
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = self._mask_out(arr.astype(c_dtype))
        batch_shape = arr_c.shape[:-3]
        n_moms = math.prod(batch_shape) if batch_shape else 1
        arr_zyx = arr_c.reshape(n_moms, self.nz * self.naky * self.nakx)

        use_mxu = self.fft_backend == "mxu"
        flat_outs: list[jax.Array] = [jnp.zeros((n_moms, 1), dtype=c_dtype)]
        for b in self.buckets:
            chains = arr_zyx[:, b.zyx_idx]  # (n_moms, C_pad, L) with zero transposes!
            if b.n_chains_padded > b.n_chains:
                chains = jnp.where(b.chain_mask[:, None], chains, jnp.zeros_like(chains))

            def _fwd_1d(x_in: jax.Array, *, _b: NTFTBucket = b) -> jax.Array:
                if use_mxu:
                    return mxu_fft(x_in, n=_b.fft_len, axis=-1, plan=_b.mxu_plan)
                return jnp.fft.fft(x_in, axis=-1)

            def _inv_1d(x_hat: jax.Array, *, _b: NTFTBucket = b) -> jax.Array:
                if use_mxu:
                    return mxu_ifft(x_hat, n=_b.fft_len, axis=-1, plan=_b.mxu_plan).astype(c_dtype)
                return jnp.fft.ifft(x_hat, axis=-1).astype(c_dtype)

            if kind == "identity":
                chains_out = chains
            elif kind == "zft":
                if use_mxu:
                    chains_out = mxu_fft(
                        chains, n=b.fft_len, axis=-1, norm="forward", plan=b.mxu_plan
                    ).astype(c_dtype)
                else:
                    norm = jnp.asarray(1.0 / float(b.fft_len), dtype=r_dtype)
                    chains_out = jnp.fft.fft(chains, axis=-1).astype(c_dtype) * norm
            elif kind == "zft_inverse":
                if use_mxu:
                    chains_out = mxu_ifft(
                        chains, n=b.fft_len, axis=-1, norm="forward", plan=b.mxu_plan
                    ).astype(c_dtype)
                else:
                    scale = jnp.asarray(float(b.fft_len), dtype=r_dtype)
                    chains_out = jnp.fft.ifft(chains, axis=-1).astype(c_dtype) * scale
            elif kind == "dz":
                kz_r = b.kz.astype(r_dtype)
                mult = jax.lax.complex(jnp.zeros_like(kz_r), kz_r).astype(c_dtype)
                chains_out = _inv_1d(_fwd_1d(chains) * mult)
            elif kind == "abs_dz":
                mult_r = jnp.abs(b.kz.astype(r_dtype))
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            elif kind == "dz2":
                kz_r = b.kz.astype(r_dtype)
                mult_r = -(kz_r * kz_r)
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            elif kind == "hyperz":
                mult_r = -b.hyperz_weight.astype(r_dtype)
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            elif kind == "dealias_chain":
                mult_r = b.kzm.astype(r_dtype)
                chains_out = _inv_1d(_fwd_1d(chains) * mult_r)
            else:  # pragma: no cover
                raise ValueError(f"Unknown NTFT bucket operator kind={kind!r}.")

            flat_outs.append(chains_out[:, : b.n_chains, :].reshape(n_moms, b.n_chains * b.fft_len))

        all_zyx = jnp.concatenate(flat_outs, axis=-1)
        res = all_zyx[:, self.inv_zyx_map].reshape(*batch_shape, self.nz, self.naky, self.nakx)
        return self._mask_out(res)

    def identity(self, arr: jax.Array) -> jax.Array:
        """NTFT gather + scatter round-trip (``linkedCopyNTFT`` -> ``linkedCopyBackNTFT``)."""
        return self._transform_buckets(arr, "identity")

    def dz(
        self, arr: jax.Array, *, accumulate: bool = False, target: jax.Array | None = None
    ) -> jax.Array:
        """NTFT parallel derivative ``d/dz`` (``GradParallelNTFT::dz``, ``L312-327``)."""
        c_dtype, _ = _complex_and_real_dtypes(arr)
        res = self._transform_buckets(arr, "dz")
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def abs_dz(
        self, arr: jax.Array, *, accumulate: bool = False, target: jax.Array | None = None
    ) -> jax.Array:
        """NTFT Landau operator ``|d/dz|`` (``GradParallelNTFT::abs_dz``, ``L379-394``)."""
        c_dtype, _ = _complex_and_real_dtypes(arr)
        res = self._transform_buckets(arr, "abs_dz")
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def dz2(self, arr: jax.Array) -> jax.Array:
        """NTFT second parallel derivative ``d^2/dz^2`` (``GradParallelNTFT::dz2``)."""
        return self._transform_buckets(arr, "dz2")

    def hyperz(
        self,
        arr: jax.Array,
        nu_hyper_z: float,
        *,
        accumulate: bool = False,
        target: jax.Array | None = None,
    ) -> jax.Array:
        """NTFT parallel hyperviscosity ``-nu_hyper_z * (|kz| / kz_max)^p_hyper_z``."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        res = jnp.asarray(nu_hyper_z, dtype=r_dtype) * self._transform_buckets(arr, "hyperz")
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def zft(self, arr: jax.Array) -> jax.Array:
        """Normalized forward NTFT chain FFT (``GradParallelNTFT::zft``, ``L254-263``)."""
        return self._transform_buckets(arr, "zft")

    def zft_inverse(self, arr: jax.Array) -> jax.Array:
        """Unnormalized inverse NTFT chain FFT (``GradParallelNTFT::zft_inverse``, ``L265-274``)."""
        return self._transform_buckets(arr, "zft_inverse")

    def dealias(self, arr: jax.Array) -> jax.Array:
        """Masked identity matching C++ no-op ``GradParallelNTFT::dealias`` (``L244-252``)."""
        self._check_shape(arr)
        c_dtype, _ = _complex_and_real_dtypes(arr)
        return self._mask_out(arr.astype(c_dtype))

    def dealias_chain(self, arr: jax.Array) -> jax.Array:
        """Opt-in helper applying the 2/3 parallel ``kz`` mask ``kzm`` on each NTFT chain."""
        return self._transform_buckets(arr, "dealias_chain")

    def apply_bcs(
        self,
        G: jax.Array,
        rhs: jax.Array,
        fields: Fields,
        solver: FieldSolver,
        dt: float | jax.Array,
        *,
        H: jax.Array | None = None,
    ) -> jax.Array:
        """Apply NTFT chain end damping on ``H`` (``dampEnds_linkedNTFT``, ``L2880-2945``).

        Matches C++ ``width = max(1, int(nLinks * widthfrac))``, ``idzl < width`` with
        ``x = idzl / width``, and ``idzl > nLinks - width - 1`` with
        ``x = (nLinks - idzl - 1) / width``, applied exclusively on non-zonal ``iky > 0`` modes.
        """
        c_dtype, r_dtype = _complex_and_real_dtypes(G)
        if self.boundary_option_periodic or self.local_limit:
            return rhs.astype(c_dtype)
        if H is None:
            H = build_nonadiabatic_h(G, fields, solver)
        h_c = H.astype(c_dtype)
        amp = jnp.asarray(self.damp_ends_amp, dtype=r_dtype) / jnp.asarray(dt, dtype=r_dtype)
        nu6 = self.end_damp_nu.astype(r_dtype)[None, None, None, :, :, :]
        mask6 = self.active_mask[None, None, None, None, :, :]
        damp_term = jnp.where(mask6, (nu6 * amp) * h_c, jnp.zeros_like(h_c))
        return rhs.astype(c_dtype) - damp_term

    def coupled_ikx_indices(self, iky: int, ikx: int) -> tuple[int, ...]:
        """Return connected component of active ``ikx`` indices coupled to ``(iky, ikx)``."""
        if not (0 <= iky < self.naky and 0 <= ikx < self.nakx):
            raise ValueError(f"(iky={iky}, ikx={ikx}) out of bounds for {(self.naky, self.nakx)}.")
        if iky == 0 or self.nakx == 1:
            return (int(ikx),)
        modes_y = np.asarray(self.mode_nums)[:, iky, :]  # (nz, nakx)
        adj: dict[int, set[int]] = {k: {k} for k in range(self.nakx)}
        for m_val in np.unique(modes_y):
            if m_val <= 0:
                continue
            touched = [int(k) for k in np.where(np.any(modes_y == m_val, axis=0))[0]]
            for k1 in touched:
                adj[k1].update(touched)
        visited: set[int] = set()
        stack = [int(ikx)]
        while stack:
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                stack.extend(adj[cur] - visited)
        return tuple(sorted(visited))

    def format_links(self) -> str:
        """Return human-readable NTFT chain summary mirroring ``GradParallelNTFT::linkPrint``."""
        half = (self.nakx + 1) // 2
        lines: list[str] = []
        for ic, b in enumerate(self.buckets):
            lines.append(f"NTFT Class {ic} (nLinks_pts={b.n_links}, nChains={b.n_chains}):")
            zyx_np = np.asarray(b.zyx_idx)[: b.n_chains]
            for row in zyx_np:
                pts = []
                for zyx in row:
                    idz = int(zyx) // (self.naky * self.nakx)
                    rem = int(zyx) % (self.naky * self.nakx)
                    idy = rem // self.nakx
                    idx = rem % self.nakx
                    signed_kx = idx if idx < half else (idx - self.nakx)
                    pts.append(f"(ky={idy},kx={signed_kx},iz={idz})")
                lines.append("  " + " ".join(pts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Precomputed Initialization Entry Points (FP64 Precomputation -> Optional Cast)
# ---------------------------------------------------------------------------


def _precompute_end_damp_linked(
    nz: int,
    naky: int,
    nakx: int,
    p_map: np.ndarray,
    n_links_map: np.ndarray,
    active_mask: np.ndarray,
    widthfrac: float,
) -> jax.Array:
    """Precompute 3D ``dampEnds_linked`` spatial profile ``nu(z, ky, kx)`` in ``float64``.

    Ported strictly from ``dampEnds_linked`` (``device_funcs.cu:L2831-2860``):
    - ``idzp = idz + nz * p_map[idy, idakx]``
    - ``width = int(nz * nLinks * widthfrac)``
    - ``idzp <= width -> x = idzp / width``
    - ``idzp >= nz * nLinks - width -> x = (nz * nLinks - idzp) / width``
    - Zero for ``idy == 0`` (zonal modes) and when ``width <= 0``.
    """
    nu = np.zeros((nz, naky, nakx), dtype=np.float64)
    idz_vec = np.arange(nz, dtype=np.int64)
    for idy in range(1, naky):
        for idx in range(nakx):
            if not active_mask[idy, idx]:
                continue
            p_val = int(p_map[idy, idx])
            n_lnk = int(n_links_map[idy, idx])
            nz_l = nz * n_lnk
            width = int(nz_l * float(widthfrac))
            if width <= 0:
                continue
            idzp = idz_vec + nz * p_val
            x_lo = idzp.astype(np.float64) / float(width)
            x_hi = (float(nz_l) - idzp.astype(np.float64)) / float(width)
            nu_lo = 1.0 - 2.0 * (x_lo**2) / (1.0 + x_lo**4)
            nu_hi = 1.0 - 2.0 * (x_hi**2) / (1.0 + x_hi**4)
            nu[:, idy, idx] = np.where(
                idzp <= width,
                nu_lo,
                np.where(idzp >= nz_l - width, nu_hi, 0.0),
            )
    return jnp.asarray(nu, dtype=jnp.float64)


def _precompute_end_damp_ntft(
    nz: int,
    naky: int,
    nakx: int,
    topo: NTFTChainTopology,
    active_mask: np.ndarray,
    widthfrac: float,
) -> jax.Array:
    """Precompute 3D ``dampEnds_linkedNTFT`` spatial profile ``nu(z, ky, kx)`` in ``float64``.

    Ported strictly from ``dampEnds_linkedNTFT`` (``device_funcs.cu:L2880-2945``):
    - ``width = int(nLinks * widthfrac); if width == 0: width = 1``
    - ``idzl < width -> x = idzl / width``
    - ``idzl > nLinks - width - 1 -> x = (nLinks - idzl - 1) / width``
    - Applied exclusively on non-zonal ``iky > 0`` chain points.
    """
    nu = np.zeros((nz, naky, nakx), dtype=np.float64)
    for ic in range(topo.n_classes):
        lnk = topo.n_links[ic]
        chn = topo.n_chains[ic]
        width = int(lnk * float(widthfrac))
        if width == 0:
            width = 1
        idzl = np.arange(lnk, dtype=np.int64)
        x_lo = idzl.astype(np.float64) / float(width)
        x_hi = (float(lnk) - idzl.astype(np.float64) - 1.0) / float(width)
        nu_lo = 1.0 - 2.0 * (x_lo**2) / (1.0 + x_lo**4)
        nu_hi = 1.0 - 2.0 * (x_hi**2) / (1.0 + x_hi**4)
        nu_chain = np.where(
            idzl < width,
            nu_lo,
            np.where(idzl > lnk - width - 1, nu_hi, 0.0),
        )
        idz_c = topo.idz_chains[ic]
        iky_c = topo.iky_chains[ic]
        ikx_c = topo.ikx_chains[ic]
        for n_idx in range(chn):
            idy_row = iky_c[n_idx]
            if idy_row[0] <= 0:
                continue
            for p_idx in range(lnk):
                idy = int(idy_row[p_idx])
                idx = int(ikx_c[n_idx, p_idx])
                idz = int(idz_c[n_idx, p_idx])
                if idy > 0 and active_mask[idy, idx]:
                    nu[idz, idy, idx] = nu_chain[p_idx]
    return jnp.asarray(nu, dtype=jnp.float64)


def init_linked_parallel_grad(
    cfg: GXConfig,
    grids: Grids,
    *,
    dtype: jnp.dtype = jnp.float64,
    chain_pad_multiple: int = 1,
    fft_backend: str | None = None,
) -> LinkedParallelGrad:
    """Initialize :class:`LinkedParallelGrad` in IEEE ``float64`` (with optional cast)."""
    if cfg.local_limit:
        raise ValueError(
            "cfg.local_limit=True uses ParallelGrad (GradParallelLocal), not LinkedParallelGrad."
        )
    topo = build_linked_topology(grids, cfg.jtwist)
    nz = int(grids.nz)
    naky = int(grids.naky)
    nakx = int(grids.nakx)
    zp = int(grids.zp)
    max_chain_len = max((max(1, nz * lnk) for lnk in topo.n_links), default=max(1, nz))
    fb_req = cfg.fft_backend if fft_backend is None else fft_backend
    resolved_fb = resolve_fft_backend(
        fb_req,
        nx=int(grids.nx),
        ny=int(grids.ny),
        nz=max(1, nz),
        max_chain_len=max_chain_len,
    )

    buckets: list[LinkedBucket] = []
    inv_yx_perm_np = np.full((naky, nakx), -1, dtype=np.int32)
    seg_offset = 0

    for ic in range(topo.n_classes):
        lnk = topo.n_links[ic]
        chn = topo.n_chains[ic]
        chn_pad = _pad_chain_count(chn, chain_pad_multiple)
        fft_len = nz * lnk

        yx_active = topo.iky_chains[ic] * nakx + topo.ikx_chains[ic]  # (chn, lnk)
        yx_padded = np.zeros((chn_pad, lnk), dtype=np.int32)
        yx_padded[:chn, :] = yx_active
        c_mask = np.zeros((chn_pad,), dtype=bool)
        c_mask[:chn] = True

        for n_idx in range(chn):
            for p_idx in range(lnk):
                idy = int(topo.iky_chains[ic][n_idx, p_idx])
                idx = int(topo.ikx_chains[ic][n_idx, p_idx])
                inv_yx_perm_np[idy, idx] = seg_offset + n_idx * lnk + p_idx
        seg_offset += chn * lnk

        # GradParallelLinked calls init_kzLinked(..., dealias=false) (grad_parallel_linked.cu:L151)
        kz64_np = _kz_linked_np(nz, zp, lnk, dealias_kz=False)
        kz64 = jnp.asarray(kz64_np, dtype=jnp.float64)
        kzm = jnp.asarray(_kzm_linked_np(fft_len), dtype=jnp.int32)
        hyp_w64 = jnp.asarray(_hyperz_weight_np(kz64_np, cfg.p_hyper_z), dtype=jnp.float64)
        mxu_plan = init_mxu_1d_plan(fft_len, dtype=jnp.float64) if resolved_fb == "mxu" else None

        buckets.append(
            LinkedBucket(
                yx_idx=jnp.asarray(yx_padded, dtype=jnp.int32),
                chain_mask=jnp.asarray(c_mask, dtype=jnp.bool_),
                kz=kz64,
                kzm=kzm,
                hyperz_weight=hyp_w64,
                n_links=lnk,
                n_chains=chn,
                n_chains_padded=chn_pad,
                fft_len=fft_len,
                mxu_plan=mxu_plan,
            )
        )

    if seg_offset != naky * nakx or np.any(inv_yx_perm_np < 0):
        raise RuntimeError("Linked chain partition did not bijectively cover (naky, nakx).")

    active_mask_np = np.asarray(grids.active_mask, dtype=bool)
    if cfg.boundary_option_periodic:
        end_damp_nu = jnp.zeros((nz, naky, nakx), dtype=jnp.float64)
    else:
        end_damp_nu = _precompute_end_damp_linked(
            nz,
            naky,
            nakx,
            topo.p_map,
            topo.n_links_map,
            active_mask_np,
            cfg.damp_ends_widthfrac,
        )

    op64 = LinkedParallelGrad(
        buckets=tuple(buckets),
        inv_yx_perm=jnp.asarray(inv_yx_perm_np, dtype=jnp.int32),
        end_damp_nu=end_damp_nu,
        active_mask=grids.active_mask,
        p_map=jnp.asarray(topo.p_map, dtype=jnp.int32),
        n_map=jnp.asarray(topo.n_map, dtype=jnp.int32),
        c_map=jnp.asarray(topo.c_map, dtype=jnp.int32),
        n_links_map=jnp.asarray(topo.n_links_map, dtype=jnp.int32),
        n_chains_map=jnp.asarray(topo.n_chains_map, dtype=jnp.int32),
        nz=nz,
        naky=naky,
        nakx=nakx,
        zp=zp,
        jtwist=int(cfg.jtwist),
        n_classes=topo.n_classes,
        n_links=topo.n_links,
        n_chains=topo.n_chains,
        chain_pad_multiple=int(chain_pad_multiple),
        local_limit=bool(cfg.local_limit),
        boundary_option_periodic=bool(cfg.boundary_option_periodic),
        dealias_kz=bool(cfg.dealias_kz),
        damp_ends_widthfrac=float(cfg.damp_ends_widthfrac),
        damp_ends_amp=float(cfg.damp_ends_amp),
        p_hyper_z=int(cfg.p_hyper_z),
        fft_backend=resolved_fb,
    )
    target = jnp.dtype(dtype)
    r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
    return op64 if r_target == jnp.float64 else op64.astype(r_target)


def init_ntft_parallel_grad(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    *,
    dtype: jnp.dtype = jnp.float64,
    n_links_max: int = 3000,
    chain_pad_multiple: int = 1,
    fft_backend: str | None = None,
) -> NTFTParallelGrad:
    """Initialize :class:`NTFTParallelGrad` in IEEE ``float64`` (with optional cast)."""
    if cfg.local_limit:
        raise ValueError(
            "cfg.local_limit=True uses ParallelGrad (GradParallelLocal), not NTFTParallelGrad."
        )
    if geo is None or geo.m0.size == 0:
        raise ValueError(
            "Geometry with populated geo.m0 (cfg.non_twist=True) is required for NTFTParallelGrad."
        )
    topo = build_ntft_topology(grids, geo.m0, cfg.jtwist, n_links_max=n_links_max)
    nz = int(grids.nz)
    naky = int(grids.naky)
    nakx = int(grids.nakx)
    zp = int(grids.zp)
    active_mask_np = np.asarray(grids.active_mask, dtype=bool)
    max_chain_len = max((max(1, lnk) for lnk in topo.n_links), default=max(1, nz))
    fb_req = cfg.fft_backend if fft_backend is None else fft_backend
    resolved_fb = resolve_fft_backend(
        fb_req,
        nx=int(grids.nx),
        ny=int(grids.ny),
        nz=max(1, nz),
        max_chain_len=max_chain_len,
    )

    buckets: list[NTFTBucket] = []
    # Slot 0 in concatenated output is reserved for 0.0 (masked origin & nExtra-trimmed points)
    inv_zyx_np = np.zeros((nz, naky, nakx), dtype=np.int32)
    pt_offset = 1

    for ic in range(topo.n_classes):
        lnk = topo.n_links[ic]
        chn = topo.n_chains[ic]
        chn_pad = _pad_chain_count(chn, chain_pad_multiple)

        zyx_active = (topo.idz_chains[ic] * naky + topo.iky_chains[ic]) * nakx + topo.ikx_chains[ic]
        zyx_padded = np.zeros((chn_pad, lnk), dtype=np.int32)
        zyx_padded[:chn, :] = zyx_active
        c_mask = np.zeros((chn_pad,), dtype=bool)
        c_mask[:chn] = True

        for n_idx in range(chn):
            for p_idx in range(lnk):
                idz = int(topo.idz_chains[ic][n_idx, p_idx])
                idy = int(topo.iky_chains[ic][n_idx, p_idx])
                idx = int(topo.ikx_chains[ic][n_idx, p_idx])
                if active_mask_np[idy, idx]:
                    inv_zyx_np[idz, idy, idx] = pt_offset + n_idx * lnk + p_idx
        pt_offset += chn * lnk

        # GradParallelNTFT calls init_kzLinkedNTFT(..., dealias=false) (grad_parallel_NTFT.cu:L163)
        kz64_np = _kz_linked_ntft_np(nz, zp, lnk, dealias_kz=False)
        kz64 = jnp.asarray(kz64_np, dtype=jnp.float64)
        kzm = jnp.asarray(_kzm_linked_np(lnk), dtype=jnp.int32)
        hyp_w64 = jnp.asarray(_hyperz_weight_np(kz64_np, cfg.p_hyper_z), dtype=jnp.float64)
        mxu_plan = init_mxu_1d_plan(lnk, dtype=jnp.float64) if resolved_fb == "mxu" else None

        buckets.append(
            NTFTBucket(
                zyx_idx=jnp.asarray(zyx_padded, dtype=jnp.int32),
                chain_mask=jnp.asarray(c_mask, dtype=jnp.bool_),
                kz=kz64,
                kzm=kzm,
                hyperz_weight=hyp_w64,
                n_links=lnk,
                n_chains=chn,
                n_chains_padded=chn_pad,
                fft_len=lnk,
                mxu_plan=mxu_plan,
            )
        )

    if cfg.boundary_option_periodic:
        end_damp_nu = jnp.zeros((nz, naky, nakx), dtype=jnp.float64)
    else:
        end_damp_nu = _precompute_end_damp_ntft(
            nz, naky, nakx, topo, active_mask_np, cfg.damp_ends_widthfrac
        )

    op64 = NTFTParallelGrad(
        buckets=tuple(buckets),
        inv_zyx_map=jnp.asarray(inv_zyx_np, dtype=jnp.int32),
        end_damp_nu=end_damp_nu,
        active_mask=grids.active_mask,
        mode_nums=jnp.asarray(topo.mode_nums_active, dtype=jnp.int32),
        n_extra=jnp.asarray(topo.n_extra, dtype=jnp.int32),
        nz=nz,
        naky=naky,
        nakx=nakx,
        zp=zp,
        jtwist=int(cfg.jtwist),
        mode_count=topo.mode_count,
        n_classes=topo.n_classes,
        n_links=topo.n_links,
        n_chains=topo.n_chains,
        n_links_max=int(n_links_max),
        chain_pad_multiple=int(chain_pad_multiple),
        local_limit=bool(cfg.local_limit),
        boundary_option_periodic=bool(cfg.boundary_option_periodic),
        dealias_kz=bool(cfg.dealias_kz),
        damp_ends_widthfrac=float(cfg.damp_ends_widthfrac),
        damp_ends_amp=float(cfg.damp_ends_amp),
        p_hyper_z=int(cfg.p_hyper_z),
        fft_backend=resolved_fb,
    )
    target = jnp.dtype(dtype)
    r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
    return op64 if r_target == jnp.float64 else op64.astype(r_target)


def init_boundary_parallel(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry | None = None,
    *,
    dtype: jnp.dtype = jnp.float64,
    n_links_max: int = 3000,
    chain_pad_multiple: int = 1,
    fft_backend: str | None = None,
) -> LinkedParallelGrad | NTFTParallelGrad:
    """Dispatch between :class:`NTFTParallelGrad` and :class:`LinkedParallelGrad`.

    Matches ``Linear_GK::Linear_GK`` (``linear.cu:L27-36``):
    - ``cfg.non_twist == True`` -> :class:`NTFTParallelGrad` (requires ``geo`` with ``geo.m0``).
    - ``cfg.non_twist == False`` -> :class:`LinkedParallelGrad` (handles both multi-link
      ``nLinks > 1`` and single-segment ``nLinks == 1`` / ``boundary="periodic"``).
    """
    if cfg.non_twist:
        if geo is None:
            raise ValueError("geo (Geometry) is required when cfg.non_twist=True.")
        return init_ntft_parallel_grad(
            cfg,
            grids,
            geo,
            dtype=dtype,
            n_links_max=n_links_max,
            chain_pad_multiple=chain_pad_multiple,
            fft_backend=fft_backend,
        )
    return init_linked_parallel_grad(
        cfg,
        grids,
        dtype=dtype,
        chain_pad_multiple=chain_pad_multiple,
        fft_backend=fft_backend,
    )
