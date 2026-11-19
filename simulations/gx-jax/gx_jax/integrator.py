"""Unified gyrokinetic RHS, cancellation-free explicit RK integrators, and trajectory drivers.

Ported strictly from:
- ``gx/include/timestepper.h``
- ``gx/src/{ts_rk3.cu,ts_rk4.cu,ts_sspx2.cu,ts_sspx3.cu,ts_k10.cu}``
- ``gx/src/run_gx.cu``

Numerical & XLA/TPU notes:
- All five explicit schemes (``sspx2``, ``rk3``, ``sspx3``, ``rk4``, ``k10``) are formulated in
  algebraically exact increment form ``g_next = g0 + delta`` so low-storage Shu-Osher / Ketcheson
  linear combinations never subtract nearly equal ``O(1)`` states in runtime ``complex64``.
- Stage-level ``dealias_kz`` rules match C++ ``ts_sspx3.cu`` and ``ts_k10.cu`` verbatim
  (applied to ``linear_rhs`` and ``fields.phi`` in ``sspx3`` and stages 1..9 of ``k10``, and to
  both ``nonlinear_rhs`` and ``linear_rhs`` in stage 10 of ``k10``).
- Time ``time`` and timestep ``dt`` are always carried in IEEE ``float64`` (matching C++
  ``double time, dt_`` in ``timestepper.h`` / ``run_gx.cu``) even when ``G`` is ``complex64``,
  preventing ``float32`` time-accumulation drift over long trajectories.
- Compiled trajectory drivers :func:`integrate_scan` (``jax.lax.scan``, reverse-mode AD
  differentiable with optional ``remat=True`` gradient checkpointing) and
  :func:`integrate_while_loop` (``jax.lax.while_loop``, adaptive CFL variable-``dt`` integration)
  maintain strict PyTree structure, shape, and dtype invariance between initial carry and loop
  body for single-trace ``@jax.jit`` compilation.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from gx_jax.diagnostics_nc import DiagnosticsState

from gx_jax.config import GXConfig
from gx_jax.fields import (
    Fields,
    _complex_and_real_dtypes,
    _time_dtype,
    enforce_reality,
    mask_g,
    solve_fields,
)
from gx_jax.geometry_nc import Geometry
from gx_jax.grids import Grids
from gx_jax.linear_rhs import (
    LinearOperator,
    compute_linear_max_frequency,
    compute_phi2_ky,
    init_linear_operator,
    linear_rhs,
)
from gx_jax.nonlinear_exb import (
    ExBShearState,
    ForcingOperator,
    NonlinearOperator,
    _safe_sqrt,
    apply_forcing,
    compute_nonlinear_max_frequency,
    flow_shear_g_shift,
    flow_shear_shift,
    init_exb_shear,
    init_forcing,
    init_nonlinear_operator,
    nonlinear_rhs,
)

# Per-scheme CFL stability factors matching include/timestepper.h (L33, L64, L122, L154)
_SCHEME_CFL_FAC: dict[str, float] = {
    "rk4": 2.82,
    "rk3": 1.73,
    "sspx3": 1.73,
    "sspx2": 1.0,
    "k10": 1.0,
}


def get_scheme_cfl_fac(scheme: str) -> float:
    """Return explicit Runge-Kutta CFL stability factor ``cfl_fac`` (``timestepper.h``)."""
    sk = scheme.strip().lower()
    if sk not in _SCHEME_CFL_FAC:
        raise ValueError(
            f"Unsupported time scheme={scheme!r}. Supported: {sorted(_SCHEME_CFL_FAC.keys())}."
        )
    return _SCHEME_CFL_FAC[sk]


def _scheme_honors_fixed_dt(scheme: str) -> bool:
    """Return whether ``scheme`` honors ``fixed_dt`` in C++ ``ts_*.cu``.

    In C++ GX, only ``RK3`` (``ts_rk3.cu:L98``), ``RK4`` (``ts_rk4.cu:L49``), and ``SSPx3``
    (``ts_sspx3.cu:L101``) check ``if (!pars_->fixed_dt)``, whereas ``SSPx2``
    (``ts_sspx2.cu:L47``) and ``Ketcheson10`` (``ts_k10.cu:L56-62``) recompute ``dt_`` from the
    CFL condition on every step regardless of ``pars_->fixed_dt``.
    """
    return scheme.strip().lower() in ("rk3", "rk4", "sspx3")


class StepResult(NamedTuple):
    """Return value of :func:`step_gyrokinetic` (unpacks as a 4-tuple ``(G, fields, dt, exb)``)."""

    G: jax.Array
    fields: Fields
    dt: jax.Array
    exb_state: ExBShearState | None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GyrokineticOperator:
    """Unified precomputed gyrokinetic operator container for linear + nonlinear evolution."""

    linear_op: LinearOperator
    nonlinear_op: NonlinearOperator | None
    grids: Grids
    geo: Geometry
    forcing_op: ForcingOperator | None
    exb_init: ExBShearState | None
    omega_linear: tuple[float, float, float]

    scheme: str
    cfl: float
    cfl_fac: float
    dt: float
    dt_min: float
    dt_max: float
    t_max: float
    nstep: int
    random_seed: int
    fixed_dt: bool
    dealias_kz: bool
    nonlinear_mode: bool
    exb_shear: bool
    forcing_init: bool

    def tree_flatten(self):
        children = (
            self.linear_op,
            self.nonlinear_op,
            self.grids,
            self.geo,
            self.forcing_op,
            self.exb_init,
        )
        aux_data = (
            self.omega_linear,
            self.scheme,
            self.cfl,
            self.cfl_fac,
            self.dt,
            self.dt_min,
            self.dt_max,
            self.t_max,
            self.nstep,
            self.random_seed,
            self.fixed_dt,
            self.dealias_kz,
            self.nonlinear_mode,
            self.exb_shear,
            self.forcing_init,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> GyrokineticOperator:
        (
            omega_linear,
            scheme,
            cfl,
            cfl_fac,
            dt,
            dt_min,
            dt_max,
            t_max,
            nstep,
            random_seed,
            fixed_dt,
            dealias_kz,
            nonlinear_mode,
            exb_shear,
            forcing_init,
        ) = aux_data
        linear_op, nonlinear_op, grids, geo, forcing_op, exb_init = children
        return cls(
            linear_op=linear_op,
            nonlinear_op=nonlinear_op,
            grids=grids,
            geo=geo,
            forcing_op=forcing_op,
            exb_init=exb_init,
            omega_linear=omega_linear,
            scheme=scheme,
            cfl=cfl,
            cfl_fac=cfl_fac,
            dt=dt,
            dt_min=dt_min,
            dt_max=dt_max,
            t_max=t_max,
            nstep=nstep,
            random_seed=random_seed,
            fixed_dt=fixed_dt,
            dealias_kz=dealias_kz,
            nonlinear_mode=nonlinear_mode,
            exb_shear=exb_shear,
            forcing_init=forcing_init,
        )

    def astype(self, dtype: jnp.dtype) -> GyrokineticOperator:
        """Cast floating/complex operator leaves to runtime ``dtype`` (e.g. ``jnp.float32``)."""
        target = jnp.dtype(dtype)
        r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        return GyrokineticOperator(
            linear_op=self.linear_op.astype(r_target),
            nonlinear_op=self.nonlinear_op.astype(r_target)
            if self.nonlinear_op is not None
            else None,
            grids=self.grids.astype(r_target),
            geo=self.geo.astype(r_target),
            forcing_op=self.forcing_op.astype(r_target) if self.forcing_op is not None else None,
            exb_init=self.exb_init.astype(r_target) if self.exb_init is not None else None,
            omega_linear=self.omega_linear,
            scheme=self.scheme,
            cfl=self.cfl,
            cfl_fac=self.cfl_fac,
            dt=self.dt,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            t_max=self.t_max,
            nstep=self.nstep,
            random_seed=self.random_seed,
            fixed_dt=self.fixed_dt,
            dealias_kz=self.dealias_kz,
            nonlinear_mode=self.nonlinear_mode,
            exb_shear=self.exb_shear,
            forcing_init=self.forcing_init,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IntegratorState:
    """Loop-carry state for :func:`integrate_scan` and :func:`integrate_while_loop`.

    All fields maintain strictly invariant shapes and dtypes across initial carry and loop body.
    ``time`` and ``dt`` are always carried in IEEE ``float64`` (matching C++ ``double time, dt_``).
    """

    G: jax.Array  # (nspecies, nhermite, nlaguerre, nz, naky, nakx) complex64 / complex128
    fields: Fields  # (nz, naky, nakx) complex64 / complex128
    time: jax.Array  # () float64
    dt: jax.Array  # () float64
    step: jax.Array  # () int32
    exb_state: ExBShearState | None = None
    key: jax.Array = dataclasses.field(default_factory=lambda: jax.random.PRNGKey(22))

    def tree_flatten(self):
        return (
            self.G,
            self.fields,
            self.time,
            self.dt,
            self.step,
            self.exb_state,
            self.key,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> IntegratorState:
        del aux_data
        G, fields, time, dt, step, exb_state, key = children
        return cls(
            G=G,
            fields=fields,
            time=time,
            dt=dt,
            step=step,
            exb_state=exb_state,
            key=key,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TrajectoryResult:
    """Output container returned by :func:`integrate_scan`."""

    G: jax.Array
    fields: Fields
    time: jax.Array
    dt: jax.Array
    steps: jax.Array
    t_history: jax.Array  # (n_chunks,) float64
    dt_history: jax.Array  # (n_chunks,) float64
    phi2_history: jax.Array  # (n_chunks, naky)
    wg_history: jax.Array  # (n_chunks,)
    G_history: jax.Array | None = None  # (n_chunks, ...) when save_g_history=True
    exb_state: ExBShearState | None = None

    def tree_flatten(self):
        children = (
            self.G,
            self.fields,
            self.time,
            self.dt,
            self.steps,
            self.t_history,
            self.dt_history,
            self.phi2_history,
            self.wg_history,
            self.G_history,
            self.exb_state,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> TrajectoryResult:
        del aux_data
        (
            G,
            fields,
            time,
            dt,
            steps,
            t_history,
            dt_history,
            phi2_history,
            wg_history,
            G_history,
            exb_state,
        ) = children
        return cls(
            G=G,
            fields=fields,
            time=time,
            dt=dt,
            steps=steps,
            t_history=t_history,
            dt_history=dt_history,
            phi2_history=phi2_history,
            wg_history=wg_history,
            G_history=G_history,
            exb_state=exb_state,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ChunkResult:
    """Output container returned by fused on-device chunked time steppers.

    Carries the post-chunk :class:`IntegratorState`, the penultimate state
    ``(G_prev, fields_prev, time_prev)`` immediately preceding the last executed
    step, the integer step count ``steps_done`` actually executed (``<= max_steps``
    if ``t_max`` was reached mid-chunk), and an optional carried
    :class:`DiagnosticsState` for exact C++ ``Diagnostics_GK::loop`` ``save_old``
    semantics.
    """

    state: IntegratorState
    G_prev: jax.Array
    fields_prev: Fields
    time_prev: jax.Array  # () float64
    steps_done: jax.Array  # () int32
    diag_state: DiagnosticsState | None = None

    def tree_flatten(self):
        children = (
            self.state,
            self.G_prev,
            self.fields_prev,
            self.time_prev,
            self.steps_done,
            self.diag_state,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> ChunkResult:
        del aux_data
        state, G_prev, fields_prev, time_prev, steps_done, diag_state = children
        return cls(
            state=state,
            G_prev=G_prev,
            fields_prev=fields_prev,
            time_prev=time_prev,
            steps_done=steps_done,
            diag_state=diag_state,
        )


def init_gyrokinetic_operator(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    *,
    linear_op: LinearOperator | None = None,
    nonlinear_op: NonlinearOperator | None = None,
    forcing_op: ForcingOperator | None = None,
    exb_init: ExBShearState | None = None,
    dtype: jnp.dtype = jnp.float64,
) -> GyrokineticOperator:
    """Initialize unified :class:`GyrokineticOperator` in IEEE ``float64`` (with optional cast)."""
    target = jnp.dtype(dtype)
    r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32

    lin64 = (
        init_linear_operator(cfg, grids, geo, dtype=jnp.float64) if linear_op is None else linear_op
    )
    if nonlinear_op is not None:
        nl64: NonlinearOperator | None = nonlinear_op
    elif cfg.nonlinear_mode:
        nl64 = init_nonlinear_operator(cfg, grids, geo, dtype=jnp.float64)
    else:
        nl64 = None

    if forcing_op is not None:
        forc64: ForcingOperator | None = forcing_op
    elif cfg.forcing_init:
        forc64 = init_forcing(cfg, grids, dtype=jnp.float64)
    else:
        forc64 = None

    if exb_init is not None:
        exb64: ExBShearState | None = exb_init
    elif cfg.exb_shear:
        exb64 = init_exb_shear(cfg, grids, geo, dtype=jnp.float64)
    else:
        exb64 = None

    w_lin = compute_linear_max_frequency(cfg, grids, geo)
    scheme_clean = cfg.scheme.strip().lower()
    cfl_fac = get_scheme_cfl_fac(scheme_clean)

    op64 = GyrokineticOperator(
        linear_op=lin64,
        nonlinear_op=nl64,
        grids=grids,
        geo=geo,
        forcing_op=forc64,
        exb_init=exb64,
        omega_linear=(float(w_lin[0]), float(w_lin[1]), float(w_lin[2])),
        scheme=scheme_clean,
        cfl=float(cfg.cfl),
        cfl_fac=float(cfl_fac),
        dt=float(cfg.dt),
        dt_min=float(cfg.dt_min),
        dt_max=float(cfg.dt_max),
        t_max=float(cfg.t_max),
        nstep=int(cfg.nstep),
        random_seed=int(cfg.random_seed),
        fixed_dt=bool(cfg.fixed_dt),
        dealias_kz=bool(cfg.dealias_kz),
        nonlinear_mode=bool(cfg.nonlinear_mode),
        exb_shear=bool(cfg.exb_shear),
        forcing_init=bool(cfg.forcing_init),
    )
    return op64 if r_target == jnp.float64 else op64.astype(r_target)


def compute_max_frequency(
    fields: Fields,
    gk_op: GyrokineticOperator,
    *,
    exb_state: ExBShearState | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute combined linear + nonlinear max frequencies ``(w0, w1, w2)``.

    Matches ``Linear_GK::get_max_frequency`` (``linear.cu:L246-272``) followed by
    ``Nonlinear_GK::get_max_frequency`` (``nonlinear.cu:L306-345``).
    """
    _, r_dtype = _complex_and_real_dtypes(fields.phi)
    w_lin = gk_op.omega_linear
    if gk_op.nonlinear_op is not None and gk_op.nonlinear_mode:
        return compute_nonlinear_max_frequency(
            fields,
            gk_op.nonlinear_op,
            omega_linear=w_lin,
            exb_state=exb_state,
        )
    return (
        jnp.asarray(w_lin[0], dtype=r_dtype),
        jnp.asarray(w_lin[1], dtype=r_dtype),
        jnp.asarray(w_lin[2], dtype=r_dtype),
    )


def compute_cfl_dt(
    fields: Fields,
    gk_op: GyrokineticOperator,
    *,
    exb_state: ExBShearState | None = None,
    scheme: str | None = None,
) -> jax.Array:
    """Compute adaptive CFL timestep ``dt`` matching C++ ``ts_*.cu``."""
    _, r_dtype = _complex_and_real_dtypes(fields.phi)
    t_dtype = _time_dtype(r_dtype)
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    if gk_op.fixed_dt and _scheme_honors_fixed_dt(sk):
        return jnp.asarray(gk_op.dt, dtype=t_dtype)

    cfl_fac = jnp.asarray(get_scheme_cfl_fac(sk), dtype=t_dtype)
    cfl = jnp.asarray(gk_op.cfl, dtype=t_dtype)
    dt_min = jnp.asarray(gk_op.dt_min, dtype=t_dtype)
    dt_max = jnp.asarray(gk_op.dt_max, dtype=t_dtype)

    w0, w1, w2 = compute_max_frequency(fields, gk_op, exb_state=exb_state)
    wmax = w0.astype(t_dtype) + w1.astype(t_dtype) + w2.astype(t_dtype)
    pos = wmax > 0.0
    safe_wmax = jnp.where(pos, wmax, jnp.ones_like(wmax))
    dt_guess = (cfl_fac * cfl) / safe_wmax

    if sk == "k10":
        # ts_k10.cu:L62: dt_ = min(pars_->cfl / wmax, dt_max)
        dt_clamped = jnp.minimum(dt_guess, dt_max)
    else:
        dt_clamped = jnp.minimum(jnp.maximum(dt_guess, dt_min), dt_max)
    return jnp.where(pos, dt_clamped, dt_max).astype(t_dtype)


def _update_linear_op_for_exb(
    lin_op: LinearOperator,
    exb_state: ExBShearState | None,
    r_dtype: jnp.dtype,
) -> LinearOperator:
    """Update ``LinearOperator`` (and ``solver.jl``/``jlb``) for shifted ``exb_state.kperp2``.

    In C++ GX, ``rhs_linear``, ``streaming_rhs``, ``conservation_terms``, ``applyBCs``, and
    ``real_space_density``/``currents`` evaluate ``Jflr``/``JflrA``/``JflrB`` on the shifted
    ``geo->kperp2``, while the quasineutrality/Ampère denominators in ``solver.cu:L69`` remain
    frozen at ``t = 0``. Evaluates strictly in ``r_dtype`` (zero ``f64`` ops in ``complex64``).
    """
    if exb_state is None:
        return lin_op

    kp2 = exb_state.kperp2.astype(r_dtype)
    omegad = exb_state.omegad.astype(r_dtype)
    cv_d = exb_state.cv_d.astype(r_dtype)
    gb_d = exb_state.gb_d.astype(r_dtype)
    if exb_state.g_exb == 0.0:
        return dataclasses.replace(
            lin_op,
            kperp2=kp2,
            omegad=omegad,
            cv_d=cv_d,
            gb_d=gb_d,
        )

    nl = lin_op.nlaguerre
    rho2 = lin_op.solver.sp_rho2.astype(r_dtype)[:, None, None, None]
    b_s = kp2[None, :, :, :] * rho2  # (nspecies, nz, naky, nakx)
    sqrt_b = _safe_sqrt(b_s)
    neg_hb_s = jnp.asarray(-0.5, dtype=r_dtype) * b_s
    j_cur = jnp.exp(neg_hb_s)
    jl_list = [j_cur]
    for l_idx in range(1, nl + 1):
        j_cur = j_cur * (neg_hb_s / jnp.asarray(float(l_idx), dtype=r_dtype))
        jl_list.append(j_cur)
    jl_all = jnp.stack(jl_list, axis=1)  # (nspecies, nl + 1, nz, naky, nakx)

    jl = jl_all[:, :nl]
    jl_p1 = jl_all[:, 1 : nl + 1]
    jl_m1 = jnp.concatenate([jnp.zeros_like(jl[:, :1]), jl[:, :-1]], axis=1)
    l_vec = jnp.arange(nl, dtype=r_dtype)[None, :, None, None, None]
    two = jnp.asarray(2.0, dtype=r_dtype)
    one = jnp.asarray(1.0, dtype=r_dtype)

    jla = l_vec * jl_m1 + (two * l_vec) * jl + (l_vec + one) * jl_p1
    jlb_all = jl_all + jnp.concatenate([jnp.zeros_like(jl_all[:, :1]), jl_all[:, :-1]], axis=1)
    jlb = jlb_all[:, :nl]
    jlb_p1 = jlb_all[:, 1 : nl + 1]
    jlb_m1 = jnp.concatenate([jnp.zeros_like(jlb[:, :1]), jlb[:, :-1]], axis=1)
    jlba = l_vec * jlb_m1 + (two * l_vec) * jlb + (l_vec + one) * jlb_p1

    solver_shifted = dataclasses.replace(lin_op.solver, jl=jl, jlb=jlb, jla=jla, kperp2=kp2)
    return dataclasses.replace(
        lin_op,
        solver=solver_shifted,
        jl=jl,
        jlb=jlb,
        jla=jla,
        jlba=jlba,
        b_s=b_s,
        sqrt_b=sqrt_b,
        kperp2=kp2,
        omegad=omegad,
        cv_d=cv_d,
        gb_d=gb_d,
    )


def gyrokinetic_rhs(
    G: jax.Array,
    fields: Fields,
    gk_op: GyrokineticOperator,
    dt: float | jax.Array,
    *,
    exb_state: ExBShearState | None = None,
    apply_kz_dealias_linear: bool = False,
    apply_kz_dealias_nonlinear: bool = False,
) -> jax.Array:
    """Evaluate unified ``nonlinear_rhs + linear_rhs`` tendency for one Runge-Kutta stage.

    Matches the per-stage operator sequence in ``ts_rk3.cu``, ``ts_rk4.cu``, ``ts_sspx2.cu``,
    ``ts_sspx3.cu``, and ``ts_k10.cu``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    f_c = fields.astype(c_dtype)
    dt_r = jnp.asarray(dt, dtype=r_dtype)
    dt_safe = jnp.where(dt_r > 0.0, dt_r, jnp.ones_like(dt_r))

    lin_op = _update_linear_op_for_exb(gk_op.linear_op, exb_state, r_dtype)
    gp = lin_op.grad_par

    rhs_lin = linear_rhs(g_c, f_c, lin_op, dt_safe, grids=gk_op.grids, geo=gk_op.geo)
    if apply_kz_dealias_linear:
        rhs_lin = gp.dealias(rhs_lin)

    if gk_op.nonlinear_mode and gk_op.nonlinear_op is not None:
        rhs_nl = nonlinear_rhs(
            g_c,
            f_c,
            gk_op.nonlinear_op,
            exb_state=exb_state,
        )
        if apply_kz_dealias_nonlinear:
            rhs_nl = gp.dealias(rhs_nl)
        total = rhs_nl + rhs_lin
    else:
        total = rhs_lin

    mask6 = lin_op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, total, jnp.zeros_like(total))


def _solve_stage_fields(
    G: jax.Array,
    gk_op: GyrokineticOperator,
    *,
    exb_state: ExBShearState | None = None,
    apply_kz_dealias: bool = False,
) -> Fields:
    """Solve quasineutrality & Ampère's laws and optionally apply ``dealias_kz`` to ``phi``."""
    _, r_dtype = _complex_and_real_dtypes(G)
    lin_op = _update_linear_op_for_exb(gk_op.linear_op, exb_state, r_dtype)
    f_out = solve_fields(G, lin_op.solver)
    if apply_kz_dealias and gk_op.dealias_kz:
        phi_d = lin_op.grad_par.dealias(f_out.phi)
        f_out = Fields(phi=phi_d, apar=f_out.apar, bpar=f_out.bpar, phi_avg=f_out.phi_avg)
    return f_out


def step_gyrokinetic(
    G: jax.Array,
    fields: Fields,
    gk_op: GyrokineticOperator,
    dt: float | jax.Array | None = None,
    *,
    scheme: str | None = None,
    exb_state: ExBShearState | None = None,
    enforce_reality_cond: bool = False,
    key: jax.Array | None = None,
    step_idx: int | jax.Array = 0,
) -> StepResult:
    """Advance ``(G, fields)`` by one explicit gyrokinetic timestep ``dt``.

    Returns :class:`StepResult` ``(G_next, fields_next, dt_used, exb_next)`` (which unpacks as a
    4-tuple). All five schemes (``sspx2``, ``rk3``, ``sspx3``, ``rk4``, ``k10``) use
    cancellation-free increment updates ``g0 + delta`` matching C++ ``ts_*.cu``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    t_dtype = _time_dtype(r_dtype)
    g0 = G.astype(c_dtype)
    f0 = fields.astype(c_dtype)
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    dealias_kz = bool(gk_op.dealias_kz)
    exb_cur = exb_state.astype(r_dtype) if exb_state is not None else None

    if dt is None:
        dt_f64 = compute_cfl_dt(f0, gk_op, exb_state=exb_cur, scheme=sk)
    else:
        dt_f64 = jnp.asarray(dt, dtype=t_dtype)
    dt_r = dt_f64.astype(r_dtype)

    # For rk3, sspx2, sspx3, k10, ExB flow shear shifts occur at the start of advance()
    if exb_cur is not None and sk in ("rk3", "sspx2", "sspx3", "k10"):
        exb_cur, f0 = flow_shear_shift(exb_cur, f0, gk_op.grids, gk_op.geo, dt_r)
        g0 = flow_shear_g_shift(g0, exb_cur, gk_op.grids)

    def _rhs(
        g_in: jax.Array,
        f_in: Fields,
        exb_in: ExBShearState | None,
        *,
        d_lin: bool = False,
        d_nl: bool = False,
    ) -> jax.Array:
        return gyrokinetic_rhs(
            g_in,
            f_in,
            gk_op,
            dt_r,
            exb_state=exb_in,
            apply_kz_dealias_linear=d_lin,
            apply_kz_dealias_nonlinear=d_nl,
        )

    if sk == "rk4":
        half = jnp.asarray(0.5, dtype=r_dtype)
        sixth = jnp.asarray(1.0 / 6.0, dtype=r_dtype)
        two = jnp.asarray(2.0, dtype=r_dtype)

        k1 = _rhs(g0, f0, exb_cur)
        g1 = g0 + (half * dt_r) * k1
        f1 = _solve_stage_fields(g1, gk_op, exb_state=exb_cur)

        if exb_cur is not None:
            exb_cur, f1 = flow_shear_shift(exb_cur, f1, gk_op.grids, gk_op.geo, half * dt_r)
            g0 = flow_shear_g_shift(g0, exb_cur, gk_op.grids)
            k1 = flow_shear_g_shift(k1, exb_cur, gk_op.grids)
            g1 = flow_shear_g_shift(g1, exb_cur, gk_op.grids)

        k2 = _rhs(g1, f1, exb_cur)
        g2 = g0 + (half * dt_r) * k2
        f2 = _solve_stage_fields(g2, gk_op, exb_state=exb_cur)

        k3 = _rhs(g2, f2, exb_cur)
        g3 = g0 + dt_r * k3
        f3 = _solve_stage_fields(g3, gk_op, exb_state=exb_cur)

        if exb_cur is not None:
            exb_cur, f3 = flow_shear_shift(exb_cur, f3, gk_op.grids, gk_op.geo, half * dt_r)
            g0 = flow_shear_g_shift(g0, exb_cur, gk_op.grids)
            k1 = flow_shear_g_shift(k1, exb_cur, gk_op.grids)
            k2 = flow_shear_g_shift(k2, exb_cur, gk_op.grids)
            k3 = flow_shear_g_shift(k3, exb_cur, gk_op.grids)
            g3 = flow_shear_g_shift(g3, exb_cur, gk_op.grids)

        k4 = _rhs(g3, f3, exb_cur)
        g_next = g0 + (sixth * dt_r) * (k1 + two * k2 + two * k3 + k4)
        apply_final_kz_dealias = False

    elif sk == "rk3":
        # Heun's 3-stage RK3 (ts_rk3.cu:L49-119)
        c13 = jnp.asarray(1.0 / 3.0, dtype=r_dtype)
        c23 = jnp.asarray(2.0 / 3.0, dtype=r_dtype)
        c34 = jnp.asarray(0.75, dtype=r_dtype)
        c14 = jnp.asarray(0.25, dtype=r_dtype)

        k1 = _rhs(g0, f0, exb_cur)
        g_q1 = g0 + (c13 * dt_r) * k1
        f_q1 = _solve_stage_fields(g_q1, gk_op, exb_state=exb_cur)

        k2 = _rhs(g_q1, f_q1, exb_cur)
        g_q2 = g0 + (c23 * dt_r) * k2
        f_q2 = _solve_stage_fields(g_q2, gk_op, exb_state=exb_cur)

        k3 = _rhs(g_q2, f_q2, exb_cur)
        g_next = g0 + (c34 * dt_r) * k3 + (c14 * dt_r) * k1
        apply_final_kz_dealias = False

    elif sk == "sspx2":
        # 2-stage SSP RK2 (ts_sspx2.cu:L32-93) in increment form g0 + delta:
        # (2 - sqrt2)*g0 + (sqrt2 - 2)*g1 + g2 == g0 + (sqrt2 - 1)*d0 + d1
        adt = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
        c_d0 = jnp.asarray(math.sqrt(2.0) - 1.0, dtype=r_dtype)

        d0 = (adt * dt_r) * _rhs(g0, f0, exb_cur)
        g1 = g0 + d0
        f1 = _solve_stage_fields(g1, gk_op, exb_state=exb_cur)

        d1 = (adt * dt_r) * _rhs(g1, f1, exb_cur)
        g_next = g0 + (c_d0 * d0 + d1)
        apply_final_kz_dealias = False

    elif sk == "sspx3":
        # 3-stage SSP RK3 (ts_sspx3.cu:L65-133) in increment form g0 + delta:
        #   d0 = adt * dt * r0,                g1 = g0 + d0
        #   d2 = w1 * d0 + adt * dt * r1,      g2 = g0 + d2
        #   g_next = g0 + (w3 * d0 + w2 * d2 + adt * dt * r2)
        adt_f = (1.0 / 6.0) ** (1.0 / 3.0)
        wgtfac = math.sqrt(9.0 - 2.0 * (6.0 ** (2.0 / 3.0)))
        w1_f = 0.5 * (wgtfac - 1.0)
        w2_f = 0.5 * (6.0 ** (2.0 / 3.0) - 1.0 - wgtfac)
        w3_f = 1.0 / adt_f - 1.0 - w2_f * (w1_f + 1.0)

        adt = jnp.asarray(adt_f, dtype=r_dtype)
        w1 = jnp.asarray(w1_f, dtype=r_dtype)
        w2 = jnp.asarray(w2_f, dtype=r_dtype)
        w3 = jnp.asarray(w3_f, dtype=r_dtype)

        d0 = (adt * dt_r) * _rhs(g0, f0, exb_cur, d_lin=dealias_kz, d_nl=False)
        g1 = g0 + d0
        f1 = _solve_stage_fields(g1, gk_op, exb_state=exb_cur, apply_kz_dealias=True)

        d1 = (adt * dt_r) * _rhs(g1, f1, exb_cur, d_lin=dealias_kz, d_nl=False)
        d2 = w1 * d0 + d1
        g2 = g0 + d2
        f2 = _solve_stage_fields(g2, gk_op, exb_state=exb_cur, apply_kz_dealias=True)

        d_last = (adt * dt_r) * _rhs(g2, f2, exb_cur, d_lin=dealias_kz, d_nl=False)
        g_next = g0 + (w3 * d0 + w2 * d2 + d_last)
        apply_final_kz_dealias = True

    elif sk == "k10":
        # Ketcheson 10-stage, 4th-order low-storage RK (ts_k10.cu:L49-137) in increment form:
        #   d5 = sum_{j=0..4} (dt/6) r_j
        #   d_mid = 0.4 * d5
        #   d9 = 0.4 * d5 + sum_{j=5..8} (dt/6) r_j
        #   g_next = g0 + (0.36 * d5 + 0.6 * d9 + 0.1 * dt * r9)
        sixth = jnp.asarray(1.0 / 6.0, dtype=r_dtype)
        tenth = jnp.asarray(0.1, dtype=r_dtype)

        def _euler_k10(
            carry: tuple[jax.Array, jax.Array, Fields],
        ) -> tuple[jax.Array, jax.Array, Fields]:
            dq, gq, fq = carry
            rq = _rhs(gq, fq, exb_cur, d_lin=dealias_kz, d_nl=False)
            dq_next = dq + (sixth * dt_r) * rq
            gq_next = g0 + dq_next
            fq_next = _solve_stage_fields(gq_next, gk_op, exb_state=exb_cur, apply_kz_dealias=True)
            return dq_next, gq_next, fq_next

        d_acc = jnp.zeros_like(g0)
        g_q1, f_q1 = g0, f0
        for _ in range(5):
            d_acc, g_q1, f_q1 = _euler_k10((d_acc, g_q1, f_q1))
        d5 = d_acc

        d_acc = jnp.asarray(0.4, dtype=r_dtype) * d5
        g_q1 = g0 + d_acc
        f_q1 = _solve_stage_fields(g_q1, gk_op, exb_state=exb_cur, apply_kz_dealias=True)

        for _ in range(4):
            d_acc, g_q1, f_q1 = _euler_k10((d_acc, g_q1, f_q1))
        d9 = d_acc

        # In stage 10 of ts_k10.cu (L121, L129), dealias_kz is applied to both nlps and linear rhs
        r_last = _rhs(g_q1, f_q1, exb_cur, d_lin=dealias_kz, d_nl=dealias_kz)
        g_next = g0 + (
            jnp.asarray(0.36, dtype=r_dtype) * d5
            + jnp.asarray(0.6, dtype=r_dtype) * d9
            + (tenth * dt_r) * r_last
        )
        apply_final_kz_dealias = True

    else:
        raise ValueError(f"Unsupported scheme={sk!r}. Supported: {sorted(_SCHEME_CFL_FAC.keys())}.")

    if gk_op.forcing_init and gk_op.forcing_op is not None:
        g_next = apply_forcing(g_next, gk_op.forcing_op, dt_r, key, step_idx=step_idx)

    g_next = mask_g(g_next, gk_op.grids)
    if enforce_reality_cond:
        g_next = enforce_reality(g_next, gk_op.grids)

    f_next = _solve_stage_fields(
        g_next, gk_op, exb_state=exb_cur, apply_kz_dealias=apply_final_kz_dealias
    )
    return StepResult(
        G=g_next.astype(c_dtype),
        fields=f_next.astype(c_dtype),
        dt=dt_f64.astype(t_dtype),
        exb_state=exb_cur,
    )


def _resolve_initial_exb_state(
    gk_op: GyrokineticOperator,
    exb_state: ExBShearState | None,
    r_dtype: jnp.dtype,
) -> ExBShearState | None:
    """Resolve and cast initial :class:`ExBShearState` (auto-enabled when ``gk_op.exb_shear``)."""
    if exb_state is not None:
        return exb_state.astype(r_dtype)
    if gk_op.exb_shear and gk_op.exb_init is not None:
        return gk_op.exb_init.astype(r_dtype)
    return None


def init_integrator_state(
    G: jax.Array,
    gk_op: GyrokineticOperator,
    *,
    time: float | jax.Array = 0.0,
    dt: float | jax.Array | None = None,
    enforce_reality_cond: bool = False,
    exb_state: ExBShearState | None = None,
    key: jax.Array | None = None,
) -> IntegratorState:
    """Create initial :class:`IntegratorState` with consistent static shapes and dtypes."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    t_dtype = _time_dtype(r_dtype)
    g0 = mask_g(G.astype(c_dtype), gk_op.grids)
    if enforce_reality_cond:
        g0 = enforce_reality(g0, gk_op.grids)
    exb0 = _resolve_initial_exb_state(gk_op, exb_state, r_dtype)
    # Initial fieldSolve in run_gx.cu does not apply dealias_kz
    f0 = _solve_stage_fields(g0, gk_op, exb_state=exb0, apply_kz_dealias=False).astype(c_dtype)
    dt0 = (
        compute_cfl_dt(f0, gk_op, exb_state=exb0) if dt is None else jnp.asarray(dt, dtype=t_dtype)
    )
    key0 = jax.random.PRNGKey(gk_op.random_seed) if key is None else key
    return IntegratorState(
        G=g0,
        fields=f0,
        time=jnp.asarray(time, dtype=t_dtype),
        dt=dt0.astype(t_dtype),
        step=jnp.asarray(0, dtype=jnp.int32),
        exb_state=exb0,
        key=key0,
    )


def step_integrator_state(
    state: IntegratorState,
    gk_op: GyrokineticOperator,
    *,
    dt_override: float | jax.Array | None = None,
    scheme: str | None = None,
    enforce_reality_cond: bool = False,
) -> IntegratorState:
    """Advance :class:`IntegratorState` by one step preserving exact carry structure & dtypes."""
    c_dtype, r_dtype = _complex_and_real_dtypes(state.G)
    t_dtype = _time_dtype(r_dtype)
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    exb_in = state.exb_state.astype(r_dtype) if state.exb_state is not None else None

    if dt_override is not None:
        dt_step = jnp.asarray(dt_override, dtype=t_dtype)
    elif gk_op.fixed_dt and _scheme_honors_fixed_dt(sk):
        dt_step = state.dt.astype(t_dtype)
    else:
        dt_step = compute_cfl_dt(state.fields, gk_op, exb_state=exb_in, scheme=sk)

    next_key, step_key = jax.random.split(state.key)
    g_next, f_next, dt_used, exb_next = step_gyrokinetic(
        state.G,
        state.fields,
        gk_op,
        dt=dt_step,
        scheme=sk,
        exb_state=exb_in,
        enforce_reality_cond=enforce_reality_cond,
        key=step_key,
        step_idx=state.step,
    )

    return IntegratorState(
        G=g_next.astype(c_dtype),
        fields=f_next.astype(c_dtype),
        time=(state.time.astype(t_dtype) + dt_used.astype(t_dtype)),
        dt=dt_used.astype(t_dtype),
        step=(state.step + jnp.int32(1)).astype(jnp.int32),
        exb_state=exb_next,
        key=next_key,
    )


def _compute_wg_scalar(G: jax.Array, gk_op: GyrokineticOperator) -> jax.Array:
    """Compute total free-energy distribution summand ``Wg`` (``device_funcs.cu:L1409-1429``)."""
    _, r_dtype = _complex_and_real_dtypes(G)
    fac_ky = jnp.where(jnp.arange(gk_op.grids.naky) == 0, 1.0, 2.0).astype(r_dtype)[
        None, None, None, None, :, None
    ]
    vol = gk_op.geo.vol_fac.astype(r_dtype)[None, None, None, :, None, None]
    nt = gk_op.linear_op.solver.sp_nt.astype(r_dtype)[:, None, None, None, None, None]
    g2 = jnp.real(G) ** 2 + jnp.imag(G) ** 2
    mask6 = gk_op.grids.active_mask[None, None, None, None, :, :]
    summand = jnp.where(mask6, 0.5 * fac_ky * vol * nt * g2, jnp.zeros_like(g2))
    return jnp.sum(summand)


def integrate_scan(
    G: jax.Array,
    fields: Fields | None,
    gk_op: GyrokineticOperator,
    nsteps: int,
    dt: float | jax.Array | None = None,
    *,
    scheme: str | None = None,
    save_every: int = 1,
    save_g_history: bool = False,
    enforce_reality_cond: bool = False,
    exb_state: ExBShearState | None = None,
    key: jax.Array | None = None,
    remat: bool = False,
) -> TrajectoryResult:
    """Integrate ``nsteps`` fixed-``dt`` steps via ``jax.lax.scan`` (reverse-mode AD safe).

    Parameters
    ----------
    G:
        Initial distribution function of shape ``(nspecies, nhermite, nlaguerre, nz, naky, nakx)``.
    fields:
        Initial electromagnetic fields (or ``None`` to solve from ``G``).
    gk_op:
        Precomputed :class:`GyrokineticOperator`.
    nsteps:
        Total number of timesteps (must be ``>= 1`` and divisible by ``save_every``).
    dt:
        Fixed timestep size (defaults to ``gk_op.dt`` if ``fixed_dt=True`` else initial CFL ``dt``).
    save_every:
        Chunk size per outer ``scan`` iteration (``1 <= save_every <= nsteps``).
    remat:
        When ``True``, wraps each inner timestep in ``jax.checkpoint`` so reverse-mode ``jax.grad``
        recomputes stage residuals during backward pass instead of storing all steps in HBM.
    """
    if nsteps < 1:
        raise ValueError(f"nsteps={nsteps} must be >= 1.")
    if save_every < 1 or nsteps % save_every != 0:
        raise ValueError(f"nsteps={nsteps} must be divisible by save_every={save_every}.")

    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    t_dtype = _time_dtype(r_dtype)
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    g0 = mask_g(G.astype(c_dtype), gk_op.grids)
    if enforce_reality_cond:
        g0 = enforce_reality(g0, gk_op.grids)
    exb0 = _resolve_initial_exb_state(gk_op, exb_state, r_dtype)
    f0 = (
        _solve_stage_fields(g0, gk_op, exb_state=exb0, apply_kz_dealias=False)
        if fields is None
        else fields.astype(c_dtype)
    )
    dt_val = (
        jnp.asarray(dt, dtype=t_dtype)
        if dt is not None
        else (
            jnp.asarray(gk_op.dt, dtype=t_dtype)
            if gk_op.fixed_dt
            else compute_cfl_dt(f0, gk_op, exb_state=exb0, scheme=sk)
        )
    )
    key0 = jax.random.PRNGKey(gk_op.random_seed) if key is None else key

    init_state = IntegratorState(
        G=g0,
        fields=f0,
        time=jnp.asarray(0.0, dtype=t_dtype),
        dt=dt_val,
        step=jnp.asarray(0, dtype=jnp.int32),
        exb_state=exb0,
        key=key0,
    )

    n_chunks = nsteps // save_every

    def _raw_one_step(st: IntegratorState, _unused: None) -> tuple[IntegratorState, None]:
        st_next = step_integrator_state(
            st,
            gk_op,
            dt_override=st.dt,
            scheme=sk,
            enforce_reality_cond=enforce_reality_cond,
        )
        return st_next, None

    _one_step = jax.checkpoint(_raw_one_step) if remat else _raw_one_step

    def _chunk_step(st: IntegratorState, _unused: None):
        if save_every == 1:
            st_end, _ = _one_step(st, None)
        else:
            st_end, _ = jax.lax.scan(_one_step, st, None, length=save_every)
        phi2_ky = compute_phi2_ky(st_end.fields.phi, gk_op.grids, gk_op.geo)
        wg_val = _compute_wg_scalar(st_end.G, gk_op)
        g_hist_item = st_end.G if save_g_history else None
        return st_end, (st_end.time, st_end.dt, phi2_ky, wg_val, g_hist_item)

    final_state, (t_hist, dt_hist, phi2_hist, wg_hist, g_hist) = jax.lax.scan(
        _chunk_step,
        init_state,
        None,
        length=n_chunks,
    )
    return TrajectoryResult(
        G=final_state.G,
        fields=final_state.fields,
        time=final_state.time,
        dt=final_state.dt,
        steps=final_state.step,
        t_history=t_hist,
        dt_history=dt_hist,
        phi2_history=phi2_hist,
        wg_history=wg_hist,
        G_history=g_hist,
        exb_state=final_state.exb_state,
    )


def integrate_while_loop(
    G: jax.Array,
    fields: Fields | None,
    gk_op: GyrokineticOperator,
    *,
    t_max: float | jax.Array | None = None,
    max_steps: int | jax.Array | None = None,
    dt_init: float | jax.Array | None = None,
    time_init: float | jax.Array = 0.0,
    scheme: str | None = None,
    clamp_to_t_max: bool = False,
    enforce_reality_cond: bool = False,
    exb_state: ExBShearState | None = None,
    key: jax.Array | None = None,
) -> IntegratorState:
    """Integrate with adaptive CFL ``dt`` up to ``t_max`` or ``max_steps`` via ``lax.while_loop``.

    Ported from the main timestep loop in ``gx/src/run_gx.cu:L128-148``:
    ``while (counter < pars->nstep && time < pars->t_max)``.
    By default (``clamp_to_t_max=False``), matches C++ ``run_gx.cu`` where the final step uses the
    unclamped ``dt`` (and may overshoot ``t_max``). When ``clamp_to_t_max=True`` (opt-in), the
    final step's ``dt`` is clamped to ``t_max - time`` so the trajectory lands on ``t_max``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    t_dtype = _time_dtype(r_dtype)
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    g0 = mask_g(G.astype(c_dtype), gk_op.grids)
    if enforce_reality_cond:
        g0 = enforce_reality(g0, gk_op.grids)
    exb0 = _resolve_initial_exb_state(gk_op, exb_state, r_dtype)
    f0 = (
        _solve_stage_fields(g0, gk_op, exb_state=exb0, apply_kz_dealias=False)
        if fields is None
        else fields.astype(c_dtype)
    )

    t0 = jnp.asarray(time_init, dtype=t_dtype)
    t_end = jnp.asarray(gk_op.t_max if t_max is None else t_max, dtype=t_dtype)
    n_max = jnp.asarray(gk_op.nstep if max_steps is None else max_steps, dtype=jnp.int32)
    dt0 = (
        jnp.asarray(dt_init, dtype=t_dtype)
        if dt_init is not None
        else compute_cfl_dt(f0, gk_op, exb_state=exb0, scheme=sk)
    )
    key0 = jax.random.PRNGKey(gk_op.random_seed) if key is None else key

    init_carry = IntegratorState(
        G=g0,
        fields=f0,
        time=t0,
        dt=dt0,
        step=jnp.asarray(0, dtype=jnp.int32),
        exb_state=exb0,
        key=key0,
    )

    def _cond(st: IntegratorState) -> jax.Array:
        return (st.step < n_max) & (st.time < t_end)

    def _body(st: IntegratorState) -> IntegratorState:
        dt_cfl = (
            st.dt
            if (gk_op.fixed_dt and _scheme_honors_fixed_dt(sk))
            else compute_cfl_dt(st.fields, gk_op, exb_state=st.exb_state, scheme=sk)
        )
        rem = jnp.maximum(t_end - st.time, jnp.zeros((), dtype=t_dtype))
        if clamp_to_t_max:
            will_hit_end = (st.time + dt_cfl) >= t_end
            dt_step = jnp.where(will_hit_end, rem, dt_cfl)
        else:
            will_hit_end = jnp.asarray(False)
            dt_step = dt_cfl

        st_next = step_integrator_state(
            st,
            gk_op,
            dt_override=dt_step,
            scheme=sk,
            enforce_reality_cond=enforce_reality_cond,
        )
        exact_t = jnp.where(will_hit_end, t_end, st_next.time).astype(t_dtype)
        return IntegratorState(
            G=st_next.G.astype(c_dtype),
            fields=st_next.fields.astype(c_dtype),
            time=exact_t,
            dt=dt_step.astype(t_dtype),
            step=st_next.step.astype(jnp.int32),
            exb_state=st_next.exb_state,
            key=st_next.key,
        )

    return jax.lax.while_loop(_cond, _body, init_carry)


def _resolve_t_limit(
    t_max: float | jax.Array, *, dtype: jnp.dtype | type | None = None
) -> jax.Array:
    """Compute stopping threshold ``t_max - eps_t`` without ``inf - inf`` NaNs."""
    if dtype is not None:
        t_dtype = jnp.dtype(dtype)
    elif isinstance(t_max, jax.Array) and jnp.issubdtype(t_max.dtype, jnp.floating):
        t_dtype = jnp.dtype(t_max.dtype)
    else:
        t_dtype = _time_dtype(jnp.float64)
    t_arr = jnp.asarray(t_max, dtype=t_dtype)
    is_fin = jnp.isfinite(t_arr)
    safe_t = jnp.where(is_fin, t_arr, jnp.asarray(0.0, dtype=t_dtype))
    eps_rel = 1.0e-6 if t_dtype == jnp.float32 else 1.0e-11
    eps = jnp.asarray(eps_rel, dtype=t_dtype) * jnp.maximum(
        jnp.abs(safe_t), jnp.asarray(1.0e-30, dtype=t_dtype)
    )
    return jnp.where(is_fin, t_arr - eps, t_arr)


def _update_diag_carry(
    st: IntegratorState,
    diag_state: DiagnosticsState,
    *,
    diag_nwrite: jax.Array,
    t_limit: jax.Array,
) -> DiagnosticsState:
    """Apply C++ ``Diagnostics_GK::loop`` pre-step ``save_old`` check inside a device loop."""
    t_dtype = st.time.dtype
    should_save = ((st.step % diag_nwrite) == jnp.int32(0)) | (
        (st.time.astype(t_dtype) + st.dt.astype(t_dtype)) > t_limit.astype(t_dtype)
    )
    g_old_next = jnp.where(should_save, st.G, diag_state.G_old)
    f_old_next = jax.tree_util.tree_map(
        lambda a, b: jnp.where(should_save, a, b),
        st.fields,
        diag_state.fields_old,
    )
    t_old_next = jnp.where(
        should_save, st.time.astype(t_dtype), diag_state.time_old.astype(t_dtype)
    )
    has_old_next = (diag_state.has_old | should_save).astype(jnp.bool_)
    return dataclasses.replace(
        diag_state,
        G_old=g_old_next,
        fields_old=f_old_next,
        time_old=t_old_next,
        has_old=has_old_next,
    )


def _run_chunk_while(
    state: IntegratorState,
    max_steps: jax.Array,
    t_limit: jax.Array,
    *,
    step_fn: Callable[[IntegratorState], IntegratorState],
    diag_state: DiagnosticsState | None = None,
    diag_nwrite: jax.Array | None = None,
) -> ChunkResult:
    """Advance up to ``max_steps`` inside a single ``jax.lax.while_loop``."""
    c_dtype, _ = _complex_and_real_dtypes(state.G)
    t_dtype = state.time.dtype
    t_lim = t_limit.astype(t_dtype)
    max_steps_i32 = jnp.asarray(max_steps, dtype=jnp.int32)
    init_carry = (
        state,
        state.G.astype(c_dtype),
        state.fields.astype(c_dtype),
        state.time.astype(t_dtype),
        jnp.asarray(0, dtype=jnp.int32),
        diag_state,
    )

    def _cond(
        carry: tuple[
            IntegratorState,
            jax.Array,
            Fields,
            jax.Array,
            jax.Array,
            DiagnosticsState | None,
        ],
    ) -> jax.Array:
        st, _g_prev, _f_prev, _t_prev, steps_done, _d_st = carry
        return (steps_done < max_steps_i32) & (st.time.astype(t_dtype) < t_lim)

    def _body(
        carry: tuple[
            IntegratorState,
            jax.Array,
            Fields,
            jax.Array,
            jax.Array,
            DiagnosticsState | None,
        ],
    ) -> tuple[
        IntegratorState,
        jax.Array,
        Fields,
        jax.Array,
        jax.Array,
        DiagnosticsState | None,
    ]:
        st, _g_prev, _f_prev, _t_prev, steps_done, d_st = carry
        if d_st is not None and diag_nwrite is not None:
            d_next: DiagnosticsState | None = _update_diag_carry(
                st, d_st, diag_nwrite=diag_nwrite, t_limit=t_lim
            )
        else:
            d_next = d_st
        st_next = step_fn(st)
        return (
            st_next,
            st.G.astype(c_dtype),
            st.fields.astype(c_dtype),
            st.time.astype(t_dtype),
            (steps_done + jnp.int32(1)).astype(jnp.int32),
            d_next,
        )

    final_st, g_prev, f_prev, t_prev, steps_done, d_end = jax.lax.while_loop(
        _cond, _body, init_carry
    )
    return ChunkResult(
        state=final_st,
        G_prev=g_prev,
        fields_prev=f_prev,
        time_prev=t_prev,
        steps_done=steps_done,
        diag_state=d_end,
    )


def _run_chunk_scan(
    state: IntegratorState,
    fixed_len: int,
    t_limit: jax.Array,
    *,
    step_fn: Callable[[IntegratorState], IntegratorState],
    diag_state: DiagnosticsState | None = None,
    diag_nwrite: jax.Array | None = None,
) -> ChunkResult:
    """Advance exactly ``fixed_len`` steps inside ``jax.lax.scan`` (fixed-``dt`` fast path)."""
    c_dtype, _ = _complex_and_real_dtypes(state.G)
    t_dtype = state.time.dtype
    t_lim = t_limit.astype(t_dtype)
    init_carry = (
        state,
        state.G.astype(c_dtype),
        state.fields.astype(c_dtype),
        state.time.astype(t_dtype),
        diag_state,
    )

    def _scan_body(
        carry: tuple[
            IntegratorState,
            jax.Array,
            Fields,
            jax.Array,
            DiagnosticsState | None,
        ],
        _unused: None,
    ) -> tuple[
        tuple[
            IntegratorState,
            jax.Array,
            Fields,
            jax.Array,
            DiagnosticsState | None,
        ],
        None,
    ]:
        st, _g_prev, _f_prev, _t_prev, d_st = carry
        if d_st is not None and diag_nwrite is not None:
            d_next: DiagnosticsState | None = _update_diag_carry(
                st, d_st, diag_nwrite=diag_nwrite, t_limit=t_lim
            )
        else:
            d_next = d_st
        st_next = step_fn(st)
        return (
            st_next,
            st.G.astype(c_dtype),
            st.fields.astype(c_dtype),
            st.time.astype(t_dtype),
            d_next,
        ), None

    (final_st, g_prev, f_prev, t_prev, d_end), _ = jax.lax.scan(
        _scan_body,
        init_carry,
        None,
        length=fixed_len,
    )
    return ChunkResult(
        state=final_st,
        G_prev=g_prev,
        fields_prev=f_prev,
        time_prev=t_prev,
        steps_done=jnp.asarray(fixed_len, dtype=jnp.int32),
        diag_state=d_end,
    )


def _is_truly_fixed_dt(
    gk_op: GyrokineticOperator,
    scheme: str,
    dt_override: float | jax.Array | None,
) -> bool:
    """Return True when timestep ``dt`` is fixed across steps in a chunk."""
    return _scheme_honors_fixed_dt(scheme) and (dt_override is not None or bool(gk_op.fixed_dt))


def advance_integrator_chunk(
    state: IntegratorState,
    gk_op: GyrokineticOperator,
    max_steps: int | jax.Array,
    *,
    dt_override: float | jax.Array | None = None,
    scheme: str | None = None,
    t_max: float | jax.Array | None = None,
    use_scan: bool = False,
    enforce_reality_cond: bool = False,
    diag_state: DiagnosticsState | None = None,
    diag_nwrite: int | jax.Array | None = None,
) -> ChunkResult:
    """Advance ``state`` by up to ``max_steps`` on device without host-device syncs.

    Uses ``jax.lax.while_loop`` by default (supporting arbitrary dynamic ``max_steps``,
    adaptive CFL ``dt``, and early mid-chunk stopping at ``t_max - eps_t``). When
    ``use_scan=True`` with a fixed-``dt`` scheme and concrete ``max_steps >= 1``,
    uses ``jax.lax.scan`` whenever ``state.time + (max_steps - 1) * dt < t_limit``
    (falling back on device via ``lax.cond`` to ``while_loop`` near ``t_max``;
    FP-identical, not bit-identical vs ``while_loop`` due to ~1 ULP XLA scheduling).
    """
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    t_dtype = state.time.dtype
    t_max_eff = gk_op.t_max if t_max is None else t_max
    t_limit = _resolve_t_limit(t_max_eff, dtype=t_dtype)
    diag_nw = (
        None if diag_nwrite is None else jnp.asarray(max(1, int(diag_nwrite)), dtype=jnp.int32)
    )

    def _step_fn(st: IntegratorState) -> IntegratorState:
        return step_integrator_state(
            st,
            gk_op,
            dt_override=dt_override,
            scheme=sk,
            enforce_reality_cond=enforce_reality_cond,
        )

    if use_scan:
        if not _is_truly_fixed_dt(gk_op, sk, dt_override):
            raise ValueError(
                f"use_scan=True requires a fixed-dt scheme honoring fixed_dt or dt_override; "
                f"got scheme={sk!r}, fixed_dt={gk_op.fixed_dt}, dt_override={dt_override}."
            )
        if isinstance(max_steps, jax.Array) or int(max_steps) < 1:
            raise ValueError(
                f"use_scan=True requires concrete positive int max_steps, got {max_steps!r}."
            )
        fixed_len = int(max_steps)
        if not isinstance(t_max_eff, jax.Array) and not math.isfinite(float(t_max_eff)):
            return _run_chunk_scan(
                state,
                fixed_len,
                t_limit,
                step_fn=_step_fn,
                diag_state=diag_state,
                diag_nwrite=diag_nw,
            )
        dt_eff = (
            jnp.asarray(dt_override, dtype=t_dtype)
            if dt_override is not None
            else state.dt.astype(t_dtype)
        )
        slack_rel = 1.0e-6 if t_dtype == jnp.float32 else 1.0e-12
        slack = jnp.asarray(slack_rel * float(fixed_len), dtype=t_dtype) * jnp.maximum(
            jnp.abs(t_limit), jnp.asarray(1.0, dtype=t_dtype)
        )
        all_fit = (
            state.time.astype(t_dtype) + jnp.asarray(fixed_len - 1, dtype=t_dtype) * dt_eff
        ) < (t_limit - slack)
        return jax.lax.cond(
            all_fit,
            lambda: _run_chunk_scan(
                state,
                fixed_len,
                t_limit,
                step_fn=_step_fn,
                diag_state=diag_state,
                diag_nwrite=diag_nw,
            ),
            lambda: _run_chunk_while(
                state,
                jnp.asarray(fixed_len, dtype=jnp.int32),
                t_limit,
                step_fn=_step_fn,
                diag_state=diag_state,
                diag_nwrite=diag_nw,
            ),
        )

    return _run_chunk_while(
        state,
        jnp.asarray(max_steps, dtype=jnp.int32),
        t_limit,
        step_fn=_step_fn,
        diag_state=diag_state,
        diag_nwrite=diag_nw,
    )


def make_chunk_stepper(
    gk_op: GyrokineticOperator,
    *,
    dt_override: float | jax.Array | None = None,
    scheme: str | None = None,
    t_max: float | jax.Array | None = None,
    fixed_chunk_len: int | None = None,
    enforce_reality_cond: bool = False,
    diag_nwrite: int | None = None,
) -> Callable[..., ChunkResult]:
    """Compile a reusable single-trace on-device chunk stepper for ``gk_op``.

    Compiles at most:
      1. One ``jax.lax.while_loop`` kernel accepting dynamic ``(max_steps, t_max)``
         for adaptive CFL stepping, initial/irregular chunks, and ``t_max`` tail chunks.
      2. One ``jax.lax.scan`` fast-path kernel for uniform ``fixed_chunk_len`` blocks
         when ``fixed_dt`` is truly fixed (``_scheme_honors_fixed_dt(scheme)`` and
         ``(dt_override is not None or gk_op.fixed_dt)``; FP-identical, not
         bit-identical vs ``while_loop``).
    """
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    op_t_dtype = _time_dtype(gk_op.geo.bmag.dtype)
    default_t_max = gk_op.t_max if t_max is None else t_max
    is_fixed_dt = _is_truly_fixed_dt(gk_op, sk, dt_override)
    diag_nw_val = None if diag_nwrite is None else max(1, int(diag_nwrite))
    steady_len = (
        int(fixed_chunk_len) if (fixed_chunk_len is not None and fixed_chunk_len >= 1) else None
    )

    trace_counts: dict[str, int] = {"while": 0, "scan": 0}

    def _diag_nw_arr() -> jax.Array | None:
        return None if diag_nw_val is None else jnp.asarray(diag_nw_val, dtype=jnp.int32)

    def _step_with_dt(st: IntegratorState, dt_in: jax.Array | None) -> IntegratorState:
        return step_integrator_state(
            st,
            gk_op,
            dt_override=dt_in,
            scheme=sk,
            enforce_reality_cond=enforce_reality_cond,
        )

    @jax.jit
    def _while_jit(
        st: IntegratorState,
        max_steps_arr: jax.Array,
        t_max_arr: jax.Array,
        dt_in: jax.Array | None,
        d_st: DiagnosticsState | None,
    ) -> ChunkResult:
        trace_counts["while"] += 1
        t_limit = _resolve_t_limit(t_max_arr, dtype=st.time.dtype)
        return _run_chunk_while(
            st,
            max_steps_arr,
            t_limit,
            step_fn=lambda s: _step_with_dt(s, dt_in),
            diag_state=d_st,
            diag_nwrite=_diag_nw_arr(),
        )

    _scan_pure_jit: Callable[..., ChunkResult] | None = None
    _scan_cond_jit: Callable[..., ChunkResult] | None = None
    if is_fixed_dt and steady_len is not None:

        @jax.jit
        def _scan_pure_jit_fn(
            st: IntegratorState,
            t_max_arr: jax.Array,
            dt_in: jax.Array | None,
            d_st: DiagnosticsState | None,
        ) -> ChunkResult:
            trace_counts["scan"] += 1
            t_limit = _resolve_t_limit(t_max_arr, dtype=st.time.dtype)
            return _run_chunk_scan(
                st,
                steady_len,
                t_limit,
                step_fn=lambda s: _step_with_dt(s, dt_in),
                diag_state=d_st,
                diag_nwrite=_diag_nw_arr(),
            )

        @jax.jit
        def _scan_cond_jit_fn(
            st: IntegratorState,
            t_max_arr: jax.Array,
            dt_in: jax.Array | None,
            d_st: DiagnosticsState | None,
        ) -> ChunkResult:
            trace_counts["scan"] += 1
            t_dt = st.time.dtype
            t_limit = _resolve_t_limit(t_max_arr, dtype=t_dt)
            dt_eff = dt_in.astype(t_dt) if dt_in is not None else st.dt.astype(t_dt)
            slack_rel = 1.0e-6 if t_dt == jnp.float32 else 1.0e-12
            slack = jnp.asarray(slack_rel * float(steady_len), dtype=t_dt) * jnp.maximum(
                jnp.abs(t_limit), jnp.asarray(1.0, dtype=t_dt)
            )
            t_last_step_start = (
                st.time.astype(t_dt) + jnp.asarray(steady_len - 1, dtype=t_dt) * dt_eff
            )
            all_fit = t_last_step_start < (t_limit - slack)
            return jax.lax.cond(
                all_fit,
                lambda: _run_chunk_scan(
                    st,
                    steady_len,
                    t_limit,
                    step_fn=lambda s: _step_with_dt(s, dt_in),
                    diag_state=d_st,
                    diag_nwrite=_diag_nw_arr(),
                ),
                lambda: _run_chunk_while(
                    st,
                    jnp.asarray(steady_len, dtype=jnp.int32),
                    t_limit,
                    step_fn=lambda s: _step_with_dt(s, dt_in),
                    diag_state=d_st,
                    diag_nwrite=_diag_nw_arr(),
                ),
            )

        _scan_pure_jit = _scan_pure_jit_fn
        _scan_cond_jit = _scan_cond_jit_fn

    def _stepper(
        state: IntegratorState,
        max_steps: int | jax.Array,
        *,
        t_max: float | jax.Array | None = None,
        diag_state: DiagnosticsState | None = None,
        safe_for_scan: bool | None = None,
    ) -> ChunkResult:
        t_dt = state.time.dtype if isinstance(state.time, jax.Array) else op_t_dtype
        t_max_use = default_t_max if t_max is None else t_max
        t_max_arr = jnp.asarray(t_max_use, dtype=t_dt)
        dt_arr = None if dt_override is None else jnp.asarray(dt_override, dtype=t_dt)
        is_steady_len = (
            steady_len is not None
            and not isinstance(max_steps, jax.Array)
            and int(max_steps) == steady_len
        )
        if _scan_pure_jit is not None and is_steady_len:
            if safe_for_scan is True or (
                safe_for_scan is None
                and not isinstance(t_max_use, jax.Array)
                and not math.isfinite(float(t_max_use))
            ):
                return _scan_pure_jit(state, t_max_arr, dt_arr, diag_state)
            if safe_for_scan is None and _scan_cond_jit is not None:
                return _scan_cond_jit(state, t_max_arr, dt_arr, diag_state)
        return _while_jit(
            state,
            jnp.asarray(max_steps, dtype=jnp.int32),
            t_max_arr,
            dt_arr,
            diag_state,
        )

    _stepper.trace_counts = trace_counts  # type: ignore[attr-defined]
    _stepper.is_fixed_dt = is_fixed_dt  # type: ignore[attr-defined]
    _stepper.fixed_chunk_len = steady_len  # type: ignore[attr-defined]
    return _stepper


__all__ = [
    "ChunkResult",
    "GyrokineticOperator",
    "IntegratorState",
    "StepResult",
    "TrajectoryResult",
    "advance_integrator_chunk",
    "compute_cfl_dt",
    "compute_max_frequency",
    "get_scheme_cfl_fac",
    "gyrokinetic_rhs",
    "init_gyrokinetic_operator",
    "init_integrator_state",
    "integrate_scan",
    "integrate_while_loop",
    "make_chunk_stepper",
    "step_gyrokinetic",
    "step_integrator_state",
]
