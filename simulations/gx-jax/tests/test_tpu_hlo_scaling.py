"""TPU StableHLO Collective Audit, Corner-Mesh Stress & Scaling Tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_4device_subprocess(body: str) -> None:
    """Execute *body* in a fresh 4-device host CPU JAX subprocess."""
    script = (
        'import os\nos.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"\n' + body
    )
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_pp}" if existing_pp else str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"4-device subprocess failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_main_process_keeps_single_host_device() -> None:
    """Ensure 4-device subprocess tests do not mutate main pytest device count."""
    assert jax.device_count() == 1


def test_stablehlo_collectives_precision_and_zero_f64_hot_loop() -> None:
    """Audit StableHLO collectives, precision=highest, and zero f64 array ops."""
    _run_4device_subprocess(
        """
import ast
import re
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gx_jax import (
    SpeciesConfig,
    configure_tpu_precision,
    create_device_mesh,
    init_diagnostics_state,
    init_geometry,
    init_gyrokinetic_operator,
    load_config,
    shard_g_state,
    sharded_compute_diagnostics,
    sharded_linear_rhs,
    sharded_nonlinear_rhs,
    sharded_solve_fields,
    step_sharded_gyrokinetic,
)

assert jax.device_count() == 4
configure_tpu_precision()
assert jax.config.jax_default_matmul_precision == "highest"

ARRAY_F64_RE = re.compile(r"tensor<\\d+x[^>\\n]*(?:complex<f64>|f64)>")
ALL_GATHER_SHAPE_RE = re.compile(r"all_gather.*->\\s*tensor<([^>]+)>")
PERMUTE_PAIRS_RE = re.compile(r"source_target_pairs\\s*=\\s*dense<(\\[.*?\\])>")

def _assert_no_6d_all_gather(hlo_txt: str, label: str, allow_any: bool = False) -> None:
    lines = [ln for ln in hlo_txt.splitlines() if "all_gather" in ln]
    if not allow_any:
        assert not lines, f"{label}: unexpected all_gather ops:\\n" + "\\n".join(lines)
        return
    for ln in lines:
        m = ALL_GATHER_SHAPE_RE.search(ln)
        assert m is not None, f"{label}: unparsed all_gather line: {ln}"
        dims = [d for d in m.group(1).split("x") if d.isdigit()]
        assert len(dims) < 6, f"{label}: forbidden 6D all_gather in {ln}"

def _assert_highest_precision_no_array_f64(hlo_txt: str, label: str) -> None:
    f64_arrays = ARRAY_F64_RE.findall(hlo_txt)
    assert not f64_arrays, f"{label}: unexpected array f64 ops in c64 IR: {f64_arrays[:5]}"
    assert "precision = [DEFAULT" not in hlo_txt, f"{label}: found DEFAULT matmul precision"
    if "stablehlo.dot_general" in hlo_txt:
        assert "precision = [HIGHEST, HIGHEST]" in hlo_txt, (
            f"{label}: missing HIGHEST matmul precision in dot_general"
        )

base = load_config(Path("benchmarks/linear/KBM/kbm_miller.in"))
sp_coll = tuple(
    SpeciesConfig.from_raw(
        z=sp.z, mass=sp.mass, dens=sp.dens, temp=sp.temp,
        tprim=sp.tprim, fprim=sp.fprim,
        nu_ss=0.05 if i == 0 else 0.12,
        stype="ion" if sp.type_id == 0 else "electron",
        beta=base.beta, long_wavelength_gk=base.long_wavelength_gk,
    )
    for i, sp in enumerate(base.species)
)
cfg_em = replace(
    base, species=sp_coll, fphi=1.0, fapar=1.0, fbpar=1.0,
    nhermite=4, nlaguerre=2, nx=4, ny=4, nperiod=1, zp=1, ntheta_in=8, nz=8,
    nonlinear_mode=True, linear=False, closure_model="beer4+2",
    collisions=True, coll_conservation=True, ei_colls=True,
    hypercollisions_const=True, hypercollisions_kz=True, hyper=True,
    exb_shear=True, exb_shear_phase=True, g_exb=0.15,
    forcing_init=True, forcing_type="Kz", forcing_amp=0.02, forcing_index=1,
)
cfg_em, gr_em, geo_em = init_geometry(cfg_em)
op_em32 = init_gyrokinetic_operator(cfg_em, gr_em, geo_em, dtype=jnp.float32)

rng = np.random.default_rng(17)
g_raw = (
    rng.standard_normal(gr_em.g_shape) + 1j * rng.standard_normal(gr_em.g_shape)
) * 1e-2
g_raw[:, :, :, :, 0, 0] = 0.0
g_c64 = jnp.asarray(g_raw, dtype=jnp.complex64)

# 1. (2, 2) EM collectives: solve_fields, nonlinear_rhs (fapar=0 vs 1), step_sharded_gk
m22 = create_device_mesh((2, 2), nspecies=cfg_em.nspecies, nhermite=cfg_em.nhermite)
g22 = shard_g_state(g_c64, m22)

jf22 = jax.jit(lambda g: sharded_solve_fields(g, op_em32.linear_op.solver, m22))
hlo_f22 = str(jf22.lower(g22).compiler_ir(dialect="stablehlo"))
_assert_no_6d_all_gather(hlo_f22, "solve_fields(2,2)")
_assert_highest_precision_no_array_f64(hlo_f22, "solve_fields(2,2)")
f22_red = [ln for ln in hlo_f22.splitlines() if "all_reduce" in ln]
assert len(f22_red) >= 1
assert all(
    '#stablehlo.axis_ref<name = "sp">' in ln and '#stablehlo.axis_ref<name = "m">' in ln
    for ln in f22_red
)
f22 = jf22(g22)

# Electrostatic nonlinear_rhs (fapar=0) -> ZERO collectives
cfg_es = replace(cfg_em, fapar=0.0, fbpar=0.0)
op_es32 = init_gyrokinetic_operator(cfg_es, gr_em, geo_em, dtype=jnp.float32)
jnl_es = jax.jit(lambda g, f: sharded_nonlinear_rhs(g, f, op_es32.nonlinear_op, m22))
hlo_nl_es = str(jnl_es.lower(g22, f22).compiler_ir(dialect="stablehlo"))
for coll_op in ("all_gather", "all_reduce", "collective_permute"):
    assert coll_op not in hlo_nl_es, f"nonlinear_rhs(fapar=0) emitted {coll_op}"
_assert_highest_precision_no_array_f64(hlo_nl_es, "nonlinear_rhs(fapar=0)")

# Electromagnetic nonlinear_rhs (fapar=1) -> 2 ppermute (m+-1 flutter), 0 all_reduce
jnl_em = jax.jit(lambda g, f: sharded_nonlinear_rhs(g, f, op_em32.nonlinear_op, m22))
hlo_nl_em = str(jnl_em.lower(g22, f22).compiler_ir(dialect="stablehlo"))
_assert_no_6d_all_gather(hlo_nl_em, "nonlinear_rhs(fapar=1)")
assert "all_reduce" not in hlo_nl_em
assert len([ln for ln in hlo_nl_em.splitlines() if "collective_permute" in ln]) == 2
_assert_highest_precision_no_array_f64(hlo_nl_em, "nonlinear_rhs(fapar=1)")

# step_sharded_gyrokinetic (c64, rk4) -> zero all_gather, zero array f64 ops, HIGHEST
jstep = jax.jit(
    lambda g, f: step_sharded_gyrokinetic(g, f, op_em32, m22, dt=1e-4, scheme="rk4")
)
hlo_step = str(jstep.lower(g22, f22).compiler_ir(dialect="stablehlo"))
_assert_no_6d_all_gather(hlo_step, "step_sharded_gyrokinetic(2,2)")
_assert_highest_precision_no_array_f64(hlo_step, "step_sharded_gyrokinetic(2,2)")

# sharded_compute_diagnostics (2, 2, c128) -> zero 6D all_gather
op_em64 = init_gyrokinetic_operator(cfg_em, gr_em, geo_em, dtype=jnp.float64)
g22_64 = shard_g_state(jnp.asarray(g_raw, dtype=jnp.complex128), m22)
f22_64 = sharded_solve_fields(g22_64, op_em64.linear_op.solver, m22)
ds22 = init_diagnostics_state(g22_64, f22_64, time=0.0, has_old=True)
jdiag = jax.jit(
    lambda g, f, ds: sharded_compute_diagnostics(
        g, f, ds, gr_em, geo_em, op_em64.linear_op.solver, m22,
        time=1e-4, dt=1e-4, beta=float(cfg_em.beta),
    )
)
hlo_diag = str(jdiag.lower(g22_64, f22_64, ds22).compiler_ir(dialect="stablehlo"))
_assert_no_6d_all_gather(hlo_diag, "sharded_compute_diagnostics(2,2)", allow_any=True)

# 2. (1, 4) linear_rhs with nhermite=4 (nhermite_local==1) & nhermite=8 (==2)
#    Verify non-wrapping collective_permute pairs for both m+-1 and m+-2!
for nh_val, min_permutes in [(4, 4), (8, 2)]:
    cfg_m = replace(cfg_em, nhermite=nh_val)
    op_m32 = init_gyrokinetic_operator(cfg_m, gr_em, geo_em, dtype=jnp.float32)
    m14 = create_device_mesh((1, 4), nspecies=cfg_m.nspecies, nhermite=nh_val)
    sh = (gr_em.g_shape[0], nh_val) + gr_em.g_shape[2:]
    g_m_raw = (rng.standard_normal(sh) + 1j * rng.standard_normal(sh)) * 1e-2
    g_m = shard_g_state(jnp.asarray(g_m_raw, dtype=jnp.complex64), m14)
    f_m = sharded_solve_fields(g_m, op_m32.linear_op.solver, m14)
    jlin = jax.jit(
        lambda g, f: sharded_linear_rhs(
            g, f, op_m32.linear_op, 1e-4, m14, grids=gr_em, geo=geo_em
        )
    )
    hlo_lin = str(jlin.lower(g_m, f_m).compiler_ir(dialect="stablehlo"))
    _assert_no_6d_all_gather(hlo_lin, f"linear_rhs(1,4,M={nh_val})")
    _assert_highest_precision_no_array_f64(hlo_lin, f"linear_rhs(1,4,M={nh_val})")
    cp_lines = [ln for ln in hlo_lin.splitlines() if "collective_permute" in ln]
    assert len(cp_lines) >= min_permutes, (
        f"M={nh_val}: expected >={min_permutes} ppermutes, got {len(cp_lines)}"
    )
    for ln in cp_lines:
        pm = PERMUTE_PAIRS_RE.search(ln)
        assert pm is not None, f"Unparsed collective_permute: {ln}"
        pairs = ast.literal_eval(pm.group(1))
        diffs = {t - s for s, t in pairs}
        assert len(diffs) == 1 and next(iter(diffs)) in (-2, -1, 1, 2), (
            f"Cyclic wrap in linear_rhs(1,4,M={nh_val}) collective_permute pairs={pairs}"
        )
print("HLO AUDIT PASSED")
"""
    )


def test_single_trace_compilation_and_init_state_invariants() -> None:
    """Verify single-trace multi-step JIT compilation and init_sharded_integrator_state."""
    _run_4device_subprocess(
        """
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gx_jax import (
    benchmark_sharded_step,
    create_device_mesh,
    gather_g_state,
    init_geometry,
    init_gyrokinetic_operator,
    init_integrator_state,
    init_sharded_integrator_state,
    load_config,
    step_sharded_integrator_state,
)

base = load_config(Path("benchmarks/linear/KBM/kbm_miller.in"))
cfg = replace(
    base,
    nhermite=4, nlaguerre=2, nx=6, ny=6, nperiod=1, zp=1, ntheta_in=8, nz=8,
    nonlinear_mode=True, linear=False, closure_model="beer4+2",
    exb_shear=True, exb_shear_phase=True, g_exb=0.15,
    forcing_init=True, forcing_type="Kz", forcing_amp=0.02, forcing_index=1,
)
cfg, grids, geo = init_geometry(cfg)
gk_op = init_gyrokinetic_operator(cfg, grids, geo, dtype=jnp.float32)
mesh = create_device_mesh((2, 2), nspecies=cfg.nspecies, nhermite=cfg.nhermite)

rng = np.random.default_rng(5)
g_raw = (rng.standard_normal(grids.g_shape) + 1j * rng.standard_normal(grids.g_shape)) * 1e-2
# Leave (0, 0) nonzero to test enforce_reality_cond=True and mask_g parity!
g0 = jnp.asarray(g_raw, dtype=jnp.complex64)

# 1. Single-trace invariance across multi-step step_sharded_integrator_state
st = init_sharded_integrator_state(g0, gk_op, mesh, dt=None)
assert not isinstance(st.time.sharding, jax.sharding.SingleDeviceSharding), (
    f"init_sharded_integrator_state placed time on {st.time.sharding}"
)
assert not isinstance(st.dt.sharding, jax.sharding.SingleDeviceSharding)
assert not isinstance(st.step.sharding, jax.sharding.SingleDeviceSharding)
assert not isinstance(st.key.sharding, jax.sharding.SingleDeviceSharding)

traces = [0]

@jax.jit
def _step(s):
    traces[0] += 1
    return step_sharded_integrator_state(s, gk_op, mesh, scheme="rk3")

for _ in range(4):
    st = _step(st)
assert traces[0] == 1, f"Expected single JIT trace across steps, got trace_count={traces[0]}"

# 2. enforce_reality_cond=True & False parity vs 1-device init_integrator_state
gk_op64 = init_gyrokinetic_operator(cfg, grids, geo, dtype=jnp.float64)
g0_64 = jnp.asarray(g_raw, dtype=jnp.complex128)
for reality in (True, False):
    ref_st = init_integrator_state(g0_64, gk_op64, dt=1e-4, enforce_reality_cond=reality)
    sh_st = init_sharded_integrator_state(
        g0_64, gk_op64, mesh, dt=1e-4, enforce_reality_cond=reality
    )
    np.testing.assert_allclose(
        np.asarray(gather_g_state(sh_st.G, mesh)), np.asarray(ref_st.G), rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(sh_st.fields.phi), np.asarray(ref_st.fields.phi), rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(sh_st.fields.apar), np.asarray(ref_st.fields.apar), rtol=1e-13, atol=1e-15
    )

# 3. Smoke check benchmark_sharded_step with warmup_steps=1
bench = benchmark_sharded_step(cfg, grids, geo, mesh, warmup_steps=1, timed_steps=2, scheme="rk3")
assert bench["steps_per_sec"] > 0.0
print("SINGLE TRACE AND INIT INVARIANTS PASSED")
"""
    )


def test_corner_mesh_equivalence_kbm_closures_forcing_and_boltzmann() -> None:
    """Verify <1e-12 (c128) and <1e-6 (c64) across (1,4) M=4/8 and (2,2) EM+ExB+Forcing."""
    _run_4device_subprocess(
        """
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gx_jax import (
    SpeciesConfig,
    compute_linear_dt,
    create_device_mesh,
    gather_g_state,
    init_geometry,
    init_gyrokinetic_operator,
    init_integrator_state,
    init_sharded_integrator_state,
    linear_rhs,
    load_config,
    shard_g_state,
    sharded_linear_rhs,
    sharded_solve_fields,
    solve_fields,
    step_integrator_state,
    step_sharded_integrator_state,
)

base = load_config(Path("benchmarks/linear/KBM/kbm_miller.in"))
sp_coll = tuple(
    SpeciesConfig.from_raw(
        z=sp.z, mass=sp.mass, dens=sp.dens, temp=sp.temp,
        tprim=sp.tprim, fprim=sp.fprim,
        nu_ss=0.05 if i == 0 else 0.12,
        stype="ion" if sp.type_id == 0 else "electron",
        beta=base.beta, long_wavelength_gk=base.long_wavelength_gk,
    )
    for i, sp in enumerate(base.species)
)
cfg_em = replace(
    base, species=sp_coll, fphi=1.0, fapar=1.0, fbpar=1.0,
    nperiod=1, zp=1, ntheta_in=8, nz=8,
)

def _check_case(mshape, ov, schemes_64=("rk4", "sspx3"), check_c64=True):
    c = replace(cfg_em, **ov)
    c, gr, ge = init_geometry(c)
    dt_safe = min(compute_linear_dt(c, gr, ge), 1e-3)
    op64 = init_gyrokinetic_operator(c, gr, ge, dtype=jnp.float64)
    mesh = create_device_mesh(mshape, nspecies=c.nspecies, nhermite=c.nhermite)
    rng = np.random.default_rng(7 + c.nhermite)
    g_raw = (rng.standard_normal(gr.g_shape) + 1j * rng.standard_normal(gr.g_shape)) * 1e-3
    g_raw[:, :, :, :, 0, 0] = 0.0
    g64 = jnp.asarray(g_raw, dtype=jnp.complex128)

    for scheme in schemes_64:
        st_r = init_integrator_state(g64, op64, dt=dt_safe)
        st_s = init_sharded_integrator_state(g64, op64, mesh, dt=dt_safe)
        jr = jax.jit(lambda s: step_integrator_state(s, op64, dt_override=dt_safe, scheme=scheme))
        js = jax.jit(
            lambda s: step_sharded_integrator_state(
                s, op64, mesh, dt_override=dt_safe, scheme=scheme
            )
        )
        for _ in range(2):
            st_r = jr(st_r)
            st_s = js(st_s)
        np.testing.assert_allclose(
            np.asarray(gather_g_state(st_s.G, mesh)), np.asarray(st_r.G), rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            np.asarray(st_s.fields.phi), np.asarray(st_r.fields.phi), rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            np.asarray(st_s.fields.apar), np.asarray(st_r.fields.apar), rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            np.asarray(st_s.fields.bpar), np.asarray(st_r.fields.bpar), rtol=1e-12, atol=1e-13
        )

    if check_c64:
        op32 = init_gyrokinetic_operator(c, gr, ge, dtype=jnp.float32)
        g32 = jnp.asarray(g_raw, dtype=jnp.complex64)
        st_r32 = init_integrator_state(g32, op32, dt=dt_safe)
        st_s32 = init_sharded_integrator_state(g32, op32, mesh, dt=dt_safe)
        jr32 = jax.jit(
            lambda s: step_integrator_state(s, op32, dt_override=dt_safe, scheme="sspx2")
        )
        js32 = jax.jit(
            lambda s: step_sharded_integrator_state(
                s, op32, mesh, dt_override=dt_safe, scheme="sspx2"
            )
        )
        for _ in range(2):
            st_r32 = jr32(st_r32)
            st_s32 = js32(st_s32)
        np.testing.assert_allclose(
            np.asarray(gather_g_state(st_s32.G, mesh)),
            np.asarray(st_r32.G),
            rtol=1e-6,
            atol=1e-6,
        )

# 1. (1, 4) with nhermite=4 (nhermite_local == 1, m+-2 spans 2 shards)
_check_case(
    (1, 4),
    dict(
        nhermite=4, nlaguerre=2, nx=4, ny=4, nonlinear_mode=True, linear=False,
        closure_model="beer4+2", collisions=True, coll_conservation=True, ei_colls=True,
        hypercollisions_const=True, hypercollisions_kz=True, hyper=True,
        exb_shear=True, exb_shear_phase=True, g_exb=0.15,
        forcing_init=True, forcing_type="Kz", forcing_amp=0.02, forcing_index=1,
    ),
    schemes_64=("sspx3",),
)

# 2. (1, 4) with nhermite=8 (nhermite_local == 2) + Smith parallel closure
_check_case(
    (1, 4),
    dict(
        nhermite=8, nlaguerre=2, nx=4, ny=4, nonlinear_mode=True, linear=False,
        closure_model="smith_par", smith_par_q=3,
        collisions=True, coll_conservation=True, ei_colls=True,
        hypercollisions_const=True, hypercollisions_kz=True, hyper=True,
    ),
    schemes_64=("sspx2",),
)

# 3. (2, 2) 2-species EM + Collisions + hb_hyper + ExB shear + general Forcing across "sp"
_check_case(
    (2, 2),
    dict(
        nhermite=4, nlaguerre=2, nx=4, ny=4, nonlinear_mode=True, linear=False,
        closure_model="beer4+2", collisions=True, coll_conservation=True, ei_colls=True,
        hypercollisions_const=True, hypercollisions_kz=True, hyper=True, hb_hyper=True,
        exb_shear=True, exb_shear_phase=True, g_exb=0.15,
        forcing_init=True, forcing_type="general", stir_field="ppar",
        forcing_amp=0.02, forcing_index=1,
    ),
    schemes_64=("rk4",),
)

# 4. KzImpulse deterministic per-global-species PRNG across "sp" shards + smith_perp
cfg_imp = replace(
    cfg_em, nhermite=4, nlaguerre=4, nx=4, ny=4,
    forcing_init=True, forcing_type="KzImpulse", stir_field="qperp",
    forcing_amp=0.03, forcing_index=1, closure_model="smith_perp", smith_perp_q=3,
)
cfg_imp, gr_imp, ge_imp = init_geometry(cfg_imp)
op_imp = init_gyrokinetic_operator(cfg_imp, gr_imp, ge_imp, dtype=jnp.float64)
m22 = create_device_mesh((2, 2), nspecies=cfg_imp.nspecies, nhermite=4)
g_imp = jnp.zeros(gr_imp.g_shape, dtype=jnp.complex128)
dt_imp = 1e-4
sr = init_integrator_state(g_imp, op_imp, dt=dt_imp)
ss = init_sharded_integrator_state(g_imp, op_imp, m22, dt=dt_imp)
j_imp_r = jax.jit(lambda s: step_integrator_state(s, op_imp, dt_override=dt_imp, scheme="sspx2"))
j_imp_s = jax.jit(
    lambda s: step_sharded_integrator_state(s, op_imp, m22, dt_override=dt_imp, scheme="sspx2")
)
for _ in range(2):
    sr = j_imp_r(sr)
    ss = j_imp_s(ss)
np.testing.assert_allclose(
    np.asarray(gather_g_state(ss.G, m22)), np.asarray(sr.G), rtol=1e-12, atol=1e-13
)

# smith_perp RHS parity check
rng = np.random.default_rng(19)
gp_raw = (rng.standard_normal(gr_imp.g_shape) + 1j * rng.standard_normal(gr_imp.g_shape)) * 1e-3
gp = jnp.asarray(gp_raw, dtype=jnp.complex128)
fp_r = solve_fields(gp, op_imp.linear_op.solver)
fp_s = sharded_solve_fields(shard_g_state(gp, m22), op_imp.linear_op.solver, m22)
lin_r = linear_rhs(gp, fp_r, op_imp.linear_op, dt_imp)
lin_s = sharded_linear_rhs(
    shard_g_state(gp, m22), fp_s, op_imp.linear_op, dt_imp, m22, grids=gr_imp, geo=ge_imp
)
np.testing.assert_allclose(
    np.asarray(gather_g_state(lin_s, m22)), np.asarray(lin_r), rtol=1e-12, atol=1e-13
)

# 5. Adiabatic/Boltzmann electrons solve_fields on (1, 4)
cfg_b = replace(
    load_config(
        Path("benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in")
    ),
    nhermite=4, nlaguerre=2, nx=4, ny=4, nperiod=1, zp=1, ntheta_in=8, nz=8,
)
cfg_b, gr_b, ge_b = init_geometry(cfg_b)
op_b = init_gyrokinetic_operator(cfg_b, gr_b, ge_b, dtype=jnp.float64)
m14 = create_device_mesh((1, 4), nspecies=1, nhermite=4)
gb_raw = (rng.standard_normal(gr_b.g_shape) + 1j * rng.standard_normal(gr_b.g_shape)) * 1e-3
gb = jnp.asarray(gb_raw, dtype=jnp.complex128)
fb_r = solve_fields(gb, op_b.linear_op.solver)
fb_s = sharded_solve_fields(shard_g_state(gb, m14), op_b.linear_op.solver, m14)
np.testing.assert_allclose(np.asarray(fb_s.phi), np.asarray(fb_r.phi), rtol=1e-13, atol=1e-14)
print("CORNER MESH STRESS EQUIVALENCE PASSED")
"""
    )


def test_w7x_stellarator_linked_and_ntft_and_diagnostics_parity() -> None:
    """Verify 3D W7-X Linked & NTFT multi-step parity and sharded_compute_diagnostics."""
    _run_4device_subprocess(
        """
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gx_jax import (
    build_vmec_eik_nc,
    compute_diagnostics_step,
    create_device_mesh,
    gather_g_state,
    init_diagnostics_state,
    init_geometry,
    init_gyrokinetic_operator,
    init_integrator_state,
    init_sharded_integrator_state,
    load_config,
    load_eik_nc,
    shard_g_state,
    sharded_compute_diagnostics,
    sharded_solve_fields,
    solve_fields,
    step_integrator_state,
    step_sharded_integrator_state,
)

w7x_dir = Path("benchmarks/linear/ITG_w7x")
w7x_lin = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
w7x_nl = load_config(
    Path("benchmarks/nonlinear/w7x/w7x_adiabatic_electrons.in")
)

with tempfile.TemporaryDirectory() as tmpdir:
    eik_path = Path(tmpdir) / "w7x.eik.nc"
    build_vmec_eik_nc(
        replace(w7x_lin, ntheta_in=16, nz=16, npol=1.0),
        w7x_dir / "wout_w7x.nc",
        eik_path,
    )

    for non_twist, bnd in [(False, "linked"), (True, "nontwist")]:
        c = replace(
            w7x_nl,
            nhermite=4, nlaguerre=2, nx=4, ny=4, ntheta_in=16, nz=16, nperiod=1,
            non_twist=non_twist, boundary=bnd, geo_option="nc", geofilename=str(eik_path),
            nonlinear_mode=True, linear=False,
        )
        c, gr, ge = load_eik_nc(eik_path, c)
        dt_w = 5e-4
        op64 = init_gyrokinetic_operator(c, gr, ge, dtype=jnp.float64)
        mesh = create_device_mesh((1, 4), nspecies=c.nspecies, nhermite=4)
        rng = np.random.default_rng(101 + int(non_twist))
        g_raw = (rng.standard_normal(gr.g_shape) + 1j * rng.standard_normal(gr.g_shape)) * 1e-3
        g_raw[:, :, :, :, 0, 0] = 0.0
        g64 = jnp.asarray(g_raw, dtype=jnp.complex128)
        jdiag_r = jax.jit(
            lambda g, f, ds: compute_diagnostics_step(
                g, f, ds, gr, ge, op64.linear_op.solver,
                time=dt_w, dt=dt_w, beta=float(c.beta),
            )
        )
        jdiag_s = jax.jit(
            lambda g, f, ds: sharded_compute_diagnostics(
                g, f, ds, gr, ge, op64.linear_op.solver, mesh,
                time=dt_w, dt=dt_w, beta=float(c.beta),
            )
        )

        for scheme in ("rk4", "sspx3"):
            sr = init_integrator_state(g64, op64, dt=dt_w)
            ss = init_sharded_integrator_state(g64, op64, mesh, dt=dt_w)
            jr = jax.jit(lambda s: step_integrator_state(s, op64, dt_override=dt_w, scheme=scheme))
            js = jax.jit(
                lambda s: step_sharded_integrator_state(
                    s, op64, mesh, dt_override=dt_w, scheme=scheme
                )
            )
            ds_r = init_diagnostics_state(sr.G, sr.fields, time=0.0, has_old=False)
            ds_s = init_diagnostics_state(ss.G, ss.fields, time=0.0, has_old=False)
            for step_i in range(2):
                sr = jr(sr)
                ss = js(ss)
                if step_i == 0:
                    ds_r = init_diagnostics_state(sr.G, sr.fields, time=sr.time, has_old=True)
                    ds_s = init_diagnostics_state(ss.G, ss.fields, time=ss.time, has_old=True)
                else:
                    snap_r, ds_r = jdiag_r(sr.G, sr.fields, ds_r)
                    snap_s, ds_s = jdiag_s(ss.G, ss.fields, ds_s)
                    leaves_s = jax.tree_util.tree_leaves(snap_s)
                    leaves_r = jax.tree_util.tree_leaves(snap_r)
                    for ls, lr in zip(leaves_s, leaves_r):
                        np.testing.assert_allclose(
                            np.asarray(ls), np.asarray(lr), rtol=1e-12, atol=1e-13
                        )
            np.testing.assert_allclose(
                np.asarray(gather_g_state(ss.G, mesh)), np.asarray(sr.G), rtol=1e-12, atol=1e-13
            )
            np.testing.assert_allclose(
                np.asarray(ss.fields.phi), np.asarray(sr.fields.phi), rtol=1e-12, atol=1e-13
            )

        if non_twist:
            op32 = init_gyrokinetic_operator(c, gr, ge, dtype=jnp.float32)
            g32 = jnp.asarray(g_raw, dtype=jnp.complex64)
            sr32 = init_integrator_state(g32, op32, dt=dt_w)
            ss32 = init_sharded_integrator_state(g32, op32, mesh, dt=dt_w)
            jr32 = jax.jit(
                lambda s: step_integrator_state(s, op32, dt_override=dt_w, scheme="rk4")
            )
            js32 = jax.jit(
                lambda s: step_sharded_integrator_state(
                    s, op32, mesh, dt_override=dt_w, scheme="rk4"
                )
            )
            for _ in range(2):
                sr32 = jr32(sr32)
                ss32 = js32(ss32)
            np.testing.assert_allclose(
                np.asarray(gather_g_state(ss32.G, mesh)),
                np.asarray(sr32.G),
                rtol=1e-6,
                atol=1e-6,
            )

# 2-species EM KBM (2, 2) full DiagnosticsSnapshot PyTree parity
cfg_kbm = replace(
    load_config(Path("benchmarks/linear/KBM/kbm_miller.in")),
    nhermite=4, nlaguerre=2, nx=4, ny=4, nperiod=1, zp=1, ntheta_in=8, nz=8,
)
cfg_kbm, gr_k, ge_k = init_geometry(cfg_kbm)
op_k = init_gyrokinetic_operator(cfg_kbm, gr_k, ge_k, dtype=jnp.float64)
m22 = create_device_mesh((2, 2), nspecies=cfg_kbm.nspecies, nhermite=4)
rng = np.random.default_rng(77)
gk_raw = (rng.standard_normal(gr_k.g_shape) + 1j * rng.standard_normal(gr_k.g_shape)) * 1e-2
gk0 = jnp.asarray(gk_raw, dtype=jnp.complex128).at[:, :, :, :, 0, 0].set(0.0)
gk1 = gk0 * 1.01 + 1e-4j * jnp.conj(gk0)
fk0_r = solve_fields(gk0, op_k.linear_op.solver)
fk1_r = solve_fields(gk1, op_k.linear_op.solver)
gk0_s = shard_g_state(gk0, m22)
gk1_s = shard_g_state(gk1, m22)
fk0_s = sharded_solve_fields(gk0_s, op_k.linear_op.solver, m22)
fk1_s = sharded_solve_fields(gk1_s, op_k.linear_op.solver, m22)
ds_kr = init_diagnostics_state(gk0, fk0_r, time=0.0, has_old=True)
ds_ks = init_diagnostics_state(gk0_s, fk0_s, time=0.0, has_old=True)
jk_r = jax.jit(
    lambda g, f, ds: compute_diagnostics_step(
        g, f, ds, gr_k, ge_k, op_k.linear_op.solver,
        time=1e-3, dt=1e-3, beta=float(cfg_kbm.beta),
    )
)
jk_s = jax.jit(
    lambda g, f, ds: sharded_compute_diagnostics(
        g, f, ds, gr_k, ge_k, op_k.linear_op.solver, m22,
        time=1e-3, dt=1e-3, beta=float(cfg_kbm.beta),
    )
)
sk_r, _ = jk_r(gk1, fk1_r, ds_kr)
sk_s, _ = jk_s(gk1_s, fk1_s, ds_ks)
for ls, lr in zip(jax.tree_util.tree_leaves(sk_s), jax.tree_util.tree_leaves(sk_r)):
    np.testing.assert_allclose(np.asarray(ls), np.asarray(lr), rtol=1e-12, atol=1e-13)
print("W7-X AND DIAGNOSTICS PARITY PASSED")
"""
    )


def test_sharded_cli_end_to_end_fp32_and_benchmark() -> None:
    """Verify cli_main --mesh 1x4 --precision fp32 executes cleanly."""
    _run_4device_subprocess(
        """
import tempfile
from pathlib import Path

from gx_jax import cli_main

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    in_path = tmp / "mini_itg.in"
    in_path.write_text(
        '''[Dimensions]
ntheta = 8
nperiod = 1
ny = 4
nx = 4
nhermite = 4
nlaguerre = 2
nspecies = 1

[Domain]
y0 = 10.0
boundary = "linked"

[Physics]
nonlinear_mode = false

[Boltzmann]
add_boltzmann_species = true
boltzmann_type = "electrons"
tau_fac = 1.0

[Time]
dt = 0.001
nstep = 1
scheme = "rk3"

[Initialization]
init_field = "density"
init_amp = 1.0e-5
ikpar_init = 0

[Geometry]
geo_option = "s-alpha"
eps = 0.167
qsf = 1.4
shat = 0.8

[Diagnostics]
nwrite = 2
write_free_energy = true

[species]
z = [1.0]
mass = [1.0]
dens = [1.0]
temp = [1.0]
tprim = [3.0]
fprim = [1.0]
vnewk = [0.0]
type = ["ion"]
''',
        encoding="utf-8",
    )
    out_nc = tmp / "mini_itg.out.nc"
    rc = cli_main([
        str(in_path),
        "-o", str(out_nc),
        "--mesh", "1x4",
        "--precision", "fp32",
        "--nstep", "1",
        "--nwrite", "2",
    ])
    assert rc == 0
    assert out_nc.exists()
print("SHARDED CLI PASSED")
"""
    )
