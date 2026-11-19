"""Build-time StableHLO verification of zero 64-bit device tensors in fp32 mode."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from gx_jax.config import GXConfig, SpeciesConfig, load_config
from gx_jax.diagnostics_nc import init_diagnostics_state
from gx_jax.ensemble import (
    batched_compute_diagnostics_step,
    create_ensemble_mesh,
    init_batched_ensemble,
    init_batched_integrator_state,
    make_ensemble_chunk_stepper,
    shard_ensemble_tree,
)
from gx_jax.fields import init_moments
from gx_jax.geometry_nc import init_geometry
from gx_jax.integrator import init_gyrokinetic_operator
from gx_jax.shard_tpu import (
    _place_diag_state_on_mesh,
    advance_sharded_integrator_chunk,
    create_device_mesh,
    host_precompute_context,
    init_sharded_integrator_state,
    make_sharded_chunk_stepper,
    move_tree_to_default_device,
    sharded_compute_diagnostics,
)


def main() -> int:
    sp_i = SpeciesConfig.from_raw(
        z=1.0,
        mass=1.0,
        dens=1.0,
        temp=1.0,
        tprim=3.0,
        fprim=1.0,
        nu_ss=0.01,
        stype="ion",
        beta=1.0e-3,
    )
    sp_e = SpeciesConfig.from_raw(
        z=-1.0,
        mass=2.7e-4,
        dens=1.0,
        temp=1.0,
        tprim=3.0,
        fprim=1.0,
        nu_ss=0.01,
        stype="electron",
        beta=1.0e-3,
    )

    for nt in (False, True):
        cfg = GXConfig(
            nx=16,
            ny=16,
            nz=16,
            ntheta_in=16,
            nhermite=4,
            nlaguerre=2,
            nspecies=2,
            nonlinear_mode=True,
            linear=False,
            non_twist=nt,
            beta=1.0e-3,
            fapar=1.0,
            fbpar=1.0,
            all_kinetic=True,
            collisions=True,
            hyper=True,
            hypercollisions_kz=True,
            write_omega=True,
            write_fluxes=True,
            write_free_energy=True,
            species=(sp_i, sp_e),
        )
        mesh = create_device_mesh((2, 2), enable_x64=True)
        with host_precompute_context():
            cfg, grids_h, geo_h = init_geometry(cfg, dtype=jnp.float32)
            g0_h = init_moments(cfg, grids_h, geo_h, dtype=jnp.complex64)
            gk_h = init_gyrokinetic_operator(cfg, grids_h, geo_h, dtype=jnp.float32)
        grids = move_tree_to_default_device(grids_h.astype(jnp.float32))
        geo = move_tree_to_default_device(geo_h)
        g0 = move_tree_to_default_device(g0_h)
        gk = move_tree_to_default_device(gk_h)

        st = init_sharded_integrator_state(g0, gk, mesh, time=0.0)
        dg = _place_diag_state_on_mesh(
            init_diagnostics_state(st.G, st.fields, time=st.time, has_old=False),
            mesh,
        )
        chunk_fn = jax.jit(
            lambda s, d, n, _gk=gk, _mesh=mesh, _cfg=cfg: advance_sharded_integrator_chunk(
                s,
                _gk,
                _mesh,
                n,
                scheme=_cfg.scheme,
                t_max=float(_cfg.t_max),
                diag_state=d,
                diag_nwrite=int(_cfg.nwrite),
            )
        )
        hlo1 = chunk_fn.lower(st, dg, jnp.int32(2)).as_text()
        stepper = make_sharded_chunk_stepper(
            gk,
            mesh,
            scheme=cfg.scheme,
            t_max=float(cfg.t_max),
            diag_nwrite=int(cfg.nwrite),
        )
        _ = stepper(st, 2, diag_state=dg)

        diag_fn = jax.jit(
            lambda g, f, d, t, dt, _grids=grids, _geo=geo, _gk=gk, _mesh=mesh, _cfg=cfg: (
                sharded_compute_diagnostics(
                    g,
                    f,
                    d,
                    _grids,
                    _geo,
                    _gk.linear_op.solver,
                    _mesh,
                    time=t,
                    dt=dt,
                    beta=float(_cfg.beta),
                )
            )
        )
        hlo2 = diag_fn.lower(st.G, st.fields, dg, st.time, st.dt).as_text()
        bad1 = re.findall(r"tensor<[^>]*?(?:f64|c128)>", hlo1)
        bad2 = re.findall(r"tensor<[^>]*?(?:f64|c128)>", hlo2)
        if bad1 or bad2:
            hlo2_lines = [ln.strip() for ln in hlo2.splitlines() if "f64" in ln or "c128" in ln]
            raise AssertionError(
                f"Found 64-bit tensors (nt={nt}): step={bad1[:5]}, diag={bad2[:5]}\n"
                + "\n".join(hlo2_lines[:20])
            )

    repo_root = Path(__file__).resolve().parent.parent
    decks = [str(repo_root / f"benchmarks/linear/ITG_w7x/w7x_opt_{i}.in") for i in range(4)]
    specs, b_ops = init_batched_ensemble(decks, dtype=jnp.float32, nstep=10, nwrite=5)
    emesh = create_ensemble_mesh(4, num_devices=4)
    with host_precompute_context():
        gs = jnp.stack(
            [init_moments(s.cfg, s.grids, s.geo, dtype=jnp.complex64) for s in specs],
            axis=0,
        )
    gs = move_tree_to_default_device(gs)
    ops_sh = shard_ensemble_tree(b_ops, emesh, batch_size=4)
    g_sh_b = shard_ensemble_tree(gs, emesh, batch_size=4)
    est = init_batched_integrator_state(g_sh_b, ops_sh, time=[0.0] * 4, mesh=emesh)
    edg = jax.vmap(lambda gm, fm, tm: init_diagnostics_state(gm, fm, time=tm, has_old=False))(
        est.G, est.fields, est.time
    )
    estepper = make_ensemble_chunk_stepper(ops_sh, mesh=emesh)
    hlo3 = estepper.lower(est, edg, jnp.int32(2)).as_text()
    ediag = jax.jit(
        lambda g_b, f_b, d_b, t_b, dt_b: batched_compute_diagnostics_step(
            g_b, f_b, d_b, ops_sh, time=t_b, dt=dt_b
        )
    )
    hlo4 = ediag.lower(est.G, est.fields, edg, est.time, est.dt).as_text()
    bad3 = re.findall(r"tensor<[^>]*?(?:f64|c128)>", hlo3)
    bad4 = re.findall(r"tensor<[^>]*?(?:f64|c128)>", hlo4)
    if bad3 or bad4:
        ens_lines = [
            ln.strip() for ln in (hlo3 + "\n" + hlo4).splitlines() if "f64" in ln or "c128" in ln
        ]
        raise AssertionError(
            f"Found 64-bit tensors in ensemble: step={bad3[:5]}, diag={bad4[:5]}\n"
            + "\n".join(ens_lines[:20])
        )

    med_deck = repo_root / "benchmarks/timing/cyclone_adiab_med_6m.in"
    for fft_backend in ("fft", "mxu"):
        cfg_m = replace(load_config(med_deck), fft_backend=fft_backend)
        mesh_m = create_device_mesh((1, 4), enable_x64=True)
        with host_precompute_context():
            cfg_m, grids_m_h, geo_m_h = init_geometry(cfg_m, dtype=jnp.float32)
            g0_m_h = init_moments(cfg_m, grids_m_h, geo_m_h, dtype=jnp.complex64)
            gk_m_h = init_gyrokinetic_operator(cfg_m, grids_m_h, geo_m_h, dtype=jnp.float32)
        grids_m = move_tree_to_default_device(grids_m_h.astype(jnp.float32))
        geo_m = move_tree_to_default_device(geo_m_h)
        g0_m = move_tree_to_default_device(g0_m_h)
        gk_m = move_tree_to_default_device(gk_m_h)
        st_m = init_sharded_integrator_state(g0_m, gk_m, mesh_m, time=0.0)
        dg_m = _place_diag_state_on_mesh(
            init_diagnostics_state(st_m.G, st_m.fields, time=st_m.time, has_old=False),
            mesh_m,
        )
        chunk_m_fn = jax.jit(
            lambda s, d, n, _gk=gk_m, _mesh=mesh_m, _cfg=cfg_m: advance_sharded_integrator_chunk(
                s,
                _gk,
                _mesh,
                n,
                scheme=_cfg.scheme,
                t_max=float(_cfg.t_max),
                diag_state=d,
                diag_nwrite=int(_cfg.nwrite),
            )
        )
        hlo_m1 = chunk_m_fn.lower(st_m, dg_m, jnp.int32(2)).as_text()
        stepper_m = make_sharded_chunk_stepper(
            gk_m,
            mesh_m,
            scheme=cfg_m.scheme,
            t_max=float(cfg_m.t_max),
            diag_nwrite=int(cfg_m.nwrite),
        )
        _ = stepper_m(st_m, 2, diag_state=dg_m)
        diag_m = jax.jit(
            lambda g, f, d, t, dt, _grids=grids_m, _geo=geo_m, _gk=gk_m, _mesh=mesh_m, _cfg=cfg_m: (
                sharded_compute_diagnostics(
                    g,
                    f,
                    d,
                    _grids,
                    _geo,
                    _gk.linear_op.solver,
                    _mesh,
                    time=t,
                    dt=dt,
                    beta=float(_cfg.beta),
                )
            )
        )
        hlo_m2 = diag_m.lower(st_m.G, st_m.fields, dg_m, st_m.time, st_m.dt).as_text()
        bad_m1 = re.findall(r"tensor<[^>]*?(?:f64|c128)>", hlo_m1)
        bad_m2 = re.findall(r"tensor<[^>]*?(?:f64|c128)>", hlo_m2)
        if bad_m1 or bad_m2:
            raise AssertionError(
                f"Found 64-bit tensors in cyclone_adiab_med_6m ({fft_backend}): "
                f"step={bad_m1[:5]}, diag={bad_m2[:5]}"
            )

    print(
        "StableHLO zero-f64/c128 check PASSED "
        "(sharded Linked, sharded NTFT, cyclone fft/mxu, and 4-chip W7-X VMEC ensemble)!"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
