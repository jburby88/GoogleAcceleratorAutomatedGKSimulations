"""TOML input parser and physics/domain configuration for GX-JAX.

Ported from ``gx/include/{parameters.h,species.h}`` and
``gx/src/parameters.cu``.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any


class BoltzmannOption(IntEnum):
    """Adiabatic species response mode (``parameters.h:L46-47``)."""

    NONE = 0
    BOLTZMANN_IONS = 1
    BOLTZMANN_ELECTRONS = 2


def c_round(x: float) -> int:
    """Round half away from zero, matching C/C++ ``std::round``."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


@dataclass(frozen=True)
class SpeciesConfig:
    """Normalized kinetic species parameters (``species.h``, ``parameters.cu:L1055-1097``)."""

    z: float
    mass: float
    dens: float
    temp: float
    tprim: float
    fprim: float
    nu_ss: float
    rho2: float
    rho2_long_wavelength_GK: float
    tz: float
    zt: float
    vt: float
    nt: float
    nz: float
    jparfac: float
    jperpfac: float
    type_id: int  # 0 = ion, 1 = electron

    @classmethod
    def from_raw(
        cls,
        *,
        z: float,
        mass: float,
        dens: float,
        temp: float = 1.0,
        tprim: float = 0.0,
        fprim: float = 0.0,
        nu_ss: float = 0.0,
        stype: str = "ion",
        beta: float = 0.0,
        long_wavelength_gk: bool = False,
    ) -> SpeciesConfig:
        vt = math.sqrt(temp / mass)
        tz = temp / z
        zt = z / temp
        rho2_raw = temp * mass / (z * z)
        if long_wavelength_gk:
            rho2 = 0.0
            rho2_lw = rho2_raw
        else:
            rho2 = rho2_raw
            rho2_lw = 0.0
        nt = dens * temp
        nz = dens * z
        jparfac = nz * vt * beta / 2.0
        jperpfac = -dens * temp * beta / 2.0
        type_id = 0 if stype.lower() == "ion" else 1
        return cls(
            z=float(z),
            mass=float(mass),
            dens=float(dens),
            temp=float(temp),
            tprim=float(tprim),
            fprim=float(fprim),
            nu_ss=float(nu_ss),
            rho2=float(rho2),
            rho2_long_wavelength_GK=float(rho2_lw),
            tz=float(tz),
            zt=float(zt),
            vt=float(vt),
            nt=float(nt),
            nz=float(nz),
            jparfac=float(jparfac),
            jperpfac=float(jperpfac),
            type_id=type_id,
        )


@dataclass(frozen=True)
class GXConfig:
    """Complete simulation configuration parsed from a GX ``.in`` TOML file."""

    # Identification
    run_name: str = "gx"
    input_path: Path | None = None

    # [Dimensions]
    nx: int = 4
    ny: int = 4
    nz: int = 32
    ntheta_in: int = 32
    nhermite: int = 4
    nlaguerre: int = 2
    nspecies: int = 1
    nperiod: int = 1

    # [Domain]
    x0: float = -1.0
    y0: float = 10.0
    z0: float = 1.0
    jtwist: int = -1000
    zp: int = 1
    boundary: str = "linked"
    boundary_option_periodic: bool = False
    non_twist: bool = False
    long_wavelength_gk: bool = False
    zero_shat_threshold: float = 1.0e-5

    # [Physics]
    beta: float = 0.0
    nonlinear_mode: bool = False
    linear: bool = True
    g_exb: float = 0.0
    exb_shear: bool = False
    exb_shear_phase: bool = False
    fphi: float = 1.0
    fapar: float = 0.0
    fbpar: float = 0.0
    ei_colls: bool = True
    coll_conservation: bool = True
    collisions: bool = False

    # [Boltzmann]
    all_kinetic: bool = True
    add_boltzmann_species: bool = False
    boltzmann_opt: BoltzmannOption = BoltzmannOption.BOLTZMANN_ELECTRONS
    boltzmann_type: str = "electrons"
    tau_fac: float = 1.0

    # [Geometry]
    geo_option: str = "s-alpha"
    geofilename: str = "eik.out"
    vmec_file: str = ""
    igeo: int = -1
    slab: bool = False
    const_curv: bool = False
    drhodpsi: float = 1.0
    kxfac: float = 1.0
    rmaj: float = 1.0
    r_geo: float = 1.0
    shift: float = 0.0
    eps: float = 0.167
    qsf: float = 1.4
    shat: float = 0.8
    akappa: float = 1.0
    akappri: float = 0.0
    tri: float = 1.0
    tripri: float = 0.0
    betaprim: float = -1.0
    rhoc: float = 0.5
    torflux: float = 0.64
    npol: float = 1.0
    npol_min: float | None = None
    alpha: float = 0.0
    zero_shat: bool = False
    rbzeta_override: float = 0.0
    b_ref: float = 1.0
    a_ref: float = 1.0
    grhoavg: float = 1.0
    surfarea: float = 1.0

    # [Time]
    dt: float = 0.05
    dt_max: float = 0.05
    dt_min: float = 1.0e-7
    fixed_dt: bool = False
    nstep: int = 2_000_000_000
    scheme: str = "rk3"
    cfl: float = 0.9
    stages: int = 10
    t_max: float = 1.0e20

    # [Initialization]
    init_field: str = "density"
    init_amp: float = 1.0e-5
    kpar_init: float = 0.0
    ikpar_init: int = 0
    random_init: bool = False
    gaussian_init: bool = False
    gauss_env_const_coeff: float = 1.0
    gauss_env_sin_coeff: float = 0.0
    gaussian_width: float = 0.5
    init_electrons_only: bool = False
    random_seed: int = 22

    # [Dissipation] & [Expert]
    closure_model: str = "none"
    smith_par_q: int = 3
    smith_perp_q: int = 3
    hb_hyper: bool = False
    d_hb: float = 1.0
    p_hb: int = 2
    w_osc: float = 0.0
    dealias_kz: bool = False
    local_limit: bool = False
    init_single: bool = False
    ikx_single: int = 0
    iky_single: int = 1
    damp_ends_widthfrac: float = 0.125
    damp_ends_amp: float = 0.1
    hyper: bool = False
    d_hyper: float = 0.1
    p_hyper: int = 2
    hyperz: bool = False
    nu_hyper_z: float = 0.1
    p_hyper_z: int = 6
    hypercollisions_const: bool = False
    hypercollisions_kz: bool = False
    nu_hyper_l: float = 0.0
    nu_hyper_m: float = 1.0
    nu_hyper_lm: float = 0.0
    p_hyper_l: int = 6
    p_hyper_m: int = 2
    p_hyper_lm: int = 6

    # [Diagnostics]
    nwrite: int = 1000
    navg: int = 10
    nwrite_big: int = 100_000
    fixed_amplitude: bool = False
    write_omega: bool = False
    write_free_energy: bool = True
    write_fluxes: bool = False
    write_moms: bool = False
    write_fields: bool = False

    # [Restart] (parameters.cu:L155-170)
    restart: bool = False
    restart_if_exists: bool = False
    save_for_restart: bool = True
    restart_to_file: str = ""
    restart_from_file: str = ""
    restart_with_perturb: bool = False
    append_on_restart: bool = True
    restart_scale: float = 1.0
    nsave: int = 10_000

    # [Forcing] (parameters.cu:L346-360)
    forcing_init: bool = False
    forcing_type: str = "Kz"
    stir_field: str = "density"
    forcing_amp: float = 1.0
    pos_forcing_amp: float = 1.0
    neg_forcing_amp: float = 1.0
    forcing_index: int = 1
    no_fields: bool = False
    forcing_kz: int = 0
    forcing_k2min: int = 0
    forcing_k2max: int = 0

    # Kinetic species tuple
    species: tuple[SpeciesConfig, ...] = ()
    vtmax: float = 1.0
    vtmin: float = 1.0
    tzmax: float = 1.0
    etamax: float = -1.0
    ne: float = 0.0
    te: float = 0.0

    # Debug flag & TPU MXU / FFT spectral backend
    debug: bool = False
    fft_backend: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.fft_backend, str):
            raise ValueError(
                f"Unsupported fft_backend={self.fft_backend!r}. "
                "Expected one of ('auto', 'fft', 'mxu')."
            )
        fb_clean = self.fft_backend.strip().lower()
        if fb_clean not in ("auto", "fft", "mxu"):
            raise ValueError(
                f"Unsupported fft_backend={self.fft_backend!r}. "
                "Expected one of ('auto', 'fft', 'mxu')."
            )
        if fb_clean != self.fft_backend:
            object.__setattr__(self, "fft_backend", fb_clean)


def _parse_fft_backend_nml(nml: dict[str, Any]) -> str:
    """Extract and validate ``fft_backend`` from ``[Expert]``, ``[Dimensions]``, or top-level."""
    for sec_name in ("Expert", "Dimensions", "Controls"):
        sec_dict = nml.get(sec_name)
        if isinstance(sec_dict, dict):
            for key, val in sec_dict.items():
                if key.lower() == "fft_backend":
                    if not isinstance(val, str):
                        raise ValueError(
                            f"Unsupported fft_backend={val!r}. "
                            "Expected one of ('auto', 'fft', 'mxu')."
                        )
                    fb = val.strip().lower()
                    if fb not in ("auto", "fft", "mxu"):
                        raise ValueError(
                            f"Unsupported fft_backend={val!r}. "
                            "Expected one of ('auto', 'fft', 'mxu')."
                        )
                    return fb
    for key, val in nml.items():
        if not isinstance(val, dict) and key.lower() == "fft_backend":
            if not isinstance(val, str):
                raise ValueError(
                    f"Unsupported fft_backend={val!r}. Expected one of ('auto', 'fft', 'mxu')."
                )
            fb = val.strip().lower()
            if fb not in ("auto", "fft", "mxu"):
                raise ValueError(
                    f"Unsupported fft_backend={val!r}. Expected one of ('auto', 'fft', 'mxu')."
                )
            return fb
    return "auto"


def _sec(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return section dictionary if present, else fall back to Controls + top-level."""
    val = raw.get(name)
    if isinstance(val, dict):
        return val
    ctrl = raw.get("Controls")
    if isinstance(ctrl, dict):
        return {**ctrl, **raw}
    return raw


def load_config(path: str | Path) -> GXConfig:
    """Parse a GX ``.in`` TOML file strictly following ``Parameters::get_nml_vars``."""
    in_path = Path(path)
    with in_path.open("rb") as fh:
        nml = tomllib.load(fh)

    run_name = in_path.stem
    debug = bool(nml.get("debug", False))
    fft_backend = _parse_fft_backend_nml(nml)

    # [Dimensions] (parameters.cu:L57-76)
    dims = _sec(nml, "Dimensions")
    ntheta_in = int(dims.get("ntheta", 32))
    ny_in = int(dims.get("ny", 0))
    nx_in = int(dims.get("nx", 0))
    nky_in = int(dims.get("nky", 0))
    nkx_in = int(dims.get("nkx", 0))
    nm_in = int(dims.get("nhermite", 4))
    nl_in = int(dims.get("nlaguerre", 2))
    nspec_in = int(dims.get("nspecies", 1))
    nperiod = int(dims.get("nperiod", 1))

    if ny_in <= 0 and nky_in <= 0:
        raise ValueError("Must set ny or nky in [Dimensions].")
    if nky_in > 0 and ny_in > 0:
        if nky_in != 1 + (ny_in - 1) // 3:
            raise ValueError(f"Inconsistent nky={nky_in} and ny={ny_in}.")
    elif nky_in > 0:
        ny_in = 3 * (nky_in - 1) + 1

    if nx_in <= 0 and nkx_in <= 0:
        raise ValueError("Must set nx or nkx in [Dimensions].")
    if nkx_in > 0 and nx_in > 0:
        if nkx_in != 1 + 2 * ((nx_in - 1) // 3):
            raise ValueError(f"Inconsistent nkx={nkx_in} and nx={nx_in}.")
    elif nkx_in > 0:
        nx_in = ((nkx_in - 1) // 2) * 3 + 1

    # [Domain] (parameters.cu:L82-98)
    dom = _sec(nml, "Domain")
    y0 = float(dom.get("y0", dims.get("y0", 10.0)))
    x0 = float(dom.get("x0", dims.get("x0", -1.0)))
    z0 = float(dom.get("z0", 1.0))
    jtwist = int(dom.get("jtwist", -1000))
    zp = int(dom.get("zp", 2 * nperiod - 1))
    boundary = str(dom.get("boundary", "linked"))
    non_twist = bool(dom.get("nonTwist", False))
    long_wavelength_gk = bool(dom.get("long_wavelength_GK", False))
    zero_shat_threshold = float(dom.get("zero_shat_threshold", 1.0e-5))

    # [Time] & legacy [Controls] fallback (parameters.cu:L100-115, L311-330)
    ctrl_sec = _sec(nml, "Controls")
    tsec = _sec(nml, "Time")
    dt = float(tsec.get("dt", 0.05))
    dt_max = float(tsec.get("dt_max", dt))
    dt_min = float(tsec.get("dt_min", 1.0e-7))
    fixed_dt = bool(tsec.get("fixed_dt", False))
    nstep = int(tsec.get("nstep", 2_000_000_000))
    scheme = str(tsec.get("scheme", ctrl_sec.get("scheme", "rk3")))
    cfl = float(tsec.get("cfl", ctrl_sec.get("cfl", 0.9)))
    stages = int(tsec.get("stages", ctrl_sec.get("stages", 10)))
    t_max = float(tsec.get("t_max", 1.0e20))
    nwrite_time = int(tsec.get("nwrite", 1000))
    navg_time = int(tsec.get("navg", 10))

    # [Initialization] (parameters.cu:L116-154, L321-324)
    init_sec = _sec(nml, "Initialization")
    init_field = str(init_sec.get("init_field", ctrl_sec.get("init_field", "density")))
    init_amp = float(init_sec.get("init_amp", ctrl_sec.get("init_amp", 1.0e-5)))
    kpar_init = float(init_sec.get("kpar_init", ctrl_sec.get("kpar_init", 0.0)))
    ikpar_init = int(init_sec.get("ikpar_init", ctrl_sec.get("ikpar_init", int(kpar_init))))
    random_init = bool(init_sec.get("random_init", False))
    if random_init:
        ikpar_init = 0
    gaussian_init = bool(init_sec.get("gaussian_init", False))
    gauss_env_const_coeff = float(init_sec.get("gaussian_envelope_constant_coefficient", 1.0))
    gauss_env_sin_coeff = float(init_sec.get("gaussian_envelope_sine_coefficient", 0.0))
    gaussian_width = float(init_sec.get("gaussian_width", 0.5))
    init_electrons_only = bool(init_sec.get("init_electrons_only", False))
    random_seed = int(init_sec.get("random_seed", 22))

    # [Dissipation] (parameters.cu:L171-194, L314-316, L325-330)
    diss = _sec(nml, "Dissipation")
    closure_model = str(diss.get("closure_model", ctrl_sec.get("closure_model", "none")))
    smith_par_q = int(diss.get("smith_par_q", ctrl_sec.get("smith_par_q", 3)))
    smith_perp_q = int(diss.get("smith_perp_q", ctrl_sec.get("smith_perp_q", 3)))
    d_hb = float(diss.get("D_HB", ctrl_sec.get("D_HB", 1.0)))
    w_osc = float(diss.get("w_osc", ctrl_sec.get("w_osc", 0.0)))
    p_hb = int(diss.get("p_HB", 2))
    hb_hyper = bool(diss.get("HB_hyper", False))
    d_hyper = float(diss.get("D_hyper", ctrl_sec.get("D_hyper", 0.1)))
    nu_hyper_z = float(diss.get("nu_hyper_z", 0.1))
    nu_hyper_l = float(diss.get("nu_hyper_l", ctrl_sec.get("nu_hyper_l", 0.0)))
    nu_hyper_m = float(diss.get("nu_hyper_m", ctrl_sec.get("nu_hyper_m", 1.0)))
    nu_hyper_lm = float(diss.get("nu_hyper_lm", 0.0))
    p_hyper = int(diss.get("p_hyper", 2))
    p_hyper_z = int(diss.get("p_hyper_z", 6))
    p_hyper_l = int(diss.get("p_hyper_l", 6))
    p_hyper_m = int(diss.get("p_hyper_m", min(20, nm_in // 2)))
    p_hyper_lm = int(diss.get("p_hyper_lm", 6))
    hyper = bool(diss.get("hyper", False))
    hypercollisions_const = bool(diss.get("hypercollisions_const", False))
    hypercollisions_kz = bool(diss.get("hypercollisions_kz", False))
    hypercollisions_kz = bool(diss.get("hypercollisions", hypercollisions_kz))
    hyperz = bool(diss.get("hyperz", False))

    # [Expert] (parameters.cu:L218-252, L312)
    exp_sec = _sec(nml, "Expert")
    dealias_kz = bool(exp_sec.get("dealias_kz", ctrl_sec.get("dealias_kz", False)))
    local_limit = bool(exp_sec.get("local_limit", False))
    init_single = bool(exp_sec.get("init_single", False))
    ikx_single = int(exp_sec.get("ikx_single", 0))
    iky_single = int(exp_sec.get("iky_single", 1))
    hegna = bool(exp_sec.get("hegna", False))
    damp_ends_widthfrac = float(exp_sec.get("damp_ends_widthfrac", 0.125))
    damp_ends_amp = float(exp_sec.get("damp_ends_amp", 0.1))
    if hegna:
        nm_in = 4
        nl_in = 2

    # [Diagnostics] (parameters.cu:L255-266)
    diag = _sec(nml, "Diagnostics")
    nwrite = int(diag.get("nwrite", nwrite_time))
    navg = int(diag.get("navg", navg_time))
    nwrite_big = int(diag.get("nwrite_big", nwrite * 100))
    fixed_amplitude = bool(diag.get("fixed_amplitude", False))
    write_omega = bool(diag.get("omega", False))
    write_free_energy = bool(diag.get("free_energy", True))
    write_fluxes = bool(diag.get("fluxes", False))
    write_moms = bool(diag.get("moments", False))
    write_fields = bool(diag.get("fields", False))

    # [Restart] (parameters.cu:L155-170)
    rest_sec = _sec(nml, "Restart")
    default_restart_file = f"{run_name}.restart.nc"
    restart = bool(rest_sec.get("restart", False))
    restart_if_exists = bool(rest_sec.get("restart_if_exists", False))
    save_for_restart = bool(rest_sec.get("save_for_restart", True))
    restart_to_file = str(rest_sec.get("restart_to_file", default_restart_file))
    restart_from_file = str(rest_sec.get("restart_from_file", default_restart_file))
    restart_with_perturb = bool(rest_sec.get("restart_with_perturb", False))
    append_on_restart = bool(rest_sec.get("append_on_restart", True))
    restart_scale = float(rest_sec.get("scale", 1.0))
    nsave = max(1, int(rest_sec.get("nsave", tsec.get("nsave", 10_000))))

    # [Forcing] (parameters.cu:L346-360)
    forc_sec = _sec(nml, "Forcing")
    forcing_type = str(forc_sec.get("forcing_type", "Kz"))
    stir_field = str(forc_sec.get("stir_field", "density"))
    forcing_amp = float(forc_sec.get("forcing_amp", 1.0))
    pos_forcing_amp = float(forc_sec.get("pos_forcing_amp", 1.0))
    neg_forcing_amp = float(forc_sec.get("neg_forcing_amp", 1.0))
    forcing_index = int(forc_sec.get("forcing_index", 1))
    forcing_init = bool(forc_sec.get("forcing_init", False))
    no_fields = bool(forc_sec.get("no_fields", False))
    forcing_kz = int(forc_sec.get("forcing_kz", 0))
    forcing_k2min = int(forc_sec.get("forcing_k2min", 0))
    forcing_k2max = int(forc_sec.get("forcing_k2max", 0))

    # [Boltzmann] (parameters.cu:L382-441)
    bz = _sec(nml, "Boltzmann")
    add_boltzmann_species = bool(bz.get("add_Boltzmann_species", False))
    btype = str(bz.get("Boltzmann_type", "electrons"))
    iphi00 = int(bz.get("iphi00", -2))
    if iphi00 > 0:
        if iphi00 == 1:
            btype = "Ions"
        elif iphi00 == 2:
            btype = "Electrons"
        add_boltzmann_species = True

    btype_low = btype.lower()
    if btype_low in ("electrons", "electron"):
        boltzmann_opt = BoltzmannOption.BOLTZMANN_ELECTRONS
    elif btype_low in ("ions", "ion"):
        boltzmann_opt = BoltzmannOption.BOLTZMANN_IONS
    else:
        raise ValueError(f"Unsupported Boltzmann_type={btype!r}.")

    ti_ov_te = float(bz.get("TiTe", 1.0))
    tau_fac_in = float(bz.get("tau_fac", -1.0))
    if tau_fac_in > 0.0:
        ti_ov_te = tau_fac_in
    tau_fac = ti_ov_te
    all_kinetic = not add_boltzmann_species

    # [Geometry] (parameters.cu:L444-511, geometry.cu:L20-23)
    geo = _sec(nml, "Geometry")
    geo_option = str(geo.get("geo_option", "none"))
    geofilename = str(geo.get("geofile", "eik.out"))
    geofilename = str(geo.get("geo_file", geofilename))
    vmec_file = str(geo.get("vmec_file", ""))
    slab = bool(geo.get("slab", False)) or (geo_option == "slab")
    const_curv = bool(geo.get("const_curv", False)) or (geo_option == "const-curv")
    igeo = int(geo.get("igeo", -1))
    beta_geo = float(geo.get("beta", 0.0))
    drhodpsi = float(geo.get("drhodpsi", 1.0))
    kxfac = float(geo.get("kxfac", 1.0))
    rmaj = float(geo.get("Rmaj", 1.0))
    rmaj = float(geo.get("rmaj", rmaj))
    r_geo = float(geo.get("R_geo", 1.0))
    shift = float(geo.get("shift", 0.0))
    eps = float(geo.get("eps", 0.167))
    qsf = float(geo.get("qinp", 1.4))
    akappa = float(geo.get("akappa", 1.0))
    akappri = float(geo.get("akappri", 0.0))
    tri = float(geo.get("tri", 1.0))
    tripri = float(geo.get("tripri", 0.0))
    betaprim = float(geo.get("beta_prime_input", -1.0))
    betaprim = float(geo.get("betaprim", betaprim))
    zero_shat = bool(geo.get("zero_shat", False))
    if geo_option in ("s-alpha", "slab", "const-curv") or igeo == 0:
        shat = float(geo.get("shat", 0.8))
    else:
        shat = float(geo.get("shat", 0.8))
    rbzeta_override = float(geo.get("RBzeta", 0.0))
    rhoc = float(geo.get("rhoc", 0.5))
    torflux = float(geo.get("torflux", geo.get("desired_normalized_toroidal_flux", 0.64)))
    npol = float(geo.get("npol", 1.0))
    npol_min_raw = geo.get("npol_min", None)
    npol_min = float(npol_min_raw) if npol_min_raw is not None else None
    alpha = float(geo.get("alpha", 0.0))

    # [Physics] (parameters.cu:L514-545, L313)
    phys = _sec(nml, "Physics")
    beta = float(phys.get("beta", 0.0))
    if beta == 0.0 and beta_geo > 0.0:
        beta = beta_geo
    nonlinear_mode = bool(phys.get("nonlinear_mode", ctrl_sec.get("nonlinear_mode", False)))
    linear = not nonlinear_mode
    g_exb = float(phys.get("g_exb", 0.0))
    exb_shear = bool(phys.get("ExBshear", g_exb != 0.0))
    exb_shear_phase = bool(phys.get("ExBshear_phase", True)) if exb_shear else False
    fphi = float(phys.get("fphi", 1.0))
    fapar = float(phys.get("fapar", 1.0 if beta > 0.0 else 0.0))
    fbpar = float(phys.get("fbpar", 1.0 if beta > 0.0 else 0.0))

    if not all_kinetic and boltzmann_opt == BoltzmannOption.BOLTZMANN_ELECTRONS and beta > 0.0:
        beta = 0.0
        fapar = 0.0
        fbpar = 0.0
    if (
        not all_kinetic
        and boltzmann_opt == BoltzmannOption.BOLTZMANN_IONS
        and fbpar > 0.0
        and beta > 0.0
    ):
        fbpar = 0.0

    ei_colls = bool(phys.get("ei_colls", True))
    coll_conservation = bool(phys.get("coll_conservation", True))

    if write_omega and fixed_amplitude and (nonlinear_mode or nwrite < 3):
        fixed_amplitude = False

    # [species] (parameters.cu:L547-573, L1055-1097)
    species_list: list[SpeciesConfig] = []
    if "species" in nml and isinstance(nml["species"], dict):
        sp_sec = nml["species"]
        zs = sp_sec["z"]
        masses = sp_sec["mass"]
        densities = sp_sec["dens"]
        temps = sp_sec.get("temp", [1.0] * nspec_in)
        tprims = sp_sec["tprim"]
        fprims = sp_sec["fprim"]
        vnewks = sp_sec.get("vnewk", [0.0] * nspec_in)
        stypes = sp_sec["type"]
        for i in range(nspec_in):
            species_list.append(
                SpeciesConfig.from_raw(
                    z=float(zs[i]),
                    mass=float(masses[i]),
                    dens=float(densities[i]),
                    temp=float(temps[i]),
                    tprim=float(tprims[i]),
                    fprim=float(fprims[i]),
                    nu_ss=float(vnewks[i]),
                    stype=str(stypes[i]),
                    beta=beta,
                    long_wavelength_gk=long_wavelength_gk,
                )
            )

    collisions = any(sp.nu_ss > 0.0 for sp in species_list)
    vtmax = max((sp.vt for sp in species_list), default=1.0)
    vtmin = min((sp.vt for sp in species_list), default=1.0)
    tzmax = max((abs(sp.tz) for sp in species_list), default=1.0)
    etamax = -1.0
    ne = 0.0
    te = 0.0
    for sp in species_list:
        if sp.fprim != 0.0:
            etamax = max(etamax, sp.tprim / sp.fprim)
        elif sp.tprim > 0.0:
            etamax = math.inf
        if sp.type_id == 1:
            ne = sp.dens
            te = sp.temp

    # Parallel grid resolution normalization (parameters.cu:L576-579)
    nz_in = ntheta_in
    if nz_in != 1:
        ntgrid = nz_in // 2 + (nperiod - 1) * nz_in
        nz_in = 2 * ntgrid

    if closure_model == "beer4+2":
        nm_in = 4
        nl_in = 2

    boundary_option_periodic = boundary in ("periodic", "exact periodic", "forced periodic")

    return GXConfig(
        run_name=run_name,
        input_path=in_path,
        nx=nx_in,
        ny=ny_in,
        nz=nz_in,
        ntheta_in=ntheta_in,
        nhermite=nm_in,
        nlaguerre=nl_in,
        nspecies=nspec_in,
        nperiod=nperiod,
        x0=x0,
        y0=y0,
        z0=z0,
        jtwist=jtwist,
        zp=zp,
        boundary=boundary,
        boundary_option_periodic=boundary_option_periodic,
        non_twist=non_twist,
        long_wavelength_gk=long_wavelength_gk,
        zero_shat_threshold=zero_shat_threshold,
        beta=beta,
        nonlinear_mode=nonlinear_mode,
        linear=linear,
        g_exb=g_exb,
        exb_shear=exb_shear,
        exb_shear_phase=exb_shear_phase,
        fphi=fphi,
        fapar=fapar,
        fbpar=fbpar,
        ei_colls=ei_colls,
        coll_conservation=coll_conservation,
        collisions=collisions,
        all_kinetic=all_kinetic,
        add_boltzmann_species=add_boltzmann_species,
        boltzmann_opt=boltzmann_opt,
        boltzmann_type=btype,
        tau_fac=tau_fac,
        geo_option=geo_option,
        geofilename=geofilename,
        vmec_file=vmec_file,
        igeo=igeo,
        slab=slab,
        const_curv=const_curv,
        drhodpsi=drhodpsi,
        kxfac=kxfac,
        rmaj=rmaj,
        r_geo=r_geo,
        shift=shift,
        eps=eps,
        qsf=qsf,
        shat=shat,
        akappa=akappa,
        akappri=akappri,
        tri=tri,
        tripri=tripri,
        betaprim=betaprim,
        rhoc=rhoc,
        torflux=torflux,
        npol=npol,
        npol_min=npol_min,
        alpha=alpha,
        zero_shat=zero_shat,
        rbzeta_override=rbzeta_override,
        dt=dt,
        dt_max=dt_max,
        dt_min=dt_min,
        fixed_dt=fixed_dt,
        nstep=nstep,
        scheme=scheme,
        cfl=cfl,
        stages=stages,
        t_max=t_max,
        init_field=init_field,
        init_amp=init_amp,
        kpar_init=kpar_init,
        ikpar_init=ikpar_init,
        random_init=random_init,
        gaussian_init=gaussian_init,
        gauss_env_const_coeff=gauss_env_const_coeff,
        gauss_env_sin_coeff=gauss_env_sin_coeff,
        gaussian_width=gaussian_width,
        init_electrons_only=init_electrons_only,
        random_seed=random_seed,
        closure_model=closure_model,
        smith_par_q=smith_par_q,
        smith_perp_q=smith_perp_q,
        hb_hyper=hb_hyper,
        d_hb=d_hb,
        p_hb=p_hb,
        w_osc=w_osc,
        dealias_kz=dealias_kz,
        local_limit=local_limit,
        init_single=init_single,
        ikx_single=ikx_single,
        iky_single=iky_single,
        damp_ends_widthfrac=damp_ends_widthfrac,
        damp_ends_amp=damp_ends_amp,
        hyper=hyper,
        d_hyper=d_hyper,
        p_hyper=p_hyper,
        hyperz=hyperz,
        nu_hyper_z=nu_hyper_z,
        p_hyper_z=p_hyper_z,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        p_hyper_l=p_hyper_l,
        p_hyper_m=p_hyper_m,
        p_hyper_lm=p_hyper_lm,
        nwrite=nwrite,
        navg=navg,
        nwrite_big=nwrite_big,
        fixed_amplitude=fixed_amplitude,
        write_omega=write_omega,
        write_free_energy=write_free_energy,
        write_fluxes=write_fluxes,
        write_moms=write_moms,
        write_fields=write_fields,
        restart=restart,
        restart_if_exists=restart_if_exists,
        save_for_restart=save_for_restart,
        restart_to_file=restart_to_file,
        restart_from_file=restart_from_file,
        restart_with_perturb=restart_with_perturb,
        append_on_restart=append_on_restart,
        restart_scale=restart_scale,
        nsave=nsave,
        forcing_init=forcing_init,
        forcing_type=forcing_type,
        stir_field=stir_field,
        forcing_amp=forcing_amp,
        pos_forcing_amp=pos_forcing_amp,
        neg_forcing_amp=neg_forcing_amp,
        forcing_index=forcing_index,
        no_fields=no_fields,
        forcing_kz=forcing_kz,
        forcing_k2min=forcing_k2min,
        forcing_k2max=forcing_k2max,
        species=tuple(species_list),
        vtmax=vtmax,
        vtmin=vtmin,
        tzmax=tzmax,
        etamax=etamax,
        ne=ne,
        te=te,
        debug=debug,
        fft_backend=fft_backend,
    )


def resolve_jtwist_x0(
    cfg: GXConfig,
    shat: float,
    gds21_0: float,
    gds22_0: float,
) -> GXConfig:
    """Compute final ``jtwist`` and ``x0`` from geometry (``Parameters::set_jtwist_x0``)."""
    if cfg.jtwist == 0:
        raise ValueError("jtwist = 0 is not allowed in GX.")

    twist_shift_geo_fac = 2.0 * shat * gds21_0 / gds22_0
    zero_shat = cfg.zero_shat
    boundary_periodic = cfg.boundary_option_periodic
    jtwist = cfg.jtwist
    x0 = cfg.x0

    if abs(shat) < cfg.zero_shat_threshold:
        zero_shat = True
        boundary_periodic = True

    if boundary_periodic:
        jtwist = 2 * cfg.nx
        if x0 == -1.0:
            x0 = cfg.y0
    else:
        if jtwist == -1000 and x0 < 0.0:
            jtwist = c_round(twist_shift_geo_fac)
            if jtwist == 0:
                jtwist = 1
            x0 = cfg.y0 * abs(jtwist) / abs(twist_shift_geo_fac)
        elif x0 < 0.0:
            x0 = cfg.y0 * abs(jtwist) / abs(twist_shift_geo_fac)
        else:
            jtwist_0 = c_round(twist_shift_geo_fac / cfg.y0 * x0)
            if jtwist_0 == 0:
                jtwist_0 = 1
            jtwist = jtwist_0
            x0 = cfg.y0 * abs(jtwist) / abs(twist_shift_geo_fac)

    return dataclasses.replace(
        cfg,
        jtwist=jtwist,
        x0=float(x0),
        zero_shat=zero_shat,
        boundary_option_periodic=boundary_periodic,
    )
