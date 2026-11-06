"""Static checks for Dockerfile.gx-runner and k8s/compute-classes.yaml (GCP-04, GCP-06, GCP-11).

docker, hadolint and kubectl are not available to the test suite, so these tests parse the files:
the Dockerfile as logical instructions per build stage (with the build's bash guards executed on
sample inputs), the compute classes with yaml.safe_load_all against the subset of the GKE
ComputeClass schema we use
(https://cloud.google.com/kubernetes-engine/docs/reference/crds/computeclass).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

GCP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GCP_DIR.parents[2]
DOCKERFILE = GCP_DIR / "Dockerfile.gx-runner"
COMPUTE_CLASSES = GCP_DIR / "k8s" / "compute-classes.yaml"
LOCAL_PYPROJECTS = (
    REPO_ROOT / "platform" / "hpc-core" / "pyproject.toml",
    REPO_ROOT / "workstreams" / "kr2-gk-database" / "hpc" / "pyproject.toml",
    GCP_DIR / "pyproject.toml",
)

Instruction = tuple[str, str]

# --------------------------------------------------------------------------- Dockerfile parsing


def _instructions(text: str) -> list[Instruction]:
    """(KEYWORD, args) per instruction, continuations joined, comment and blank lines dropped."""
    instructions: list[Instruction] = []
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buf.append(line[:-1].strip())
            continue
        buf.append(line)
        keyword, _, args = " ".join(buf).partition(" ")
        instructions.append((keyword.upper(), args.strip()))
        buf = []
    assert not buf, "dangling line continuation at end of Dockerfile"
    return instructions


def _global_args(instructions: list[Instruction]) -> dict[str, str | None]:
    args: dict[str, str | None] = {}
    for keyword, rest in instructions:
        if keyword == "FROM":
            break
        if keyword == "ARG":
            name, sep, default = rest.partition("=")
            args[name] = default.strip('"') if sep else None
    return args


def _stages(instructions: list[Instruction]) -> list[list[Instruction]]:
    stages: list[list[Instruction]] = []
    for keyword, rest in instructions:
        if keyword == "FROM":
            stages.append([(keyword, rest)])
        elif stages:
            stages[-1].append((keyword, rest))
    return stages


def _stage(stages: list[list[Instruction]], name: str) -> list[Instruction]:
    matches = [s for s in stages if s[0][1].lower().endswith(f" as {name}")]
    assert len(matches) == 1, f"expected exactly one stage named {name}"
    return matches[0]


def _values(stage: list[Instruction], keyword: str) -> list[str]:
    return [rest for kw, rest in stage if kw == keyword]


def _env_tokens(stage: list[Instruction]) -> list[str]:
    return " ".join(_values(stage, "ENV")).split()


_APT_INSTALL = re.compile(r"apt-get install\s+(?P<args>.*?)(?:;|&&|$)")
_PIP_INSTALL = re.compile(r"python3 -m pip install\s+(?P<args>.*?)(?:;|&&|$)")


def _apt_packages(stage: list[Instruction]) -> list[str]:
    packages: list[str] = []
    for run in _values(stage, "RUN"):
        for match in _APT_INSTALL.finditer(run):
            packages += [
                tok for tok in match.group("args").split() if not tok.startswith(("-", '"$'))
            ]
    return packages


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def instructions(dockerfile_text: str) -> list[Instruction]:
    return _instructions(dockerfile_text)


@pytest.fixture(scope="module")
def stages(instructions: list[Instruction]) -> list[list[Instruction]]:
    return _stages(instructions)


@pytest.fixture(scope="module")
def builder_run(stages: list[list[Instruction]]) -> str:
    return " ".join(_values(_stage(stages, "gx-builder"), "RUN"))


def _bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **env},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# --------------------------------------------------------------------------- base images


def test_base_images_are_pinned_by_tag_to_published_noble_cuda_images(
    instructions: list[Instruction], stages: list[list[Instruction]]
) -> None:
    global_args = _global_args(instructions)
    cuda = global_args["CUDA_VERSION"]
    assert cuda is not None and re.fullmatch(r"\d+\.\d+\.\d+", cuda)
    # nvidia/cuda publishes ubuntu24.04 images only from CUDA 12.5 on: the previous
    # 12.4.1-*-ubuntu24.04 tags do not exist, so that build could never start.
    assert tuple(int(part) for part in cuda.split(".")[:2]) >= (12, 5)
    assert [stage[0][1] for stage in stages] == [
        "nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04"
        "${CUDA_DEVEL_DIGEST:+@${CUDA_DEVEL_DIGEST}} AS gx-builder",
        "nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04"
        "${CUDA_RUNTIME_DIGEST:+@${CUDA_RUNTIME_DIGEST}} AS runtime",
    ]
    for name in ("CUDA_DEVEL_DIGEST", "CUDA_RUNTIME_DIGEST"):
        digest = global_args[name]
        # Empty = documented opt-in; a committed value must be a full sha256 digest.
        assert digest in ("", None) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest)


# --------------------------------------------------------------------------- GCP-06 runtime


def test_runtime_stage_runs_as_non_root_uid_10001_with_home_tmp(
    stages: list[list[Instruction]],
) -> None:
    final = stages[-1]
    assert "-runtime-ubuntu24.04" in final[0][1]
    users = _values(final, "USER")
    assert users == ["10001:10001"]
    keywords = [kw for kw, _ in final]
    last_run = max(i for i, kw in enumerate(keywords) if kw == "RUN")
    assert keywords.index("USER") > last_run, "USER must follow every RUN"
    assert keywords[-1] == "ENTRYPOINT"
    runs = _values(final, "RUN")
    assert any(
        "groupadd --gid 10001 gx" in run
        and "useradd --uid 10001 --gid 10001 --home-dir /tmp --no-create-home" in run
        for run in runs
    )
    assert any("chmod 1777 /tmp" in run for run in runs)
    assert "HOME=/tmp" in _env_tokens(final)
    assert "HDF5_USE_FILE_LOCKING=FALSE" in _env_tokens(final)


def test_runtime_stage_installs_no_dev_or_toolchain_packages(
    stages: list[list[Instruction]], builder_run: str
) -> None:
    final = stages[-1]
    packages = _apt_packages(final)
    assert {
        "python3",
        "python-is-python3",
        "python3-netcdf4",
        "python3-numpy",
        "python3-scipy",
        "ca-certificates",
    } <= set(packages)
    assert [p for p in packages if re.search(r"-dev($|[=:])", p)] == []
    toolchain = {"build-essential", "gcc", "g++", "make", "git", "pkg-config", "cuda-nvcc"}
    assert toolchain.isdisjoint(packages)
    # GX's own runtime libraries are the Debian owners of what `ldd gx` resolves in the builder;
    # the builder rejects -dev owners and the runtime stage re-checks that ldd resolves.
    assert "ldd /usr/local/bin/gx" in builder_run and 'dpkg -S "${f}"' in builder_run
    assert "grep -E -- '-dev$' /opt/gx/runtime-packages.txt" in builder_run
    runtime_runs = " ".join(_values(final, "RUN"))
    assert "done < /opt/gx/runtime-packages.txt" in runtime_runs
    assert "ldd /usr/local/bin/gx | grep 'not found'" in runtime_runs
    assert "apt-get purge -y --auto-remove python3-pip python3-setuptools python3-wheel" in (
        runtime_runs
    )
    assert (
        'command -v python && test -f /opt/gx-src/geometry_modules/miller/gx_geo.py && python -c "import numpy, scipy"'
        in runtime_runs
    )
    copies = [f"COPY {rest}" for rest in _values(final, "COPY")]
    assert "COPY --from=gx-builder /usr/local/bin/gx /usr/local/bin/gx" in copies
    assert (
        "COPY --from=gx-builder /opt/gx/geometry_modules/ /opt/gx-src/geometry_modules/" in copies
    )
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" not in _env_tokens(final)


def test_builder_installs_mpi_netcdf_and_verifies_mpi_netcdf_linkage(
    stages: list[list[Instruction]], builder_run: str
) -> None:
    builder_pkgs = set(_apt_packages(_stage(stages, "gx-builder")))
    assert {"libnetcdf-mpi-dev", "libhdf5-openmpi-dev", "libnccl-dev"} <= builder_pkgs
    assert "libnetcdf-dev" not in builder_pkgs
    assert "netcdf/mpi/lib/libnetcdf" in builder_run
    assert "test -f /opt/gx-src/geometry_modules/miller/gx_geo.py" in builder_run
    pyproject = tomllib.loads((GCP_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["optional-dependencies"]["nc"] == ["netCDF4>=1.6.0"]


# --------------------------------------------------------------------------- GCP-11 pinning


def test_gx_commit_is_required_and_verified_after_checkout(
    stages: list[list[Instruction]], dockerfile_text: str, builder_run: str
) -> None:
    builder_args = _values(_stage(stages, "gx-builder"), "ARG")
    assert "GX_COMMIT" in builder_args, "GX_COMMIT must be declared without a default"
    assert not [a for a in builder_args if a.startswith("GX_COMMIT=")]
    assert "ARG GX_REPO=https://bitbucket.org/gyrokinetics/gx.git" in dockerfile_text
    assert "GX_REPO_REF" not in dockerfile_text
    assert "--branch" not in builder_run and "--depth" not in builder_run
    clone = builder_run.index('git clone --no-checkout "${GX_REPO}" /opt/gx-src')
    checkout = builder_run.index('checkout --detach "${GX_COMMIT}"')
    verify = builder_run.index('if [[ "${gx_head}" != "${GX_COMMIT}" ]]')
    build = builder_run.index("make -C /opt/gx-src")
    assert clone < checkout < verify < build
    assert 'gx_head="$(git -C /opt/gx-src rev-parse HEAD)"' in builder_run


_COMMIT_GUARD = re.compile(r'if \[\[ ! "\$\{GX_COMMIT:-\}" =~ .*?; fi;')
_SHA = "0123456789abcdef0123456789abcdef01234567"


@needs_bash
@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (_SHA, True),
        (None, False),
        ("", False),
        ("main", False),
        ("v1.0", False),
        (_SHA[:-1], False),
        (_SHA.upper(), False),
        (_SHA + "0", False),
        (_SHA + "\nmain", False),
    ],
)
def test_gx_commit_guard_accepts_only_a_full_lowercase_sha(
    builder_run: str, value: str | None, accepted: bool
) -> None:
    guard = _COMMIT_GUARD.search(builder_run)
    assert guard is not None
    env = {} if value is None else {"GX_COMMIT": value}
    proc = _bash(guard.group(0) + " echo guard-passed", env)
    assert (proc.returncode == 0) is accepted, proc.stderr
    assert ("guard-passed" in proc.stdout) is accepted


_BUILD_INFO_PRINTF = re.compile(
    r"printf '(?P<fmt>\{\"schema_version\".*?\})\\n' (?P<args>.*?) > /opt/gx/gx_build\.json"
)


def test_gx_build_info_is_written_from_verified_values_and_exported(
    stages: list[list[Instruction]], builder_run: str
) -> None:
    match = _BUILD_INFO_PRINTF.search(builder_run)
    assert match is not None
    assert match.group("args").split() == [
        '"${GX_REPO}"',
        '"${GX_COMMIT}"',
        '"${CUDA_VERSION}"',
        '"${gx_archs}"',
    ]
    archs = '"sm_75", "sm_80", "sm_89", "compute_89"'
    rendered = match.group("fmt") % (
        "https://bitbucket.org/gyrokinetics/gx.git",
        _SHA,
        "12.6.3",
        archs,
    )
    assert json.loads(rendered) == {
        "schema_version": 1,
        "gx_repo": "https://bitbucket.org/gyrokinetics/gx.git",
        "gx_commit": _SHA,
        "cuda_version": "12.6.3",
        "cuda_archs": ["sm_75", "sm_80", "sm_89", "compute_89"],
        "real_gx": True,
    }
    # Written only after the commit and device-code checks have passed.
    assert builder_run.index('if [[ "${gx_head}" != "${GX_COMMIT}" ]]') < match.start()
    assert builder_run.index("grep -qx compute_89") < match.start()
    final = stages[-1]
    assert "GX_BUILD_INFO=/opt/gx/gx_build.json" in _env_tokens(final)
    assert "--from=gx-builder /opt/gx/ /opt/gx/" in _values(final, "COPY")


# --------------------------------------------------------------------------- GPU archs


def test_nvcc_targets_sass_for_t4_a100_l4_plus_compute_89_ptx(
    stages: list[list[Instruction]],
) -> None:
    env = " ".join(_values(_stage(stages, "gx-builder"), "ENV"))
    match = re.search(r'NVCC_APPEND_FLAGS="([^"]*)"', env)
    assert match is not None
    assert set(match.group(1).split()) == {
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=compute_89",
    }


def _arch_check_script(builder_run: str, sass: Path, ptx: Path) -> str:
    patterns = (
        r'gx_sass="\$\(.*?\)";',
        r'gx_ptx="\$\(.*?\)";',
        r"for arch in sm_75 sm_80 sm_89; do .*?; done;",
        r"grep -qx compute_89 .*?; \};",
        r'gx_archs="\$\(.*?\)";',
    )
    parts = []
    for pattern in patterns:
        match = re.search(pattern, builder_run)
        assert match is not None, pattern
        parts.append(match.group(0))
    script = " ".join(parts) + ' printf "%s" "${gx_archs}"'
    return script.replace("/tmp/gx-sass.txt", str(sass)).replace("/tmp/gx-ptx.txt", str(ptx))


@needs_bash
@pytest.mark.parametrize(
    ("cubins", "ptx_archs", "archs_json", "error"),
    [
        (
            ["sm_80", "sm_75", "sm_89", "sm_75"],
            ["sm_89"],
            '"sm_75", "sm_80", "sm_89", "compute_89"',
            None,
        ),
        (
            ["sm_75", "sm_80", "sm_89", "sm_90"],
            ["sm_80", "sm_89"],
            '"sm_75", "sm_80", "sm_89", "sm_90", "compute_80", "compute_89"',
            None,
        ),
        # Some cuobjdump versions name embedded PTX after the virtual arch.
        (
            ["sm_75", "sm_80", "sm_89"],
            ["compute_89"],
            '"sm_75", "sm_80", "sm_89", "compute_89"',
            None,
        ),
        (["sm_80", "sm_89"], ["sm_89"], None, "no sm_75 SASS"),
        (["sm_75", "sm_80"], ["sm_89"], None, "no sm_89 SASS"),
        (["sm_75", "sm_80", "sm_89"], ["sm_80"], None, "no compute_89 PTX"),
        ([], [], None, "no sm_75 SASS"),
    ],
)
def test_build_fails_unless_gx_embeds_required_sass_and_ptx(
    builder_run: str,
    tmp_path: Path,
    cubins: list[str],
    ptx_archs: list[str],
    archs_json: str | None,
    error: str | None,
) -> None:
    sass = tmp_path / "sass.txt"
    ptx = tmp_path / "ptx.txt"
    # cuobjdump --list-elf / --list-ptx output format.
    sass.write_text(
        "".join(f"ELF file    {i}: gx.{i}.{a}.cubin\n" for i, a in enumerate(cubins, 1)),
        encoding="utf-8",
    )
    ptx.write_text(
        "".join(f"PTX file    {i}: gx.{i}.{a}.ptx\n" for i, a in enumerate(ptx_archs, 1)),
        encoding="utf-8",
    )
    proc = _bash(_arch_check_script(builder_run, sass, ptx), {})
    if error is not None:
        assert proc.returncode != 0
        assert error in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == archs_json


# --------------------------------------------------------------------------- Python (QA-02)


def test_runtime_python_is_noble_312_inside_requires_python(
    stages: list[list[Instruction]],
) -> None:
    runtime_runs = " ".join(_values(stages[-1], "RUN"))
    assert "assert sys.version_info[:2] == (3, 12)" in runtime_runs
    for pyproject in LOCAL_PYPROJECTS:
        requires = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "requires-python"
        ]
        assert SpecifierSet(requires).contains("3.12"), (pyproject, requires)


def test_pip_installs_are_exactly_pinned_and_resolve_nothing(
    stages: list[list[Instruction]],
) -> None:
    installs = [
        match.group("args").split()
        for run in _values(stages[-1], "RUN")
        for match in _PIP_INSTALL.finditer(run)
    ]
    assert len(installs) == 2
    third_party, local = installs
    for args in installs:
        assert {"--no-deps", "--no-cache-dir", "--break-system-packages"} <= set(args)
    assert "--only-binary=:all:" in third_party
    pins = {
        canonicalize_name(name): version
        for name, version in (tok.split("==") for tok in third_party if not tok.startswith("-"))
    }
    assert all(re.fullmatch(r"\d+(\.\d+)*", v) for v in pins.values()), pins
    assert {"--no-index", "--no-build-isolation"} <= set(local)
    assert [tok for tok in local if not tok.startswith("-")] == [
        "/opt/ut-austin/platform/hpc-core",
        "/opt/ut-austin/workstreams/kr2-gk-database/hpc",
        "/opt/ut-austin/workstreams/kr2-gk-database/gcp",
    ]
    local_names = set()
    deps: list[Requirement] = []
    for pyproject in LOCAL_PYPROJECTS:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        local_names.add(canonicalize_name(project["name"]))
        deps += [Requirement(dep) for dep in project.get("dependencies", [])]
    for req in deps:
        name = canonicalize_name(req.name)
        if name in local_names:
            continue
        assert name in pins, f"{req} is not pinned in Dockerfile.gx-runner"
        assert req.specifier.contains(pins[name]), (str(req), pins[name])
    # requests' own runtime dependencies are pinned too, so nothing is resolved transitively.
    assert {"urllib3", "idna", "certifi", "charset-normalizer"} <= set(pins)


# --------------------------------------------------------------------------- compute classes

DEFAULT_CLASSES = ("gx-gpu-obtainable", "gx-gpu-l4-or-a100", "gx-gpu-highmem", "gx-cpu")
GPU_FOR_FAMILY: dict[str, str | None] = {
    "g2": "nvidia-l4",
    "n1": "nvidia-tesla-t4",
    "a2": "nvidia-tesla-a100",
    "e2": None,
}


@pytest.fixture(scope="module")
def compute_classes() -> dict[str, dict[str, Any]]:
    docs = [
        doc
        for doc in yaml.safe_load_all(COMPUTE_CLASSES.read_text(encoding="utf-8"))
        if doc is not None
    ]
    by_name = {doc["metadata"]["name"]: doc for doc in docs}
    assert len(by_name) == len(docs), "duplicate ComputeClass names"
    return by_name


def test_compute_classes_follow_the_gke_computeclass_schema(
    compute_classes: dict[str, dict[str, Any]],
) -> None:
    for name, doc in compute_classes.items():
        assert doc["apiVersion"] == "cloud.google.com/v1"
        assert doc["kind"] == "ComputeClass"
        assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name) and len(name) <= 63
        spec = doc["spec"]
        assert set(spec) == {"priorities", "nodePoolAutoCreation", "whenUnsatisfiable"}, name
        assert spec["nodePoolAutoCreation"] == {"enabled": True}
        # ScaleUpAnyway would fall back to the cluster's default (on-demand) machine series.
        assert spec["whenUnsatisfiable"] == "DoNotScaleUp", name
        assert isinstance(spec["priorities"], list) and spec["priorities"], name
        for rule in spec["priorities"]:
            assert set(rule) <= {"machineFamily", "gpu", "spot", "flexStart"}, (name, rule)
            # Exactly one provisioning model per rule; flex-start cannot be combined with spot.
            assert ("spot" in rule) != ("flexStart" in rule), (name, rule)
            if "spot" in rule:
                assert isinstance(rule["spot"], bool), (name, rule)
            else:
                assert rule["flexStart"] == {"enabled": True}, (name, rule)
            family = rule["machineFamily"]
            assert family in GPU_FOR_FAMILY, (name, rule)
            if GPU_FOR_FAMILY[family] is None:
                assert "gpu" not in rule, (name, rule)
                continue
            gpu = rule["gpu"]
            assert set(gpu) == {"type", "count", "driverVersion"}, (name, rule)
            assert gpu["type"] == GPU_FOR_FAMILY[family], (name, rule)
            assert gpu["count"] == 1, (name, rule)
            assert gpu["driverVersion"] == "latest", (name, rule)


def test_default_compute_classes_are_spot_only(
    compute_classes: dict[str, dict[str, Any]],
) -> None:
    for name in DEFAULT_CLASSES:
        doc = compute_classes[name]
        rules = doc["spec"]["priorities"]
        assert all(rule.get("spot") is True for rule in rules), name
        assert not any("flexStart" in rule for rule in rules), name
        assert doc["metadata"]["labels"]["provisioning"] == "spot-only"


def test_on_demand_and_flex_start_live_only_in_opt_in_ondemand_classes(
    compute_classes: dict[str, dict[str, Any]],
) -> None:
    assert set(compute_classes) == set(DEFAULT_CLASSES) | {f"{n}-ondemand" for n in DEFAULT_CLASSES}
    for name in DEFAULT_CLASSES:
        default_rules = compute_classes[name]["spec"]["priorities"]
        opt_in = compute_classes[f"{name}-ondemand"]
        rules = opt_in["spec"]["priorities"]
        # Same Spot options in the same order, Spot first; the opt-in only adds paid fallbacks.
        assert rules[0] == default_rules[0], name
        assert [rule for rule in rules if rule.get("spot") is True] == default_rules, name
        assert any(rule.get("spot") is not True for rule in rules), name
        assert opt_in["metadata"]["labels"]["provisioning"] == "on-demand-opt-in"


def test_gpu_classes_respect_vram_tiers(compute_classes: dict[str, dict[str, Any]]) -> None:
    def gpu_types(name: str) -> set[str]:
        rules = compute_classes[name]["spec"]["priorities"]
        return {rule["gpu"]["type"] for rule in rules if "gpu" in rule}

    for suffix in ("", "-ondemand"):
        assert gpu_types(f"gx-gpu-obtainable{suffix}") == {
            "nvidia-l4",
            "nvidia-tesla-t4",
            "nvidia-tesla-a100",
        }
        # 13.5-20 GB VRAM: never the 16 GB T4.
        assert gpu_types(f"gx-gpu-l4-or-a100{suffix}") == {"nvidia-l4", "nvidia-tesla-a100"}
        assert gpu_types(f"gx-gpu-highmem{suffix}") == {"nvidia-tesla-a100"}
        assert gpu_types(f"gx-cpu{suffix}") == set()
