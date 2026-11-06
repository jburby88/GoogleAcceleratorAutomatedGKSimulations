"""Tests for scripts/check_boundaries_and_deps.py, the TACC zero-AI/cloud-SDK CI gate.

Fake secrets and machine paths are assembled at runtime (see ``_fake``) so that the gate's
real-repo hygiene scan, which also reads this file, never sees them as literals.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_boundaries_and_deps.py"

HPC_CORE = Path("platform/hpc-core/hpc_core")
KR2_HPC = Path("workstreams/kr2-gk-database/hpc")
SMOKE = Path("scripts/tacc_smoke_test.py")


def _load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_boundaries_and_deps", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fake(*parts: str) -> str:
    """Join fragments at runtime so no secret-shaped literal exists in this file."""
    return "".join(parts)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal clean TACC tree, installed as the gate's REPO_ROOT."""
    _write(tmp_path / HPC_CORE / "__init__.py", "import json\n")
    _write(tmp_path / KR2_HPC / "tacc_exec" / "__init__.py", "from hpc_core import client\n")
    _write(tmp_path / SMOKE, "import json\nimport urllib.request\n")
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    return tmp_path


# --- AST import scan -------------------------------------------------------------------------


def test_clean_tree_passes(repo: Path) -> None:
    assert gate.check_ast_imports() == []


def test_relative_imports_are_accepted(repo: Path) -> None:
    _write(
        repo / HPC_CORE / "client.py",
        "from .protocol import DiagnosticSummary\nfrom . import state\n",
    )
    _write(
        repo / KR2_HPC / "tacc_exec" / "patchers" / "__init__.py",
        "from .gx_toml import patch_gx_toml\nfrom ..watchdog import SimulationWatchdog\n",
    )
    assert gate.check_ast_imports() == []


CLOUD_IMPORTS = [
    "import google.cloud.storage",
    "from google.cloud import bigquery",
    "import vertexai",
    "from vertexai.generative_models import GenerativeModel",
    "import openai",
    "from openai import OpenAI",
]
# tools/ sits outside tacc_exec/: the whole kr2 hpc/ tree is TACC runtime.
TACC_FILES = [
    HPC_CORE / "client.py",
    KR2_HPC / "tacc_exec" / "driver.py",
    KR2_HPC / "tools" / "sweep.py",
    SMOKE,
]


@pytest.mark.parametrize("target", TACC_FILES, ids=str)
@pytest.mark.parametrize("statement", CLOUD_IMPORTS)
def test_cloud_and_ai_sdk_imports_are_rejected(repo: Path, target: Path, statement: str) -> None:
    _write(repo / target, f"import json\n{statement}\n")
    errors = gate.check_ast_imports()
    module = statement.split()[1]
    assert len(errors) == 1, errors
    assert f"Forbidden import {module!r}" in errors[0]
    assert errors[0].endswith(f"{target.as_posix()}:2")


@pytest.mark.parametrize("module", ["requests", "f90nml", "tomlkit", "hpc_core"])
def test_requests_rejected_in_smoke_test_but_accepted_in_hpc_core(repo: Path, module: str) -> None:
    _write(repo / HPC_CORE / "client.py", f"import {module}\n")
    _write(repo / KR2_HPC / "tacc_exec" / "driver.py", f"from {module} import api\n")
    _write(repo / SMOKE, f"import json\nimport {module}\n")
    assert gate.check_ast_imports() == [
        f"[AST] Forbidden import {module!r} in stdlib-only file scripts/tacc_smoke_test.py:2"
    ]


DYNAMIC_SNIPPETS = [
    pytest.param("exec('x = 1')", id="exec"),
    pytest.param("eval('1 + 1')", id="eval"),
    pytest.param("compile('1', '<s>', 'eval')", id="compile"),
    pytest.param("__import__('vertexai')", id="__import__"),
    pytest.param("import importlib\nimportlib.import_module('vertexai')", id="import_module"),
    pytest.param(
        "from importlib import import_module as load\nload('vertexai')", id="aliased-import_module"
    ),
    pytest.param("from importlib import *\nimport_module('vertexai')", id="bare-import_module"),
    pytest.param(
        "import importlib.util\nimportlib.util.spec_from_file_location('m', 'm.py')",
        id="spec_from_file_location",
    ),
    pytest.param("from importlib import util\nutil.module_from_spec(spec)", id="module_from_spec"),
    pytest.param("loader.exec_module(module)", id="exec_module"),
    pytest.param("import builtins\nbuiltins.__import__('vertexai')", id="builtins.__import__"),
    pytest.param("import builtins\nbuiltins.exec('x = 1')", id="builtins.exec"),
    pytest.param("import builtins\ngetattr(builtins, 'exec')('x = 1')", id="getattr-builtins"),
    pytest.param(
        "import importlib\ngetattr(importlib, 'import_module')('vertexai')",
        id="getattr-import_module",
    ),
    pytest.param("import runpy\nrunpy.run_module('vertexai')", id="runpy"),
]


@pytest.mark.parametrize("snippet", DYNAMIC_SNIPPETS)
def test_dynamic_code_execution_is_rejected(repo: Path, snippet: str) -> None:
    _write(repo / KR2_HPC / "tacc_exec" / "driver.py", snippet + "\n")
    errors = gate.check_ast_imports()
    assert errors, snippet
    assert all("Forbidden dynamic" in e and "tacc_exec/driver.py" in e for e in errors), errors


def test_lookalike_calls_are_accepted(repo: Path) -> None:
    _write(
        repo / HPC_CORE / "client.py",
        "import re\nimport importlib.metadata\nfrom importlib import metadata\n"
        "from os.path import *\nfrom importlib.util import find_spec\n"
        "PATTERN = re.compile('x')\nVERSION = importlib.metadata.version('requests')\n"
        "retries = getattr(object(), 'retries', 0)\n",
    )
    assert gate.check_ast_imports() == []


def test_syntax_error_is_reported_not_raised(repo: Path) -> None:
    _write(repo / HPC_CORE / "broken.py", "def broken(:\n    pass\n")
    errors = gate.check_ast_imports()
    assert len(errors) == 1
    assert errors[0].startswith(
        "[AST] Cannot parse TACC-safe file platform/hpc-core/hpc_core/broken.py:1: SyntaxError"
    )


@pytest.mark.parametrize(
    "excluded",
    [
        KR2_HPC / "tests" / "test_driver.py",
        HPC_CORE / "tests" / "helpers.py",
        KR2_HPC / "build" / "lib" / "tacc_exec" / "driver.py",
        KR2_HPC / "ut_kr2_tacc_exec.egg-info" / "stale.py",
        HPC_CORE / "__pycache__" / "client.py",
    ],
    ids=str,
)
def test_tests_and_generated_dirs_are_not_scanned(repo: Path, excluded: Path) -> None:
    _write(repo / excluded, "import pytest\nimport vertexai\n")
    assert gate.check_ast_imports() == []


def test_ancestor_dirs_named_build_or_venv_do_not_hide_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "build" / ".venv" / "checkout"
    _write(root / HPC_CORE / "client.py", "import openai\n")
    _write(root / KR2_HPC / "tacc_exec" / "__init__.py", "")
    _write(root / SMOKE, "import json\n")
    _write(root / "config.txt", _fake("-----BEGIN ", "PRIVATE KEY", "-----\n"))
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors = gate.check_ast_imports()
    assert len(errors) == 1 and "'openai'" in errors[0], errors
    assert gate.check_public_repo_hygiene() == [
        "[PUBLIC-HYGIENE] config.txt contains forbidden PEM private key header"
    ]


def test_missing_runtime_paths_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    assert gate.check_ast_imports() == [
        "[AST] TACC runtime directory missing: platform/hpc-core/hpc_core",
        "[AST] TACC runtime directory missing: workstreams/kr2-gk-database/hpc",
        "[AST] TACC smoke test missing: scripts/tacc_smoke_test.py",
    ]


# --- Public-repo hygiene ---------------------------------------------------------------------

FAKE_FINDINGS = [
    pytest.param(_fake("AI", "za", "x" * 35), "Google API key", id="google-api-key"),
    pytest.param(
        _fake("-----BEGIN ", "RSA PRIVATE KEY", "-----"), "PEM private key header", id="pem-rsa"
    ),
    pytest.param(
        _fake("-----BEGIN ", "PRIVATE KEY", "-----"), "PEM private key header", id="pem-pkcs8"
    ),
    pytest.param(_fake("/ho", "me/", "alice/runs/gx.in"), "local Linux home path", id="home-path"),
    pytest.param(_fake("/Us", "ers/", "alice"), "local workstation user path", id="users-path"),
    pytest.param(_fake("gh", "p_", "A1" * 18), "GitHub token", id="github-token"),
    pytest.param(_fake("github", "_pat_", "B2" * 11), "GitHub fine-grained token", id="github-pat"),
    pytest.param(
        _fake("https://hooks.", "slack.com/services/", "T0123ABCD/B0123ABCD/", "c" * 24),
        "Slack webhook URL",
        id="slack-webhook",
    ),
    pytest.param(
        json.dumps(
            {_fake("ty", "pe"): _fake("service", "_account"), _fake("private", "_key"): "redacted"}
        ),
        "GCP service-account key",
        id="service-account-json",
    ),
]


@pytest.mark.parametrize(("secret", "label"), FAKE_FINDINGS)
def test_hygiene_flags_secrets_and_machine_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret: str, label: str
) -> None:
    _write(tmp_path / "config" / "leak.txt", f"value = {secret}\n")
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    errors = gate.check_public_repo_hygiene()
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"[PUBLIC-HYGIENE] config/leak.txt contains forbidden {label}")
    assert secret not in errors[0]  # file + label only, never the value


@pytest.mark.parametrize(
    "skipped_dir",
    [
        "platform/infra/terraform/.terraform/providers",
        "platform/dashboard/node_modules/pkg",
        "platform/hpc-core/build/lib",
        "dist",
        "platform/hpc-core/ut_hpc_core.egg-info",
        ".venv-tacc/lib",
        "scripts/__pycache__",
        "docs/review",
    ],
)
def test_hygiene_skips_generated_and_local_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skipped_dir: str
) -> None:
    leak = _fake("AI", "za", "y" * 35, "\n", "-----BEGIN ", "PRIVATE KEY", "-----\n")
    _write(tmp_path / skipped_dir / "state.txt", leak)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    assert gate.check_public_repo_hygiene() == []
    _write(tmp_path / "src" / "state.txt", leak)  # the same content outside it is flagged
    assert len(gate.check_public_repo_hygiene()) == 2


def test_hygiene_ignores_slack_host_mentioned_in_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "README.md", _fake("Paste your https://hooks.", "slack.com/services/... URL"))
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    assert gate.check_public_repo_hygiene() == []


# --- pyproject allowlist ---------------------------------------------------------------------

CLEAN_KR2_PYPROJECT = (
    '[project]\nname = "ut-kr2-tacc-exec"\n'
    'dependencies = ["ut-hpc-core", "f90nml>=1.4.4", "tomlkit", "requests>=2.31"]\n'
    '[project.optional-dependencies]\ndev = ["pytest>=8", "ruff"]\n'
)


@pytest.mark.parametrize(
    "hpc_core_pyproject",
    [
        pytest.param(
            '[project]\nname = "ut-hpc-core"\n'
            'dependencies = ["requests>=2.31", "google-cloud-storage>=2.0"]\n',
            id="dependencies",
        ),
        pytest.param(
            '[project]\nname = "ut-hpc-core"\ndependencies = ["requests"]\n'
            '[project.optional-dependencies]\ngcp = ["google_cloud.storage[grpc]>=2"]\n',
            id="optional-dependencies",
        ),
        pytest.param(
            '[build-system]\nrequires = ["setuptools>=68", "google-cloud-storage"]\n'
            '[project]\nname = "ut-hpc-core"\ndependencies = ["requests"]\n',
            id="build-system",
        ),
    ],
)
def test_pyproject_check_rejects_disallowed_dep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hpc_core_pyproject: str
) -> None:
    _write(tmp_path / "platform/hpc-core/pyproject.toml", hpc_core_pyproject)
    _write(tmp_path / KR2_HPC / "pyproject.toml", CLEAN_KR2_PYPROJECT)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    assert gate.check_pyproject_dependencies() == [
        "[PYPROJECT] platform/hpc-core/pyproject.toml declares disallowed runtime deps: "
        "['google-cloud-storage']"
    ]


def test_pyproject_missing_or_invalid_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "platform/hpc-core/pyproject.toml", "[project\nname = ")
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    errors = gate.check_pyproject_dependencies()
    assert [e.split(":")[0] for e in errors] == [
        "[PYPROJECT] Cannot parse platform/hpc-core/pyproject.toml",
        "[PYPROJECT] Cannot parse workstreams/kr2-gk-database/hpc/pyproject.toml",
    ]


# --- OKF <-> example config parity -----------------------------------------------------------

OKF_DOC = """---
id: allowed-parameters
triggers: ["ALL"]
# Comment lines are ignored.
parameters:
  nx:
    type: int
    min: 16
    max: 512
    gene_path: "box.nx0"
    gx_path: "Dimensions.nx"
  dt_max:
    type: float
    min: 1.0e-6
    max: 0.1
    gene_path: "general.dt_max"
    gx_path: "Time.dt"
    warm_restart_allowed: true
  # nv:
  #   type: int
---

# Allowlisted parameters
"""


def _config_params() -> dict[str, dict[str, object]]:
    return {
        "nx": {
            "type": "int",
            "min": 16,
            "max": 512,
            "gene_path": "box.nx0",
            "gx_path": "Dimensions.nx",
        },
        "dt_max": {
            "type": "float",
            "min": 1e-06,
            "max": 0.1,
            "gene_path": "general.dt_max",
            "gx_path": "Time.dt",
            "warm_restart_allowed": True,
        },
    }


def _parity_tree(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    okf: str = OKF_DOC,
    gcp_params: dict[str, dict[str, object]] | None = None,
) -> None:
    _write(root / gate.OKF_ALLOWED_PARAMETERS, okf)
    hpc_cfg, gcp_cfg = gate.PARITY_CONFIGS
    _write(root / hpc_cfg, json.dumps({"allowed_parameters": _config_params()}))
    params = _config_params() if gcp_params is None else gcp_params
    _write(root / gcp_cfg, json.dumps({"allowed_parameters": params}))
    monkeypatch.setattr(gate, "REPO_ROOT", root)


OKF_REL = "workstreams/kr2-gk-database/skill/sim-babysitter/okf/concepts/allowed-parameters.md"
GCP_REL = "workstreams/kr2-gk-database/gcp/config.batch.example.json"


def test_parity_accepts_consistent_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _parity_tree(tmp_path, monkeypatch)
    assert gate.check_okf_and_client_config_parity() == []


def test_parity_parser_accepts_comments_quotes_and_wider_indent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    okf = (
        "---\ntriggers:\n- ALL\ntags: [\n  numerics\n]\nparameters:\n"
        "    nx:   # radial modes\n"
        "        type: 'int'\n        min: 16  # floor\n        max: 512\n"
        "        gene_path: 'box.nx0'\n        gx_path: Dimensions.nx\n"
        "        warm_restart_allowed: false\n"
        "    dt_max:\n"
        "        type: float\n        min: 1.0e-6\n        max: .1\n"
        '        gene_path: "general.dt_max"  # GENE &general\n        gx_path: "Time.dt"\n'
        "        warm_restart_allowed: True\n"
        "---\n"
    )
    _parity_tree(tmp_path, monkeypatch, okf=okf)
    assert gate.check_okf_and_client_config_parity() == []


@pytest.mark.parametrize(
    ("old", "new", "detail"),
    [
        ("min: 16", "min: sixteen", "min = 'sixteen' is invalid (not a YAML int literal)"),
        ("min: 16", 'min: "16"', "min = '\"16\"' is invalid (not a YAML int literal)"),
        ("min: 1.0e-6", "min: 1e-6", "min = '1e-6' is invalid (not a YAML float literal"),
        ("max: 0.1", "max: .inf", "max = '.inf' is invalid"),
        ("max: 512", "max: 8", "min 16 > max 8"),
        ("warm_restart_allowed: true", "warm_restart_allowed: maybe", "is not true/false"),
    ],
)
def test_parity_reports_unparseable_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: str, new: str, detail: str
) -> None:
    _parity_tree(tmp_path, monkeypatch, okf=OKF_DOC.replace(old, new, 1))
    errors = gate.check_okf_and_client_config_parity()
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"[PARITY] {OKF_REL}: parameter ")
    assert detail in errors[0]


@pytest.mark.parametrize(
    ("key", "line"),
    [("type", "    type: int\n"), ("min", "    min: 16\n"), ("max", "    max: 512\n")],
)
def test_parity_reports_missing_required_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, line: str
) -> None:
    _parity_tree(tmp_path, monkeypatch, okf=OKF_DOC.replace(line, "", 1))  # drops it from nx
    errors = gate.check_okf_and_client_config_parity()
    assert errors == [f"[PARITY] {OKF_REL}: parameter 'nx' is missing required key(s) ['{key}']"]


def test_parity_reports_missing_bound_in_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = _config_params()
    del params["nx"]["max"]
    _parity_tree(tmp_path, monkeypatch, gcp_params=params)
    assert gate.check_okf_and_client_config_parity() == [
        f"[PARITY] {GCP_REL}: parameter 'nx' is missing required key(s) ['max']"
    ]


def test_parity_reports_path_and_warm_restart_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = _config_params()
    params["dt_max"]["gx_path"] = "Time.dt_old"
    del params["dt_max"]["warm_restart_allowed"]  # missing means false
    _parity_tree(tmp_path, monkeypatch, gcp_params=params)
    assert gate.check_okf_and_client_config_parity() == [
        f"[PARITY] dt_max.gx_path mismatch in {GCP_REL}: 'Time.dt_old' != 'Time.dt'",
        f"[PARITY] dt_max.warm_restart_allowed mismatch in {GCP_REL}: False != True",
    ]


def test_parity_reports_key_and_bound_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = _config_params()
    params["ny"] = dict(params["nx"])
    _parity_tree(tmp_path, monkeypatch, gcp_params=params)
    assert gate.check_okf_and_client_config_parity() == [
        f"[PARITY] Parameter keys mismatch between OKF and {GCP_REL}: "
        "only in OKF [], only in config ['ny']"
    ]
    del params["ny"]
    params["dt_max"]["max"] = 0.2
    _parity_tree(tmp_path, monkeypatch, gcp_params=params)
    assert gate.check_okf_and_client_config_parity() == [
        f"[PARITY] dt_max.max mismatch in {GCP_REL}: 0.2 != 0.1"
    ]


@pytest.mark.parametrize(
    ("okf", "detail"),
    [
        ("# Allowed parameters\n", "missing YAML frontmatter"),
        ("---\nparameters:\n  nx:\n    type: int\n", "unterminated YAML frontmatter"),
        ("---\nid: x\n---\n", "frontmatter defines no 'parameters'"),
        ("---\nparameters:\n  nx: {type: int}\n---\n", "must be a block mapping"),
    ],
)
def test_parity_reports_malformed_frontmatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, okf: str, detail: str
) -> None:
    _parity_tree(tmp_path, monkeypatch, okf=okf)
    errors = gate.check_okf_and_client_config_parity()
    assert len(errors) == 1 and errors[0].startswith(f"[PARITY] {OKF_REL}: "), errors
    assert detail in errors[0]


def test_real_okf_allowlist_parses() -> None:
    real = SCRIPT.parents[1] / gate.OKF_ALLOWED_PARAMETERS
    params = gate._parse_okf_frontmatter_params(real)
    assert params, "no parameters parsed"
    for name, spec in params.items():
        gate._normalized_spec(name, spec, yaml=True)  # raises ParityError if malformed


# --- .venv-tacc distribution probe -----------------------------------------------------------


def _fake_probe(
    monkeypatch: pytest.MonkeyPatch, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> dict[str, object]:
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    return seen


def test_venv_probe_reports_missing_interpreter(tmp_path: Path) -> None:
    errors = gate.check_tacc_venv_distributions(str(tmp_path / "missing" / "python"))
    assert len(errors) == 1 and errors[0].startswith("[VENV-TACC] Cannot run "), errors


def test_venv_probe_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert gate.check_tacc_venv_distributions("venv-py") == [
        "[VENV-TACC] Probe of venv-py timed out after 60s"
    ]
    assert seen["timeout"] == 60


def test_venv_probe_reports_failed_or_garbled_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_probe(
        monkeypatch, returncode=1, stderr="Traceback (most recent call last):\nOSError: x\n"
    )
    assert gate.check_tacc_venv_distributions("venv-py") == [
        "[VENV-TACC] Probe of venv-py failed with exit 1: OSError: x"
    ]
    _fake_probe(monkeypatch, stdout="not json")
    assert gate.check_tacc_venv_distributions("venv-py") == [
        "[VENV-TACC] Unparseable distribution list from venv-py"
    ]


def test_venv_flags_unexpected_and_missing_distributions(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_probe(
        monkeypatch,
        stdout=json.dumps(["ut-hpc-core", "requests", "google-cloud-storage", "openai"]),
    )
    assert gate.check_tacc_venv_distributions("venv-py") == [
        "[VENV-TACC] Unexpected packages installed in venv-py: ['google-cloud-storage', 'openai']",
        "[VENV-TACC] venv-py lacks the TACC runtime distributions ['ut-kr2-tacc-exec']",
    ]
    assert seen["cmd"][:2] == ["venv-py", "-I"] and seen["timeout"] == 60


def test_venv_accepts_tacc_runtime_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ["ut_hpc_core", "UT-KR2-TACC-EXEC", "requests", "urllib3", "f90nml", "pip"]
    _fake_probe(monkeypatch, stdout=json.dumps(names))
    assert gate.check_tacc_venv_distributions("venv-py") == []


def test_venv_probe_runs_real_interpreter() -> None:
    # The test interpreter has pytest installed, which is not on the TACC allowlist.
    errors = gate.check_tacc_venv_distributions(sys.executable)
    assert any("Unexpected packages" in e and "'pytest'" in e for e in errors), errors


# --- CLI -------------------------------------------------------------------------------------


def test_main_returns_one_and_prints_errors(repo: Path, capsys: pytest.CaptureFixture) -> None:
    _write(repo / HPC_CORE / "client.py", "import vertexai\n")
    assert gate.main([]) == 1
    err = capsys.readouterr().err
    assert "[AST] Forbidden import 'vertexai' in TACC-safe file" in err
    assert "FAILED:" in err


def test_main_on_real_repo_returns_zero(capsys: pytest.CaptureFixture) -> None:
    rc = gate.main([])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.out.startswith("OK:")
