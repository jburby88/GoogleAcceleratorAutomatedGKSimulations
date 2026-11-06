#!/usr/bin/env python3
"""CI gate for the TACC zero-AI/cloud-SDK boundary and public-repo invariants.

Checks (each returns human-readable errors; any error fails the gate):

* AST scan of the TACC runtime (``platform/hpc-core/hpc_core`` and
  ``workstreams/kr2-gk-database/hpc``): only stdlib, first-party and allowlisted third-party
  imports, and no dynamic import/exec escape hatches. ``scripts/tacc_smoke_test.py`` runs on a
  login node before any venv exists, so it must be stdlib-only.
* ``pyproject.toml`` dependency allowlists of the two TACC packages and, with
  ``--venv-python``, the distributions actually installed in ``.venv-tacc``.
* OKF ``allowed-parameters.md`` vs. the TACC/GCP example configs (names, types, paths, bounds,
  warm-restart flags).
* ``TODO(david)``/``TODO(tan)`` markers tracked in ``OPEN_QUESTIONS.md``.
* Public-repo hygiene: machine-specific paths, private keys and common credential formats.

Stdlib-only on purpose: CI runs it under a bare ``actions/setup-python`` interpreter and it must
also run under ``.venv-tacc``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STDLIB_MODULES = frozenset(sys.stdlib_module_names)
# Top-level modules the TACC runtime may import: stdlib, first-party, and the .venv-tacc deps.
ALLOWED_ROOT_MODULES = STDLIB_MODULES | {"hpc_core", "tacc_exec", "requests", "f90nml", "tomlkit"}
# scripts/tacc_smoke_test.py is documented to run "before creating any virtual environment".
SMOKE_TEST_ALLOWED_ROOT_MODULES = STDLIB_MODULES

# Repo-relative paths, resolved against REPO_ROOT at call time (tests monkeypatch REPO_ROOT).
TACC_RUNTIME_DIRS = (
    Path("platform/hpc-core/hpc_core"),
    Path("workstreams/kr2-gk-database/hpc"),
)
TACC_SMOKE_TEST = Path("scripts/tacc_smoke_test.py")
OKF_ALLOWED_PARAMETERS = Path(
    "workstreams/kr2-gk-database/skill/sim-babysitter/okf/concepts/allowed-parameters.md"
)
PARITY_CONFIGS = (
    Path("workstreams/kr2-gk-database/hpc/config.example.json"),
    Path("workstreams/kr2-gk-database/gcp/config.batch.example.json"),
)

ALLOWED_HPC_CORE_DEPS = {"requests"}
ALLOWED_KR2_HPC_DEPS = {"ut-hpc-core", "f90nml", "tomlkit", "requests"}

ALLOWED_TACC_VENV_PACKAGES = {
    "ut-hpc-core",
    "ut-kr2-tacc-exec",
    "f90nml",
    "tomlkit",
    "requests",
    "urllib3",
    "idna",
    "certifi",
    "charset-normalizer",
    "pip",
    "setuptools",
    "wheel",
}
# Without the runtime itself installed, the venv allowlist check would pass vacuously.
REQUIRED_TACC_VENV_PACKAGES = {"ut-hpc-core", "ut-kr2-tacc-exec"}
TACC_VENV_PROBE_TIMEOUT_S = 60

# Generated, vendored or machine-local directories (all gitignored): never scanned.
NON_SOURCE_DIR_NAMES = frozenset(
    {
        ".git",
        ".agent",
        ".agents",
        "_agent",
        "_agents",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".terraform",
        "node_modules",
        "build",
        "dist",
        "venv",
    }
)

# Builtins that compile/run code or import a module named by a runtime string.
_DYNAMIC_BUILTINS = frozenset({"exec", "eval", "compile", "__import__"})
# importlib entry points that load a module chosen at runtime (by name or file path).
_DYNAMIC_LOADERS = frozenset(
    {
        "import_module",
        "spec_from_file_location",
        "spec_from_loader",
        "module_from_spec",
        "exec_module",
        "load_module",
    }
)
_RUNPY_FUNCS = frozenset({"run_module", "run_path"})
_BUILTINS_NAMES = frozenset({"builtins", "__builtins__"})
_ALL_DYNAMIC_NAMES = _DYNAMIC_BUILTINS | _DYNAMIC_LOADERS | _RUNPY_FUNCS
# `from <module> import <name>` would let the functions above be called under an alias.
_DYNAMIC_FROM_IMPORTS = {
    "builtins": _DYNAMIC_BUILTINS,
    "importlib": _DYNAMIC_LOADERS,
    "importlib.util": _DYNAMIC_LOADERS,
    "runpy": _RUNPY_FUNCS,
}

# Reported by label only: the matched text never reaches CI logs.
HYGIENE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/[U]sers/[A-Za-z0-9_.-]+"), "local workstation user path"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/"), "local Linux home path (/home/<name>/)"),
    (
        re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"),
        "PEM private key header",
    ),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "Google API key"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), "GitHub fine-grained token"),
    (
        re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9]{8,}/[A-Za-z0-9]{8,}/[A-Za-z0-9]{16,}"),
        "Slack webhook URL",
    ),
)
# A GCP service-account key file carries both markers.
SERVICE_ACCOUNT_KEY_MARKERS = (
    re.compile(r'"type"\s*:\s*"service_account"'),
    re.compile(r'"private_key"\s*:'),
)
HYGIENE_SKIP_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".duckdb", ".wal", ".png", ".jpg", ".pdf", ".zip"}
)
HYGIENE_SKIP_REL_DIRS = frozenset({"docs/review"})  # gitignored internal review notes


def _normalize_pkg_name(raw: str) -> str:
    base = re.split(r"[<>=!~;\[\s]", raw.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", base).lower()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _is_non_source_dir(name: str) -> bool:
    return name in NON_SOURCE_DIR_NAMES or name.startswith(".venv") or name.endswith(".egg-info")


def _skip_non_source(rel_dir: Path) -> bool:
    return _is_non_source_dir(rel_dir.name)


def _skip_tacc_scan_dir(rel_dir: Path) -> bool:
    return rel_dir.name == "tests" or _is_non_source_dir(rel_dir.name)


def _walk_files(
    root: Path, skip_dir: Callable[[Path], bool], errors: list[str], tag: str
) -> Iterator[Path]:
    """Yield regular files under ``root`` in sorted order, pruning dirs where ``skip_dir`` is true.

    ``skip_dir`` receives the directory path relative to ``root``, so ancestors of ``root``
    (e.g. a checkout under a directory named ``build``) never hide the whole tree.
    Symlinked directories are not followed; unlistable directories are reported.
    """

    def on_error(exc: OSError) -> None:
        where = _rel(Path(exc.filename)) if exc.filename else _rel(root)
        errors.append(f"[{tag}] Cannot list directory {where}: {exc.strerror or exc}")

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        current = Path(dirpath)
        rel_dir = current.relative_to(root)
        dirnames[:] = sorted(d for d in dirnames if not skip_dir(rel_dir / d))
        for name in sorted(filenames):
            path = current / name
            if path.is_file():
                yield path


def _dynamic_call_label(node: ast.Call) -> str | None:
    """Return a display name when ``node`` calls a dynamic import/exec entry point."""
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in _DYNAMIC_BUILTINS or func.id in _DYNAMIC_LOADERS:
            return func.id
        if func.id == "getattr" and node.args:
            target, attr = node.args[0], node.args[1] if len(node.args) > 1 else None
            if isinstance(target, ast.Name) and target.id in _BUILTINS_NAMES:
                return f"getattr({target.id}, ...)"
            if isinstance(attr, ast.Constant) and attr.value in _ALL_DYNAMIC_NAMES:
                return f"getattr(..., {attr.value!r})"
    elif isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else None
        if (
            func.attr in _DYNAMIC_LOADERS
            or func.attr == "__import__"
            or (owner in _BUILTINS_NAMES and func.attr in _DYNAMIC_BUILTINS)
            or (owner == "runpy" and func.attr in _RUNPY_FUNCS)
        ):
            return ast.unparse(func)
    return None


def _scan_python_file(path: Path, allowed_roots: frozenset[str], kind: str) -> list[str]:
    """AST-check one file: allowlisted absolute imports only, no dynamic import/exec."""
    rel = _rel(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except SyntaxError as exc:
        return [f"[AST] Cannot parse {kind} file {rel}:{exc.lineno}: SyntaxError: {exc.msg}"]
    except (OSError, ValueError) as exc:  # unreadable, undecodable, or NUL bytes
        return [f"[AST] Cannot read {kind} file {rel}: {exc}"]

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolves inside the scanned package itself
                continue
            mods = [node.module or ""]
            loaders = _DYNAMIC_FROM_IMPORTS.get(node.module or "", frozenset())
            for alias in node.names:
                if loaders and (alias.name == "*" or alias.name in loaders):
                    found.append(
                        (
                            node.lineno,
                            f"[AST] Forbidden dynamic loader import "
                            f"'{node.module}.{alias.name}' in {kind} file {rel}:{node.lineno}",
                        )
                    )
        elif isinstance(node, ast.Call):
            label = _dynamic_call_label(node)
            if label:
                found.append(
                    (
                        node.lineno,
                        f"[AST] Forbidden dynamic call {label!r} in {kind} file {rel}:{node.lineno}",
                    )
                )
        for mod in mods:
            if mod.split(".")[0] not in allowed_roots:
                found.append(
                    (
                        node.lineno,
                        f"[AST] Forbidden import {mod!r} in {kind} file {rel}:{node.lineno}",
                    )
                )
    return [msg for _, msg in sorted(found)]


def check_ast_imports() -> list[str]:
    """AST-scan the TACC runtime and the stdlib-only login-node smoke test.

    Every ``.py`` under ``TACC_RUNTIME_DIRS`` is scanned except ``tests/`` and generated or
    machine-local directories (``build/``, ``*.egg-info``, ``__pycache__``, venvs, ...).
    """
    errors: list[str] = []
    for rel_dir in TACC_RUNTIME_DIRS:
        root = REPO_ROOT / rel_dir
        if not root.is_dir():
            errors.append(f"[AST] TACC runtime directory missing: {rel_dir.as_posix()}")
            continue
        for path in _walk_files(root, _skip_tacc_scan_dir, errors, "AST"):
            if path.suffix == ".py":
                errors.extend(_scan_python_file(path, ALLOWED_ROOT_MODULES, "TACC-safe"))
    smoke = REPO_ROOT / TACC_SMOKE_TEST
    if smoke.is_file():
        errors.extend(_scan_python_file(smoke, SMOKE_TEST_ALLOWED_ROOT_MODULES, "stdlib-only"))
    else:
        errors.append(f"[AST] TACC smoke test missing: {TACC_SMOKE_TEST.as_posix()}")
    return errors


def check_pyproject_dependencies() -> list[str]:
    """Verify `hpc-core` and `kr2-gk-database/hpc` `pyproject.toml` files declare only allowlisted packages."""
    errors: list[str] = []
    checks = [
        (Path("platform/hpc-core/pyproject.toml"), ALLOWED_HPC_CORE_DEPS),
        (Path("workstreams/kr2-gk-database/hpc/pyproject.toml"), ALLOWED_KR2_HPC_DEPS),
    ]
    for rel_path, allowed in checks:
        try:
            data = tomllib.loads((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # missing, undecodable, or invalid TOML
            errors.append(f"[PYPROJECT] Cannot parse {rel_path.as_posix()}: {exc}")
            continue
        try:
            project = data.get("project", {})
            groups = [
                project.get("dependencies", []),
                *project.get("optional-dependencies", {}).values(),
                data.get("build-system", {}).get("requires", []),
            ]
        except AttributeError:  # a table key holding a non-table value
            groups = [None]
        if not all(isinstance(g, list) and all(isinstance(d, str) for d in g) for g in groups):
            errors.append(f"[PYPROJECT] {rel_path.as_posix()} has a malformed dependency list")
            continue
        deps = {_normalize_pkg_name(d) for group in groups for d in group}

        allowed_with_dev = allowed | {"setuptools", "wheel", "pytest", "ruff"}
        disallowed = deps - allowed_with_dev

        if disallowed:
            errors.append(
                f"[PYPROJECT] {rel_path.as_posix()} declares disallowed runtime deps: {sorted(disallowed)}"
            )
    return errors


def check_tacc_venv_distributions(venv_python: str) -> list[str]:
    """Inspect installed distributions inside `.venv-tacc` against the strict TACC allowlist."""
    probe = (
        "import importlib.metadata as m, json; "
        "print(json.dumps(sorted({d.metadata['Name'] for d in m.distributions() if d.metadata.get('Name')})))"
    )
    try:
        # -I: ignore PYTHONPATH/user site so only the venv's own distributions are listed.
        proc = subprocess.run(
            [venv_python, "-I", "-c", probe],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=TACC_VENV_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [f"[VENV-TACC] Probe of {venv_python} timed out after {TACC_VENV_PROBE_TIMEOUT_S}s"]
    except OSError as exc:
        return [f"[VENV-TACC] Cannot run {venv_python}: {exc}"]
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1][:300]
        return [f"[VENV-TACC] Probe of {venv_python} failed with exit {proc.returncode}: {last}"]
    try:
        names = json.loads(proc.stdout)
    except ValueError:
        names = None
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return [f"[VENV-TACC] Unparseable distribution list from {venv_python}"]

    errors: list[str] = []
    installed = {_normalize_pkg_name(name) for name in names}
    unexpected = installed - ALLOWED_TACC_VENV_PACKAGES
    if unexpected:
        errors.append(
            f"[VENV-TACC] Unexpected packages installed in {venv_python}: {sorted(unexpected)}"
        )
    missing = REQUIRED_TACC_VENV_PACKAGES - installed
    if missing:
        errors.append(
            f"[VENV-TACC] {venv_python} lacks the TACC runtime distributions {sorted(missing)}"
        )
    return errors


class ParityError(ValueError):
    """Malformed OKF frontmatter or example config, reported as a [PARITY] error."""


_YAML_KEY = re.compile(r"(?P<key>[^:]+?)\s*:(?:\s+(?P<value>.*))?")
# YAML 1.1 scalars as resolved by PyYAML in okf_loader: e.g. `1e-6` (no dot, unsigned
# exponent) and `016` (octal) are not decimal numbers there, so they are rejected here too.
_YAML_INT = re.compile(r"[-+]?(?:0|[1-9][0-9_]*)")
_YAML_FLOAT = re.compile(
    r"[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9_]+(?:[eE][-+][0-9]+)?"
)
_YAML_TRUE = frozenset({"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"})
_YAML_FALSE = frozenset({"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"})
_MISSING = object()


def _yaml_scalar(raw: str | None, where: str) -> str:
    """Return a raw YAML scalar without its trailing comment (quotes are kept)."""
    value = (raw or "").strip()
    if value[:1] in ("'", '"'):
        close = value.find(value[0], 1)
        if close == -1 or value[close + 1 :].strip()[:1] not in ("", "#"):
            raise ParityError(f"{where}: cannot parse value {value!r}")
        return value[: close + 1]
    if value.startswith("#"):
        return ""
    comment = re.search(r"\s#", value)
    return value[: comment.start()].rstrip() if comment else value


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_okf_frontmatter_params(md_path: Path) -> dict[str, dict[str, str]]:
    """Parse the ``parameters:`` mapping of an OKF concept's YAML frontmatter (stdlib only).

    Supports the YAML subset the file uses: space-indented block mappings, plain or quoted
    scalars, full-line and trailing comments. Returns raw scalars (quotes kept, so a quoted
    number stays a string as in YAML). Anything else raises ParityError instead of silently
    dropping a parameter.
    """
    text = md_path.read_text(encoding="utf-8").lstrip("\ufeff")
    if not text.startswith("---\n"):
        hint = " (CRLF line endings are not supported)" if text.startswith("---\r\n") else ""
        raise ParityError(f"missing YAML frontmatter: the file must start with a '---' line{hint}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ParityError("unterminated YAML frontmatter: no closing '---' line")

    params: dict[str, dict[str, str]] = {}
    seen_parameters = in_params = False
    param_indent: int | None = None
    field_indent: int | None = None
    current: dict[str, str] | None = None
    for lineno, line in enumerate(text[4:end].split("\n"), start=2):
        body = line.lstrip(" ")
        if not body.strip() or body.startswith("#"):
            continue
        where = f"line {lineno}"
        if body.startswith("\t"):
            raise ParityError(f"{where}: tab indentation is not valid YAML")
        indent = len(line) - len(body)
        is_item = body.rstrip() == "-" or body.startswith("- ")
        match = None if is_item else _YAML_KEY.fullmatch(body.rstrip())
        if indent == 0:
            if match is None:
                if in_params:
                    raise ParityError(f"{where}: cannot parse {body.strip()!r}")
                continue  # continuation of another top-level value (list item, flow, ...)
            in_params = _unquote(match["key"]) == "parameters"
            if in_params:
                if seen_parameters:
                    raise ParityError(f"{where}: duplicate 'parameters' key")
                if _yaml_scalar(match["value"], where):
                    raise ParityError(f"{where}: 'parameters' must be a block mapping")
                seen_parameters = True
            continue
        if not in_params:
            continue
        if param_indent is None:
            param_indent = indent
        if indent == param_indent:
            if match is None:
                raise ParityError(f"{where}: cannot parse {body.strip()!r}")
            name = _unquote(match["key"])
            if _yaml_scalar(match["value"], where):
                raise ParityError(f"{where}: parameter {name!r} must be a block mapping")
            if name in params:
                raise ParityError(f"{where}: duplicate parameter {name!r}")
            current = params[name] = {}
            field_indent = None
            continue
        if indent < param_indent or current is None:
            raise ParityError(f"{where}: inconsistent indentation")
        if field_indent is None:
            field_indent = indent
        if indent > field_indent or is_item:
            continue  # nested value of a field the parity check does not compare
        if indent < field_indent:
            raise ParityError(f"{where}: inconsistent indentation")
        if match is None:
            raise ParityError(f"{where}: cannot parse {body.strip()!r}")
        key = _unquote(match["key"])
        if key in current:
            raise ParityError(f"{where}: duplicate key {key!r}")
        current[key] = _yaml_scalar(match["value"], where)
    return params


def _yaml_number(raw: object, vtype: str) -> int | float:
    text = str(raw)
    if _YAML_INT.fullmatch(text):
        value = int(text.replace("_", ""))
        return value if vtype == "int" else float(value)
    if vtype == "float" and _YAML_FLOAT.fullmatch(text):
        return float(text.replace("_", ""))
    hint = ", write e.g. 1.0e-6" if vtype == "float" else ""
    raise ValueError(f"not a YAML {vtype} literal{hint}")


def _json_number(raw: object, vtype: str) -> int | float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("not a JSON number")
    if vtype == "int" and not isinstance(raw, int):
        raise ValueError("not a JSON integer")
    return raw if vtype == "int" else float(raw)


def _normalized_spec(name: str, spec: Mapping[str, object], *, yaml: bool) -> dict[str, object]:
    """Validate one parameter spec (raw OKF scalars if ``yaml`` else JSON values).

    Returns the compared fields: type, min, max, gene_path, gx_path, warm_restart_allowed.
    """
    missing = [k for k in ("type", "min", "max") if spec.get(k) in (None, "")]
    if missing:
        raise ParityError(f"parameter {name!r} is missing required key(s) {missing}")
    vtype = _unquote(str(spec["type"])) if yaml else spec["type"]
    if vtype not in ("int", "float"):
        raise ParityError(f"parameter {name!r}: type must be 'int' or 'float', got {vtype!r}")
    to_number = _yaml_number if yaml else _json_number
    bounds: list[int | float] = []
    for bound in ("min", "max"):
        raw = spec[bound]
        try:
            value = to_number(raw, vtype)
            if not math.isfinite(value):
                raise ValueError("not finite")
        except ValueError as exc:
            raise ParityError(f"parameter {name!r}: {bound} = {raw!r} is invalid ({exc})") from None
        bounds.append(value)
    lo, hi = bounds
    if lo > hi:
        raise ParityError(f"parameter {name!r}: min {lo} > max {hi}")
    out: dict[str, object] = {"type": vtype, "min": lo, "max": hi}
    for field in ("gene_path", "gx_path"):
        value = spec.get(field)
        out[field] = None if value is None else (_unquote(str(value)) if yaml else str(value))
    warm = spec.get("warm_restart_allowed", _MISSING)
    if warm is _MISSING:
        out["warm_restart_allowed"] = False  # missing means false on both sides
    elif yaml and warm in _YAML_TRUE | _YAML_FALSE:
        out["warm_restart_allowed"] = warm in _YAML_TRUE
    elif not yaml and isinstance(warm, bool):
        out["warm_restart_allowed"] = warm
    else:
        raise ParityError(f"parameter {name!r}: warm_restart_allowed = {warm!r} is not true/false")
    return out


def check_okf_and_client_config_parity() -> list[str]:
    """Verify OKF allowed-parameters.md and the TACC/GCP example configs agree on every parameter.

    Compares names, type, gene_path, gx_path, min/max and warm_restart_allowed (missing = false).
    Malformed input on either side is reported as a [PARITY] error, never a traceback.
    """
    okf_rel = OKF_ALLOWED_PARAMETERS.as_posix()
    try:
        okf_raw = _parse_okf_frontmatter_params(REPO_ROOT / OKF_ALLOWED_PARAMETERS)
    except (OSError, ValueError) as exc:
        return [f"[PARITY] {okf_rel}: {exc}"]
    if not okf_raw:
        return [f"[PARITY] {okf_rel}: frontmatter defines no 'parameters'"]

    errors: list[str] = []
    okf_specs: dict[str, dict[str, object]] = {}
    for name, spec in okf_raw.items():
        try:
            okf_specs[name] = _normalized_spec(name, spec, yaml=True)
        except ParityError as exc:
            errors.append(f"[PARITY] {okf_rel}: {exc}")

    for cfg in PARITY_CONFIGS:
        cfg_rel = cfg.as_posix()
        try:
            data = json.loads((REPO_ROOT / cfg).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"[PARITY] Cannot read {cfg_rel}: {exc}")
            continue
        cfg_params = data.get("allowed_parameters") if isinstance(data, dict) else None
        if not isinstance(cfg_params, dict):
            errors.append(f"[PARITY] {cfg_rel} has no 'allowed_parameters' object")
            continue
        only_okf = sorted(set(okf_raw) - set(cfg_params))
        only_cfg = sorted(set(cfg_params) - set(okf_raw))
        if only_okf or only_cfg:
            errors.append(
                f"[PARITY] Parameter keys mismatch between OKF and {cfg_rel}: "
                f"only in OKF {only_okf}, only in config {only_cfg}"
            )
            continue
        for name in okf_raw:
            try:
                if not isinstance(cfg_params[name], dict):
                    raise ParityError(f"parameter {name!r} must be a JSON object")
                cspec = _normalized_spec(name, cfg_params[name], yaml=False)
            except ParityError as exc:
                errors.append(f"[PARITY] {cfg_rel}: {exc}")
                continue
            ospec = okf_specs.get(name)
            if ospec is None:
                continue  # OKF side already reported
            for field in ("type", "gene_path", "gx_path", "warm_restart_allowed"):
                if cspec[field] != ospec[field]:
                    errors.append(
                        f"[PARITY] {name}.{field} mismatch in {cfg_rel}: "
                        f"{cspec[field]!r} != {ospec[field]!r}"
                    )
            for bound in ("min", "max"):
                if not math.isclose(cspec[bound], ospec[bound], rel_tol=1e-12, abs_tol=0.0):
                    errors.append(
                        f"[PARITY] {name}.{bound} mismatch in {cfg_rel}: "
                        f"{cspec[bound]} != {ospec[bound]}"
                    )
    return errors


def check_todo_tracking_parity() -> list[str]:
    """Ensure every inline `TODO(david)` or `TODO(tan)` marker is tracked in `OPEN_QUESTIONS.md`."""
    errors: list[str] = []
    open_q_path = REPO_ROOT / "workstreams" / "kr2-gk-database" / "docs" / "OPEN_QUESTIONS.md"
    if not open_q_path.is_file():
        return ["[TODO] Missing workstreams/kr2-gk-database/docs/OPEN_QUESTIONS.md"]

    open_q_text = open_q_path.read_text(encoding="utf-8")
    todo_re = re.compile(r"TODO\((david|tan)\)")

    for folder_name in ("platform", "workstreams"):
        for path in _walk_files(REPO_ROOT / folder_name, _skip_non_source, errors, "TODO"):
            if path == open_q_path:
                continue
            if path.suffix not in (
                ".py",
                ".sql",
                ".tf",
                ".md",
                ".yaml",
                ".yml",
                ".json",
            ):
                continue
            rel = str(path.relative_to(REPO_ROOT))
            text = path.read_text(encoding="utf-8", errors="ignore")
            if todo_re.search(text) and path.name not in open_q_text and rel not in open_q_text:
                errors.append(
                    f"[TODO] File {rel} contains TODO(david)/TODO(tan) but is not listed in OPEN_QUESTIONS.md"
                )
    return errors


def check_public_repo_hygiene() -> list[str]:
    """Flag machine-specific paths, private keys and credential formats in repo files.

    Only the file and a label are reported, never the matched text, so CI logs cannot leak
    a secret. Generated, vendored and machine-local (gitignored) directories are skipped.
    """
    errors: list[str] = []

    def skip(rel: Path) -> bool:
        return _is_non_source_dir(rel.name) or rel.as_posix() in HYGIENE_SKIP_REL_DIRS

    for path in _walk_files(REPO_ROOT, skip, errors, "PUBLIC-HYGIENE"):
        if path.name == ".env" or path.suffix in HYGIENE_SKIP_SUFFIXES:
            continue
        rel = _rel(path)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            errors.append(f"[PUBLIC-HYGIENE] Cannot read {rel}: {exc.strerror or exc}")
            continue
        for regex, label in HYGIENE_PATTERNS:
            if regex.search(text):
                errors.append(f"[PUBLIC-HYGIENE] {rel} contains forbidden {label}")
        if all(marker.search(text) for marker in SERVICE_ACCOUNT_KEY_MARKERS):
            errors.append(f"[PUBLIC-HYGIENE] {rel} contains forbidden GCP service-account key")
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for repository boundary, dependency, and hygiene verification."""
    parser = argparse.ArgumentParser(
        description="Verify TACC Zero-AI compliance and repo invariants."
    )
    parser.add_argument(
        "--venv-python", default=None, help="Optional path to .venv-tacc/bin/python"
    )
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    all_errors.extend(check_ast_imports())
    all_errors.extend(check_pyproject_dependencies())
    all_errors.extend(check_okf_and_client_config_parity())
    all_errors.extend(check_todo_tracking_parity())
    all_errors.extend(check_public_repo_hygiene())
    if args.venv_python:
        all_errors.extend(check_tacc_venv_distributions(args.venv_python))

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        print(f"FAILED: {len(all_errors)} error(s)", file=sys.stderr)
        return 1

    venv_note = ", .venv-tacc distributions" if args.venv_python else ""
    print(
        "OK: Zero-AI TACC boundaries, pyproject allowlists, OKF parameter parity, "
        f"OPEN_QUESTIONS tracking, public-repo hygiene{venv_note} verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
