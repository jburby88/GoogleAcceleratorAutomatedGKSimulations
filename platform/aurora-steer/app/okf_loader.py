"""Dynamic Open Knowledge Framework (OKF 2.0) skill & runbook loader.

Design & Collaborator Workflow:
- Physics skills under `simulations/skill/<skill_name>/okf/`
  own their OKF 2.0 knowledge graph (`manifest.yaml` + `concepts/*.md`).
- Physicists (e.g., Prof. David Hatch's group for `sim-babysitter`) can refine
  troubleshooting runbooks and numerical parameter bounds via GitHub PRs by
  editing Markdown/YAML files without touching Python application code.
- At `/triage` time, `LoadedSkill.select_concepts_for_failure()` loads only the
  concepts triggered by the incoming `failure_type` (`ALL` + matching runbook),
  while `_parse_parameter_rules()` extracts the authoritative `[min, max]`
  parameter guardrails from `allowed-parameters.md`.

Because these files drive server-side guardrails, parsing is strict and fails
closed: any malformed bound, unknown key, duplicate YAML key or unknown action
raises `OKFLoadError` (the service then refuses to start / triage).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.schemas import ACTION_TYPES


class OKFLoadError(RuntimeError):
    """Raised when an OKF skill or concept file is missing, malformed, or unsafe."""


class SkillNotFoundError(OKFLoadError):
    """Raised when no OKF skill directory exists for a (well-formed) skill name."""


@dataclass(frozen=True)
class ParameterRule:
    """Single allowlisted numerical solver parameter specification."""

    name: str
    value_type: str  # "int" or "float"
    min_value: int | float
    max_value: int | float
    gene_path: str
    gx_path: str
    # True only for parameters that may change on a checkpoint (warm) restart.
    # Defaults to False: any other change requires `cold_restart` (PHYS-03).
    warm_restart_allowed: bool = False


@dataclass
class OKFConcept:
    """Parsed OKF concept markdown document."""

    concept_id: str
    path: str
    triggers: list[str]
    frontmatter: dict[str, Any]
    body: str
    # Actions a runbook permits for the failure types it is triggered by.
    allowed_actions: frozenset[str] = frozenset()


@dataclass
class LoadedSkill:
    """Complete loaded OKF skill bundle."""

    skill_name: str
    okf_dir: Path
    manifest: dict[str, Any]
    concepts: dict[str, OKFConcept]
    parameter_rules: dict[str, ParameterRule] = field(default_factory=dict)

    def select_concepts_for_failure(self, failure_type: str) -> list[OKFConcept]:
        """Return always-included concepts (`allowed-parameters`, `escalation-policy`) plus matching runbooks."""
        selected: list[OKFConcept] = []
        seen: set[str] = set()
        ft_upper = failure_type.upper()

        for always_id in ("allowed-parameters", "escalation-policy"):
            if always_id in self.concepts and always_id not in seen:
                selected.append(self.concepts[always_id])
                seen.add(always_id)

        for cid, concept in self.concepts.items():
            if cid in seen:
                continue
            triggers_upper = {t.upper() for t in concept.triggers}
            if "ALL" in triggers_upper or ft_upper in triggers_upper:
                selected.append(concept)
                seen.add(cid)

        return selected

    def allowed_actions_for(self, failure_type: str) -> frozenset[str]:
        """Actions the runbooks permit for `failure_type` (server-side policy).

        Union of `allowed_actions` over the runbooks explicitly triggered by this
        failure type (`ALL` concepts grant nothing). `escalate_to_human` is always
        allowed; a failure type without a runbook allows only escalation.
        """
        ft_upper = failure_type.upper()
        allowed: set[str] = {"escalate_to_human"}
        for concept in self.concepts.values():
            if ft_upper in {t.upper() for t in concept.triggers}:
                allowed |= concept.allowed_actions
        return frozenset(allowed)

    def format_prompt_context(self, failure_type: str) -> tuple[str, list[str]]:
        """Assemble selected OKF concepts into structured LLM prompt context."""
        chosen = self.select_concepts_for_failure(failure_type)
        blocks: list[str] = []
        concept_ids: list[str] = []
        for concept in chosen:
            concept_ids.append(concept.concept_id)
            blocks.append(f"=== OKF CONCEPT: {concept.concept_id} ===\n{concept.body.strip()}\n")
        return "\n".join(blocks), concept_ids


_SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_PARAM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SOLVER_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_PARAM_SPEC_KEYS = frozenset(
    {"type", "min", "max", "gene_path", "gx_path", "warm_restart_allowed", "description"}
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` that rejects duplicate mapping keys instead of keeping the last one."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _load_yaml(text: str, source: Path) -> Any:
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise OKFLoadError(f"Invalid YAML in {source}: {exc}") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OKFLoadError(f"Cannot read OKF file {path}: {exc}") from exc


def _default_repo_root() -> Path:
    env_root = os.getenv("REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    # platform/aurora-steer/app/okf_loader.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def resolve_skill_okf_dir(skill_name: str, skills_root: Path | None = None) -> Path:
    """Locate `<skill_name>/okf` either under an explicit skills_root or under simulations/skill/."""
    if not isinstance(skill_name, str) or not _SAFE_NAME_RE.match(skill_name):
        raise OKFLoadError(f"Unsafe or invalid skill_name: {skill_name!r}")

    candidates: list[Path] = []
    if skills_root is not None:
        candidates.append(skills_root.resolve() / skill_name / "okf")
    else:
        env_dir = os.getenv("OKF_SKILLS_DIR", "").strip()
        if env_dir:
            candidates.append(Path(env_dir).resolve() / skill_name / "okf")
        repo_root = _default_repo_root()
        candidates.append(repo_root / "simulations" / "skill" / skill_name / "okf")
        candidates.append(repo_root / "surrogates" / "skill" / skill_name / "okf")
        workstreams_dir = repo_root / "workstreams"
        if workstreams_dir.is_dir():
            for ws in sorted(workstreams_dir.iterdir()):
                if ws.is_dir():
                    candidates.append(ws / "skill" / skill_name / "okf")

    for cand in candidates:
        if cand.is_dir() and (cand / "manifest.yaml").is_file():
            return cand.resolve()

    raise SkillNotFoundError(
        f"Could not find OKF manifest for skill {skill_name!r}; searched {[str(c) for c in candidates]}"
    )


def _split_frontmatter(markdown_text: str, source_path: Path) -> tuple[dict[str, Any], str]:
    text = markdown_text.lstrip("\ufeff").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end_idx = text.find("\n---\n", 4)
    if end_idx == -1:
        raise OKFLoadError(f"Unterminated YAML frontmatter in {source_path}")
    raw_yaml = text[4:end_idx]
    body = text[end_idx + 5 :]
    parsed = _load_yaml(raw_yaml, source_path) or {}
    if not isinstance(parsed, dict):
        raise OKFLoadError(f"Frontmatter in {source_path} must be a YAML mapping")
    return parsed, body


def _parse_bound(name: str, label: str, raw: Any, vtype: str) -> int | float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise OKFLoadError(
            f"Parameter {name!r} {label} must be a number, got {raw!r} "
            "(note: YAML reads 1e-6 as a string; write 1.0e-6)"
        )
    if not math.isfinite(float(raw)):
        raise OKFLoadError(f"Parameter {name!r} {label} must be finite, got {raw!r}")
    if vtype == "int":
        if isinstance(raw, float) and not raw.is_integer():
            raise OKFLoadError(f"Integer parameter {name!r} {label} must be integral, got {raw!r}")
        return int(raw)
    return float(raw)


def _parse_parameter_rules(concept: OKFConcept) -> dict[str, ParameterRule]:
    raw_params = concept.frontmatter.get("parameters")
    if not isinstance(raw_params, dict) or not raw_params:
        raise OKFLoadError(
            "allowed-parameters.md frontmatter must define a non-empty 'parameters' mapping"
        )
    rules: dict[str, ParameterRule] = {}
    for name, spec in raw_params.items():
        if not isinstance(name, str) or not _PARAM_NAME_RE.fullmatch(name):
            raise OKFLoadError(f"Invalid parameter name {name!r}")
        if not isinstance(spec, dict):
            raise OKFLoadError(f"Parameter spec for {name!r} must be a mapping")
        unknown = set(spec) - _PARAM_SPEC_KEYS
        if unknown:
            raise OKFLoadError(f"Parameter {name!r} has unknown keys {sorted(map(str, unknown))}")
        vtype = spec.get("type")
        if vtype not in ("int", "float"):
            raise OKFLoadError(f"Parameter {name!r} type must be 'int' or 'float', got {vtype!r}")
        if "min" not in spec or "max" not in spec:
            raise OKFLoadError(f"Parameter {name!r} must define both 'min' and 'max'")
        min_v = _parse_bound(name, "min", spec["min"], vtype)
        max_v = _parse_bound(name, "max", spec["max"], vtype)
        if min_v > max_v:
            raise OKFLoadError(f"Invalid bounds for {name!r}: min={min_v} > max={max_v}")
        paths: dict[str, str] = {}
        for key in ("gene_path", "gx_path"):
            value = spec.get(key)
            if not isinstance(value, str) or not _SOLVER_PATH_RE.fullmatch(value):
                raise OKFLoadError(
                    f"Parameter {name!r} {key} must look like 'section.key', got {value!r}"
                )
            paths[key] = value
        warm_ok = spec.get("warm_restart_allowed", False)
        if not isinstance(warm_ok, bool):
            raise OKFLoadError(f"Parameter {name!r} warm_restart_allowed must be true/false")
        rules[name] = ParameterRule(
            name=name,
            value_type=vtype,
            min_value=min_v,
            max_value=max_v,
            gene_path=paths["gene_path"],
            gx_path=paths["gx_path"],
            warm_restart_allowed=warm_ok,
        )
    for key in ("gene_path", "gx_path"):
        seen_paths: dict[str, str] = {}
        for rule in rules.values():
            path = getattr(rule, key)
            if path in seen_paths:
                raise OKFLoadError(
                    f"Parameters {seen_paths[path]!r} and {rule.name!r} map to the same {key} {path!r}"
                )
            seen_paths[path] = rule.name
    return rules


def _parse_str_list(raw: Any, label: str, source: Path) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(x, str) and x.strip() for x in raw):
        raise OKFLoadError(f"{label} in {source} must be a list of non-empty strings")
    return [x.strip() for x in raw]


def load_skill(skill_name: str = "sim-babysitter", skills_root: Path | None = None) -> LoadedSkill:
    """Load and validate an OKF skill bundle from disk."""
    okf_dir = resolve_skill_okf_dir(skill_name=skill_name, skills_root=skills_root)
    manifest_path = okf_dir / "manifest.yaml"
    manifest = _load_yaml(_read_text(manifest_path), manifest_path)
    if not isinstance(manifest, dict) or "concepts" not in manifest:
        raise OKFLoadError(f"Invalid manifest.yaml at {manifest_path}")

    concepts_list = manifest.get("concepts")
    if not isinstance(concepts_list, list) or not concepts_list:
        raise OKFLoadError(f"manifest.yaml in {okf_dir} must define a non-empty 'concepts' list")

    concepts: dict[str, OKFConcept] = {}
    for entry in concepts_list:
        if not isinstance(entry, dict) or "id" not in entry or "path" not in entry:
            raise OKFLoadError(f"Malformed concept entry in {manifest_path}: {entry!r}")
        cid = str(entry["id"]).strip()
        if not _SAFE_NAME_RE.fullmatch(cid):
            raise OKFLoadError(f"Invalid concept id {cid!r} in {manifest_path}")
        if cid in concepts:
            raise OKFLoadError(f"Duplicate concept id {cid!r} in {manifest_path}")
        rel_path = str(entry["path"]).strip()
        target_path = (okf_dir / rel_path).resolve()
        if not target_path.is_relative_to(okf_dir):
            raise OKFLoadError(f"Path traversal blocked in OKF manifest: {rel_path!r}")
        if not target_path.is_file():
            raise OKFLoadError(f"Missing OKF concept file referenced by manifest: {target_path}")

        frontmatter, body = _split_frontmatter(_read_text(target_path), target_path)
        triggers = _parse_str_list(
            entry.get("triggers", frontmatter.get("triggers", [])),
            f"triggers of {cid!r}",
            manifest_path,
        )
        actions = _parse_str_list(
            entry.get("allowed_actions"), f"allowed_actions of {cid!r}", manifest_path
        )
        unknown_actions = set(actions) - ACTION_TYPES
        if unknown_actions:
            raise OKFLoadError(
                f"Concept {cid!r} lists unknown allowed_actions {sorted(unknown_actions)}"
            )
        if "resubmit_walltime" in actions:
            raise OKFLoadError(
                f"Concept {cid!r}: resubmit_walltime is a local zero-LLM walltime/preemption "
                "continuation and cannot be granted to /triage"
            )
        if actions and "ALL" in {t.upper() for t in triggers}:
            raise OKFLoadError(f"Concept {cid!r} with triggers ['ALL'] must not grant actions")
        concepts[cid] = OKFConcept(
            concept_id=cid,
            path=rel_path,
            triggers=triggers,
            frontmatter=frontmatter,
            body=body,
            allowed_actions=frozenset(actions),
        )

    for required in ("allowed-parameters", "escalation-policy"):
        if required not in concepts:
            raise OKFLoadError(f"Skill {skill_name!r} is missing required {required!r} concept")

    param_rules = _parse_parameter_rules(concepts["allowed-parameters"])
    return LoadedSkill(
        skill_name=skill_name,
        okf_dir=okf_dir,
        manifest=manifest,
        concepts=concepts,
        parameter_rules=param_rules,
    )
