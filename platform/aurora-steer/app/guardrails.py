"""Server-side reject-only parameter and action guardrails.

Applied to every LLM `ActionDirective` AFTER the model returns and BEFORE anything
is recorded or sent to an executor. Every violation is fail-closed: the directive
is replaced by `escalate_to_human` (never clamped or repaired).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from app.okf_loader import ParameterRule
from app.schemas import ActionDirective

MAX_CONCEPTS_RECORDED = 12


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of applying reject-only server guardrails to an LLM ActionDirective."""

    directive: ActionDirective
    accepted: bool
    forced_escalation: bool
    rejection_reason: str | None = None


def _reject_to_escalation(reason: str, concepts: list[str]) -> GuardrailResult:
    truncated = f"Guardrail rejected LLM directive: {reason}"[:300]
    merged_concepts = list(dict.fromkeys(["allowed-parameters", "escalation-policy", *concepts]))
    return GuardrailResult(
        directive=ActionDirective(
            action="escalate_to_human",
            reason=truncated,
            parameter_modifications={},
            restart_from_checkpoint=False,
            confidence="high",
            okf_concepts_used=merged_concepts[:MAX_CONCEPTS_RECORDED],
        ),
        accepted=False,
        forced_escalation=True,
        rejection_reason=reason,
    )


def validate_directive(
    directive: ActionDirective,
    parameter_rules: Mapping[str, ParameterRule],
    *,
    allowed_actions: frozenset[str],
    current_parameters: Mapping[str, int | float] | None = None,
) -> GuardrailResult:
    """Enforce action policy, allowlist, numeric type, finite bounds and restart invariants.

    Args:
        directive: Parsed model output.
        parameter_rules: Allowlist parsed from the OKF `allowed-parameters` concept.
        allowed_actions: Actions the OKF runbooks permit for the failure type
            (`LoadedSkill.allowed_actions_for`); anything else is rejected.
        current_parameters: Values the run is currently using (from the request),
            used to reject patches that change nothing.

    Never silently clamps out-of-range values: any violation forces `escalate_to_human`.
    """
    action = directive.action
    mods = directive.parameter_modifications
    concepts = directive.okf_concepts_used

    if action not in allowed_actions:
        return _reject_to_escalation(
            f"action={action!r} is not permitted by the OKF runbook for this failure type "
            f"(allowed: {sorted(allowed_actions)})",
            concepts,
        )

    if action in ("escalate_to_human", "resubmit_walltime") and mods:
        return _reject_to_escalation(
            f"action={action!r} must not modify parameters (got {sorted(mods.keys())})",
            concepts,
        )

    if action == "resubmit_walltime" and not directive.restart_from_checkpoint:
        return _reject_to_escalation(
            "resubmit_walltime requires restart_from_checkpoint=True",
            concepts,
        )

    if action == "warm_restart" and not directive.restart_from_checkpoint:
        return _reject_to_escalation(
            "warm_restart requires restart_from_checkpoint=True",
            concepts,
        )

    if action == "cold_restart" and directive.restart_from_checkpoint:
        return _reject_to_escalation(
            "cold_restart requires restart_from_checkpoint=False",
            concepts,
        )

    if action in ("warm_restart", "cold_restart") and not mods:
        return _reject_to_escalation(
            f"action={action!r} requires at least one allowlisted parameter modification",
            concepts,
        )

    validated_mods: dict[str, int | float] = {}
    for param_name, raw_val in mods.items():
        if param_name not in parameter_rules:
            return _reject_to_escalation(
                f"Parameter {param_name!r} is forbidden (physics/equilibrium or non-allowlisted)",
                concepts,
            )

        rule = parameter_rules[param_name]
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            return _reject_to_escalation(
                f"Parameter {param_name!r} has non-numeric type {type(raw_val).__name__}",
                concepts,
            )

        if not math.isfinite(float(raw_val)):
            return _reject_to_escalation(
                f"Parameter {param_name!r} is non-finite ({raw_val})",
                concepts,
            )

        if rule.value_type == "int":
            if isinstance(raw_val, float) and not raw_val.is_integer():
                return _reject_to_escalation(
                    f"Integer parameter {param_name!r} received fractional float {raw_val}",
                    concepts,
                )
            val_num: int | float = int(raw_val)
        else:
            val_num = float(raw_val)

        if val_num < rule.min_value or val_num > rule.max_value:
            return _reject_to_escalation(
                f"Parameter {param_name!r}={val_num} outside [{rule.min_value}, {rule.max_value}]",
                concepts,
            )

        validated_mods[param_name] = val_num

    # Checkpoint (warm) restarts may only change parameters the allowlist marks
    # `warm_restart_allowed: true` (pure timestep controls). Grid changes are
    # incompatible with binary checkpoints and dissipation changes alter the
    # physics mid-trajectory, so everything else requires a cold restart (PHYS-03).
    if action == "warm_restart" or directive.restart_from_checkpoint:
        cold_only = sorted(k for k in validated_mods if not parameter_rules[k].warm_restart_allowed)
        if cold_only:
            return _reject_to_escalation(
                f"Changing {cold_only} on a checkpoint restart is incompatible with binary "
                "checkpoints or alters dissipation mid-run; requires action='cold_restart' and "
                "restart_from_checkpoint=False",
                concepts,
            )

    if action in ("warm_restart", "cold_restart") and current_parameters:
        unchanged = [
            k
            for k, v in validated_mods.items()
            if k in current_parameters and float(current_parameters[k]) == float(v)
        ]
        if len(unchanged) == len(validated_mods):
            return _reject_to_escalation(
                f"Patch changes nothing: {sorted(unchanged)} already have the proposed values",
                concepts,
            )

    clean_directive = directive.model_copy(update={"parameter_modifications": validated_mods})
    return GuardrailResult(
        directive=clean_directive,
        accepted=True,
        forced_escalation=(action == "escalate_to_human"),
        rejection_reason=None,
    )
