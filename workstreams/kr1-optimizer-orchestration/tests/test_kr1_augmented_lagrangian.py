"""PHR augmented Lagrangian tests on analytic problems with known KKT points (ML-01).

Every problem below has a closed-form KKT pair (x*, lam*) in this module's "+" sign
convention (grad f + sum_k lam_k grad c_k = 0, lam_j >= 0 for g_j(x) <= 0). The outer loop is
driven exactly as a KR1/KR3 caller would drive it: an inner solver minimizes
``evaluate_augmented_objective`` using ``gradient_weights`` for the gradient, then
``update_outer_loop`` updates multipliers/penalty until ``converged``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from kr1_surrogate import AugmentedLagrangianState

Vec = tuple[float, float]
Mat = tuple[tuple[float, float], tuple[float, float]]
ZERO_H: Mat = ((0.0, 0.0), (0.0, 0.0))


@dataclass(frozen=True)
class Constraint:
    value: Callable[[Vec], float]
    grad: Callable[[Vec], Vec]
    hess: Callable[[Vec], Mat] = lambda _x: ZERO_H


@dataclass(frozen=True)
class Problem:
    f: Callable[[Vec], float]
    grad_f: Callable[[Vec], Vec]
    hess_f: Callable[[Vec], Mat]
    eq: dict[str, Constraint] = field(default_factory=dict)
    ineq: dict[str, Constraint] = field(default_factory=dict)


def _residuals(p: Problem, x: Vec) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {n: c.value(x) for n, c in p.eq.items()},
        {n: c.value(x) for n, c in p.ineq.items()},
    )


def _augmented_value(state: AugmentedLagrangianState, p: Problem, x: Vec) -> float:
    eq, ineq = _residuals(p, x)
    return state.evaluate_augmented_objective(p.f(x), eq, ineq)


def _inner_minimize(state: AugmentedLagrangianState, p: Problem, x: Vec) -> Vec:
    """Damped (Levenberg-shifted) Newton on L_A(.; lam, mu) with Armijo backtracking."""
    for _ in range(200):
        eq, ineq = _residuals(p, x)
        w = state.gradient_weights(eq, ineq)
        gx, gy = p.grad_f(x)
        (h11, h12), (_, h22) = p.hess_f(x)
        for name, con in (*p.eq.items(), *p.ineq.items()):
            dcx, dcy = con.grad(x)
            (c11, c12), (_, c22) = con.hess(x)
            gx, gy = gx + w[name] * dcx, gy + w[name] * dcy
            h11, h12, h22 = h11 + w[name] * c11, h12 + w[name] * c12, h22 + w[name] * c22
            if name in p.eq or w[name] > 0.0:  # active PHR piece contributes mu * grad grad^T
                mu = state.penalty_mu
                h11, h12, h22 = h11 + mu * dcx * dcx, h12 + mu * dcx * dcy, h22 + mu * dcy * dcy
        if math.hypot(gx, gy) < 1e-11:
            return x
        for shift in (0.0, 1e-6, 1e-3, 1e-1, 1.0, 10.0, 1e3):
            a, d = h11 + shift, h22 + shift
            det = a * d - h12 * h12
            if a > 0.0 and det > 0.0:
                break
        px, py = (-(d * gx - h12 * gy) / det, -(a * gy - h12 * gx) / det)
        f0, slope, t = _augmented_value(state, p, x), gx * px + gy * py, 1.0
        noise = 1e-14 * max(1.0, abs(f0))  # accept steps whose decrease is below round-off
        while (
            _augmented_value(state, p, (x[0] + t * px, x[1] + t * py))
            > f0 + 1e-4 * t * slope + noise
        ):
            t *= 0.5
            assert t > 1e-14, "line search failed"
        x = (x[0] + t * px, x[1] + t * py)
    raise AssertionError("inner Newton solve did not converge")


def _solve(state: AugmentedLagrangianState, p: Problem, x0: Vec, max_outer: int = 60) -> Vec:
    x = x0
    for _ in range(max_outer):
        x = _inner_minimize(state, p, x)
        state.update_outer_loop(*_residuals(p, x))
        if state.converged:
            return x
    raise AssertionError(f"outer loop did not converge: {state}")


def _quadratic(cx: float, cy: float) -> tuple[Callable, Callable, Callable]:
    return (
        lambda x: (x[0] - cx) ** 2 + (x[1] - cy) ** 2,
        lambda x: (2.0 * (x[0] - cx), 2.0 * (x[1] - cy)),
        lambda _x: ((2.0, 0.0), (0.0, 2.0)),
    )


SUM_LE_2 = Constraint(lambda x: x[0] + x[1] - 2.0, lambda _x: (1.0, 1.0))
CIRCLE = Constraint(
    lambda x: x[0] ** 2 + x[1] ** 2 - 2.0,
    lambda x: (2.0 * x[0], 2.0 * x[1]),
    lambda _x: ((2.0, 0.0), (0.0, 2.0)),
)
LINEAR_F = (lambda x: x[0] + x[1], lambda _x: (1.0, 1.0), lambda _x: ZERO_H)


@pytest.mark.parametrize(
    ("problem", "x0", "x_star", "lam_star"),
    [
        # Active linear inequality: min (x1-2)^2+(x2-1)^2 s.t. x1+x2 <= 2 -> x*=(1.5,0.5), lam*=1.
        (Problem(*_quadratic(2.0, 1.0), ineq={"g": SUM_LE_2}), (0.0, 0.0), (1.5, 0.5), {"g": 1.0}),
        # Inactive inequality: unconstrained minimizer (0.5,0.5) is strictly feasible -> lam*=0.
        (Problem(*_quadratic(0.5, 0.5), ineq={"g": SUM_LE_2}), (3.0, 3.0), (0.5, 0.5), {"g": 0.0}),
        # Nonlinear active inequality (Nocedal & Wright Example 17.1 posed as x1^2+x2^2 <= 2):
        # x*=(-1,-1), lam*=1/2 (N&W report -1/2 for the equality form under L_A = f - lam c).
        (Problem(*LINEAR_F, ineq={"g": CIRCLE}), (0.0, 0.0), (-1.0, -1.0), {"g": 0.5}),
        # Mixed: min x1^2+x2^2 s.t. x1+x2-1 = 0, 0.8-x1 <= 0 -> x*=(0.8,0.2), nu*=-0.4, lam*=1.2.
        (
            Problem(
                *_quadratic(0.0, 0.0),
                eq={"h": Constraint(lambda x: x[0] + x[1] - 1.0, lambda _x: (1.0, 1.0))},
                ineq={"g": Constraint(lambda x: 0.8 - x[0], lambda _x: (-1.0, 0.0))},
            ),
            (0.0, 0.0),
            (0.8, 0.2),
            {"h": -0.4, "g": 1.2},
        ),
        # Equality path (backward compatible): N&W Example 17.1, x1^2+x2^2 = 2 -> nu* = 1/2.
        (Problem(*LINEAR_F, eq={"h": CIRCLE}), (-1.2, -0.7), (-1.0, -1.0), {"h": 0.5}),
    ],
    ids=["active-linear", "inactive-linear", "active-circle", "mixed-eq-ineq", "equality-circle"],
)
def test_outer_loop_converges_to_known_kkt_point(
    problem: Problem, x0: Vec, x_star: Vec, lam_star: dict[str, float]
) -> None:
    state = AugmentedLagrangianState(
        constraint_names=list(problem.eq),
        inequality_names=list(problem.ineq),
        constraint_tol=1e-9,
    )
    x = _solve(state, problem, x0)

    assert x == pytest.approx(x_star, abs=1e-6)
    assert state.multipliers == pytest.approx(lam_star, abs=1e-6)
    assert state.converged
    eq, ineq = _residuals(problem, x)
    assert all(abs(c) <= 1e-8 for c in eq.values())  # primal feasibility
    assert all(g <= 1e-8 for g in ineq.values())
    assert all(state.multipliers[n] >= 0.0 for n in ineq)  # dual feasibility
    assert all(abs(state.multipliers[n] * g) <= 1e-8 for n, g in ineq.items())  # complementarity


def test_inactive_constraint_with_stale_multiplier_resets_to_zero() -> None:
    # lam=5 on a constraint the minimizer satisfies strictly: max(0, lam + mu g) must zero it.
    problem = Problem(*_quadratic(0.5, 0.5), ineq={"g": SUM_LE_2})
    state = AugmentedLagrangianState(
        inequality_names=["g"], multipliers={"g": 5.0}, constraint_tol=1e-9
    )
    x = _inner_minimize(state, problem, (0.0, 0.0))
    assert x == pytest.approx((0.5, 0.5), abs=1e-9)
    state.update_outer_loop(inequality_residuals={"g": SUM_LE_2.value(x)})
    assert state.multipliers["g"] == 0.0
    # Complementarity was violated at this iterate (lam > 0 on an inactive constraint).
    assert state.prev_max_violation == pytest.approx(0.5)
    assert not state.converged
    state.update_outer_loop(inequality_residuals={"g": SUM_LE_2.value(x)})
    assert state.converged


def test_phr_inequality_term_matches_reference_formula_and_is_c1() -> None:
    rng = random.Random(1234)
    for _ in range(200):
        lam, mu = rng.uniform(0.0, 5.0), rng.uniform(0.5, 50.0)
        g = rng.uniform(-3.0, 3.0)
        state = AugmentedLagrangianState(
            inequality_names=["g"], multipliers={"g": lam}, penalty_mu=mu
        )
        expected = (max(0.0, lam + mu * g) ** 2 - lam**2) / (2.0 * mu)
        value = state.evaluate_augmented_objective(1.0, inequality_residuals={"g": g})
        assert value == pytest.approx(1.0 + expected, rel=1e-12, abs=1e-12)

        # dL_A/dg equals the gradient weight max(0, lam + mu g) (central differences).
        h = 1e-6
        plus = state.evaluate_augmented_objective(0.0, inequality_residuals={"g": g + h})
        minus = state.evaluate_augmented_objective(0.0, inequality_residuals={"g": g - h})
        weight = state.gradient_weights(inequality_residuals={"g": g})["g"]
        assert (plus - minus) / (2 * h) == pytest.approx(weight, rel=1e-6, abs=1e-4)

    # Continuous across the switching surface lam + mu g = 0.
    state = AugmentedLagrangianState(inequality_names=["g"], multipliers={"g": 2.0})
    kink = -2.0 / state.penalty_mu
    left = state.evaluate_augmented_objective(0.0, inequality_residuals={"g": kink - 1e-12})
    right = state.evaluate_augmented_objective(0.0, inequality_residuals={"g": kink + 1e-12})
    assert left == pytest.approx(right, abs=1e-9)
    assert left == pytest.approx(-(2.0**2) / (2.0 * state.penalty_mu))


def test_equality_objective_and_update_are_unchanged() -> None:
    state = AugmentedLagrangianState(["a", "b"], multipliers={"a": 0.3, "b": -0.2}, penalty_mu=4.0)
    res = {"a": 0.1, "b": -0.25}
    expected = 2.0 + 0.3 * 0.1 + 2.0 * 0.1**2 + (-0.2) * (-0.25) + 2.0 * 0.25**2
    assert state.evaluate_augmented_objective(2.0, res) == pytest.approx(expected)
    assert state.update_outer_loop(res) == pytest.approx({"a": 0.7, "b": -1.2})


def test_penalty_grows_only_on_insufficient_progress_and_is_capped() -> None:
    state = AugmentedLagrangianState(
        ["h"], inequality_names=["g"], penalty_mu=10.0, penalty_max=40.0, constraint_tol=1e-3
    )
    state.update_outer_loop({"h": 0.5}, {"g": -1.0})  # first iterate: no reference, no growth
    assert state.penalty_mu == 10.0
    state.update_outer_loop({"h": 0.2}, {"g": -1.0})  # 0.2 <= 0.75 * 0.5: enough progress
    assert state.penalty_mu == 10.0
    state.update_outer_loop({"h": 0.18}, {"g": -1.0})  # 0.18 > 0.75 * 0.2: stalled -> grow
    assert state.penalty_mu == 20.0
    state.update_outer_loop({"h": 0.17}, {"g": -1.0})
    assert state.penalty_mu == 40.0
    state.update_outer_loop({"h": 0.17}, {"g": -1.0})
    assert state.penalty_mu == 40.0  # capped at penalty_max
    assert not state.converged


def test_stalled_inequality_violation_grows_penalty() -> None:
    state = AugmentedLagrangianState(inequality_names=["g"], constraint_tol=1e-3)
    state.update_outer_loop(inequality_residuals={"g": 0.5})
    state.update_outer_loop(inequality_residuals={"g": 0.45})  # 0.45 > 0.75 * 0.5: stalled
    assert state.penalty_mu == 20.0
    assert state.multipliers["g"] == pytest.approx(9.5)  # updated with the subproblem's mu=10


def test_strictly_feasible_inequality_never_triggers_growth_or_blocks_convergence() -> None:
    # A plain |g| measure would read 5.0 here and grow mu forever; max(g, -lam/mu) reads 0.
    state = AugmentedLagrangianState(inequality_names=["g"], constraint_tol=1e-6)
    for _ in range(3):
        state.update_outer_loop(inequality_residuals={"g": -5.0})
    assert state.penalty_mu == 10.0
    assert state.multipliers["g"] == 0.0
    assert state.converged
    assert state.is_feasible(inequality_residuals={"g": -5.0})
    assert not state.is_feasible(inequality_residuals={"g": 1e-3})


def test_inequality_multiplier_is_clamped_to_safeguard_box() -> None:
    state = AugmentedLagrangianState(inequality_names=["g"], multiplier_max=3.0)
    assert state.update_outer_loop(inequality_residuals={"g": 10.0})["g"] == 3.0
    assert state.update_outer_loop(inequality_residuals={"g": -10.0})["g"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"constraint_names": ["a"], "inequality_names": ["a"]},
        {"constraint_names": ["a", "a"]},
        {"constraint_names": "abc"},
        {"constraint_names": [""]},
        {"constraint_names": ["a"], "multipliers": {"b": 1.0}},
        {"inequality_names": ["g"], "multipliers": {"g": -0.1}},
        {"constraint_names": ["a"], "multipliers": {"a": float("nan")}},
        {"constraint_names": ["a"], "multipliers": {"a": 1e9}},
        {"penalty_mu": 0.0},
        {"penalty_mu": float("inf")},
        {"penalty_growth": 1.0},
        {"penalty_max": 1.0, "penalty_mu": 10.0},
        {"improvement_ratio": 1.0},
        {"improvement_ratio": 0.0},
        {"constraint_tol": 0.0},
        {"multiplier_max": -1.0},
        {"penalty_mu": True},
        {"prev_max_violation": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AugmentedLagrangianState(**kwargs)


def test_residual_validation_fails_closed() -> None:
    state = AugmentedLagrangianState(["h"], inequality_names=["g"])
    with pytest.raises(ValueError, match="Missing inequality_residuals"):
        state.update_outer_loop({"h": 0.1})
    with pytest.raises(ValueError, match="Unknown equality_residuals"):
        state.update_outer_loop({"h": 0.1, "g": 0.2}, {"g": 0.2})
    with pytest.raises(ValueError, match="must be numeric"):
        state.evaluate_augmented_objective(1.0, {"h": True}, {"g": 0.0})
    with pytest.raises(ValueError, match="must be finite"):
        state.kkt_violation({"h": 0.0}, {"g": float("nan")})
    with pytest.raises(ValueError, match="mapping"):
        state.gradient_weights([("h", 0.1)], {"g": 0.0})  # type: ignore[arg-type]
    assert state.multipliers == {"h": 0.0, "g": 0.0}  # failed calls left the state untouched
    assert state.penalty_mu == 10.0


def test_caller_multiplier_dict_is_not_aliased() -> None:
    initial = {"h": 0.5}
    state = AugmentedLagrangianState(["h"], multipliers=initial)
    state.update_outer_loop({"h": 1.0})
    assert initial == {"h": 0.5}
    assert state.multipliers["h"] == pytest.approx(10.5)
