"""Outer-loop augmented Lagrangian (PHR) multiplier & penalty state for KR1/KR3 coupling.

Problem form (one outer iteration = one approximate inner minimization of ``L_A`` over x,
performed by the caller's optimizer; this module never sees x itself)::

    minimize f(x)  subject to  c_i(x) = 0   for i in E  (``constraint_names``)
                               g_j(x) <= 0  for j in I  (``inequality_names``)

Residuals are the raw constraint values at the current iterate: ``c_i(x)`` for equalities and
``g_j(x)`` for inequalities, where ``g_j(x) > 0`` is a violation (e.g. a heat-flux ceiling is
passed as ``Q(x) - Q_max``, a beta floor as ``beta_min - beta(x)``).

Powell-Hestenes-Rockafellar (PHR) augmented Lagrangian, "+" sign convention throughout::

    L_A(x; lam, mu) = f(x) + sum_{i in E} [lam_i c_i(x) + (mu/2) c_i(x)^2]
                           + sum_{j in I} (1/(2 mu)) [max(0, lam_j + mu g_j(x))^2 - lam_j^2]

The ``- lam_j^2`` shift is constant in x (it does not move the minimizer) and makes each
inequality term vanish at a feasible, complementary point. ``L_A`` is continuously
differentiable, with ``grad_x L_A = grad f + sum_k w_k grad c_k`` for the weights returned by
:meth:`AugmentedLagrangianState.gradient_weights`. The first-order multiplier updates are those
weights, safeguarded to ``[-multiplier_max, multiplier_max]`` (E) and ``[0, multiplier_max]`` (I)::

    lam_i <- clip(lam_i + mu c_i(x), -multiplier_max, multiplier_max)    (i in E)
    lam_j <- min(multiplier_max, max(0, lam_j + mu g_j(x)))               (j in I)

Sign-convention note: Nocedal & Wright write ``L_A = f - lam^T c`` and inequalities as
``c_i(x) >= 0``; substituting ``lam -> -lam`` for equalities and ``c = -g`` for inequalities
gives exactly the formulas above (inequality multipliers stay non-negative in both).

References:
  - J. Nocedal & S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006), Ch. 17:
    Sec. 17.3 (method of multipliers for equality constraints, Framework 17.3) and Sec. 17.4
    (practical augmented Lagrangian methods: slack elimination for inequality constraints and
    the projected multiplier update ``lam <- max(lam - mu c, 0)`` for ``c(x) >= 0``).
  - R. T. Rockafellar, "The multiplier method of Hestenes and Powell applied to convex
    programming", J. Optim. Theory Appl. 12, 555-562 (1973).
  - R. Andreani, E. G. Birgin, J. M. Martinez & M. L. Schuverdt, "On augmented Lagrangian
    methods with general lower-level constraints", SIAM J. Optim. 18, 1286-1309 (2008): the
    combined infeasibility/complementarity measure ``max(g_j, -lam_j/mu)`` and the
    insufficient-progress penalty test used by :meth:`AugmentedLagrangianState.update_outer_loop`.
  - P. F. Gil, W. Li, J. Stratton, A. A. Kaptanoglu & E. V. Stenson, "Stellarator coils for
    future fusion reactors via an augmented Lagrangian approach", Phys. Rev. Lett. 137, 065101
    (2026), doi:10.1103/n7gk-922h (augmented Lagrangian method applied to stellarator design).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real


def _finite_float(value: object, label: str) -> float:
    """Return ``value`` as a finite float; reject bools, non-numerics, NaN and +/-inf."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return out


def _validate_names(names: object, label: str) -> list[str]:
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError(f"{label} must be a list of constraint names, got {names!r}")
    out: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} entries must be non-empty strings, got {name!r}")
        out.append(name)
    duplicates = sorted({name for name in out if out.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate names in {label}: {duplicates}")
    return out


@dataclass
class AugmentedLagrangianState:
    """Lagrange multipliers ``lam`` and quadratic penalty ``mu`` carried across outer iterations.

    ``multipliers`` holds one estimate per constraint name; equality (``constraint_names``) and
    inequality (``inequality_names``) names must be disjoint. Inequality multipliers always stay
    in ``[0, multiplier_max]`` (dual feasibility). Equality-only usage is unchanged from the
    original equality-constrained interface.

    Typical outer loop (``inner_minimize`` is the caller's optimizer)::

        state = AugmentedLagrangianState(["force_balance"], inequality_names=["q_ceiling"])
        for _ in range(max_outer_iterations):
            x = inner_minimize(
                lambda x: state.evaluate_augmented_objective(
                    f(x), {"force_balance": c(x)}, {"q_ceiling": g(x)}
                ),
                x,
            )
            state.update_outer_loop({"force_balance": c(x)}, {"q_ceiling": g(x)})
            if state.converged:
                break

    Termination: :attr:`converged` certifies primal feasibility *and* complementarity within
    ``constraint_tol`` at the last outer iterate (measured with the multipliers and penalty of
    the subproblem that produced it). Dual stationarity (small ``||grad_x L_A||``) is the inner
    solver's responsibility (Nocedal & Wright, Framework 17.3). Callers must cap the number of
    outer iterations; ``penalty_mu == penalty_max`` with ``converged`` still False means the
    constraints are likely infeasible or badly scaled and the run should be reported as failed.
    """

    constraint_names: list[str] = field(default_factory=list)
    multipliers: dict[str, float] = field(default_factory=dict)
    # TODO(david): confirm parameter values
    penalty_mu: float = 10.0
    penalty_growth: float = 2.0
    penalty_max: float = 1.0e5
    improvement_ratio: float = 0.75
    constraint_tol: float = 0.05
    multiplier_max: float = 1.0e4
    prev_max_violation: float = math.inf
    inequality_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.constraint_names = _validate_names(self.constraint_names, "constraint_names")
        self.inequality_names = _validate_names(self.inequality_names, "inequality_names")
        overlap = sorted(set(self.constraint_names) & set(self.inequality_names))
        if overlap:
            raise ValueError(
                f"Constraint names must be unique across equality and inequality sets: {overlap}"
            )
        self._validate_hyperparameters()

        if not isinstance(self.multipliers, Mapping):
            raise ValueError(f"multipliers must be a mapping, got {self.multipliers!r}")
        declared = set(self.constraint_names) | set(self.inequality_names)
        unknown = sorted(repr(name) for name in self.multipliers if name not in declared)
        if unknown:
            raise ValueError(f"multipliers given for undeclared constraints: {unknown}")
        multipliers: dict[str, float] = {}
        for name in self.constraint_names:
            lam = _finite_float(self.multipliers.get(name, 0.0), f"multiplier {name!r}")
            if abs(lam) > self.multiplier_max:
                raise ValueError(f"multiplier {name!r}={lam} exceeds multiplier_max")
            multipliers[name] = lam
        for name in self.inequality_names:
            lam = _finite_float(self.multipliers.get(name, 0.0), f"multiplier {name!r}")
            if not 0.0 <= lam <= self.multiplier_max:
                raise ValueError(
                    f"inequality multiplier {name!r}={lam} must lie in [0, multiplier_max]"
                )
            multipliers[name] = lam
        # Private copy: never mutate (or alias) a dict owned by the caller.
        self.multipliers = multipliers

    def _validate_hyperparameters(self) -> None:
        for attr in (
            "penalty_mu",
            "penalty_growth",
            "penalty_max",
            "improvement_ratio",
            "constraint_tol",
            "multiplier_max",
        ):
            setattr(self, attr, _finite_float(getattr(self, attr), attr))
        if self.penalty_mu <= 0.0:
            raise ValueError(f"penalty_mu must be > 0, got {self.penalty_mu}")
        if self.penalty_growth <= 1.0:
            raise ValueError(f"penalty_growth must be > 1, got {self.penalty_growth}")
        if self.penalty_max < self.penalty_mu:
            raise ValueError("penalty_max must be >= penalty_mu")
        if not 0.0 < self.improvement_ratio < 1.0:
            raise ValueError(f"improvement_ratio must lie in (0, 1), got {self.improvement_ratio}")
        if self.constraint_tol <= 0.0:
            raise ValueError(f"constraint_tol must be > 0, got {self.constraint_tol}")
        if self.multiplier_max <= 0.0:
            raise ValueError(f"multiplier_max must be > 0, got {self.multiplier_max}")
        prev = self.prev_max_violation
        if isinstance(prev, bool) or not isinstance(prev, Real) or not float(prev) >= 0.0:
            raise ValueError(f"prev_max_violation must be >= 0 (inf allowed), got {prev!r}")
        self.prev_max_violation = float(prev)

    def _validate_residuals(
        self, residuals: Mapping[str, float] | None, names: list[str], label: str
    ) -> dict[str, float]:
        if residuals is None:
            residuals = {}
        if not isinstance(residuals, Mapping):
            raise ValueError(f"{label} must be a mapping of constraint name -> residual")
        missing = [name for name in names if name not in residuals]
        if missing:
            raise ValueError(f"Missing {label} for constraints: {missing}")
        declared = set(names)
        unknown = sorted(repr(key) for key in residuals if key not in declared)
        if unknown:
            # Fail closed: a misspelled or misfiled constraint must never be silently ignored.
            raise ValueError(f"Unknown {label} keys (not declared constraints): {unknown}")
        return {
            name: _finite_float(residuals[name], f"Constraint residual {name!r}") for name in names
        }

    def _validated(
        self,
        equality_residuals: Mapping[str, float] | None,
        inequality_residuals: Mapping[str, float] | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        # Guard against post-construction mutation of the penalty (it divides below).
        if not (math.isfinite(self.penalty_mu) and self.penalty_mu > 0.0):
            raise ValueError(f"penalty_mu must be finite and > 0, got {self.penalty_mu!r}")
        eq = self._validate_residuals(
            equality_residuals, self.constraint_names, "equality_residuals"
        )
        ineq = self._validate_residuals(
            inequality_residuals, self.inequality_names, "inequality_residuals"
        )
        return eq, ineq

    def _kkt_violation(self, eq: dict[str, float], ineq: dict[str, float]) -> float:
        violation = 0.0
        for c_val in eq.values():
            violation = max(violation, abs(c_val))
        for name, g_val in ineq.items():
            # Andreani et al. (2008): |max(g, -lam/mu)| is 0 iff g <= 0 and lam * g == 0
            # (up to the 1/mu scaling), so it measures infeasibility AND complementarity.
            v_j = max(g_val, -self.multipliers[name] / self.penalty_mu)
            violation = max(violation, abs(v_j))
        return violation

    def _gradient_weights(self, eq: dict[str, float], ineq: dict[str, float]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name, c_val in eq.items():
            weights[name] = self.multipliers[name] + self.penalty_mu * c_val
        for name, g_val in ineq.items():
            weights[name] = max(0.0, self.multipliers[name] + self.penalty_mu * g_val)
        return weights

    def evaluate_augmented_objective(
        self,
        primary_objective: float,
        equality_residuals: Mapping[str, float] | None = None,
        inequality_residuals: Mapping[str, float] | None = None,
    ) -> float:
        """Return the PHR augmented Lagrangian ``L_A(x; lam, mu)`` (module docstring formula).

        Equality terms: ``lam_i c_i + (mu/2) c_i^2``. Inequality terms:
        ``(1/(2 mu)) [max(0, lam_j + mu g_j)^2 - lam_j^2]``, evaluated in the algebraically
        equal, cancellation-free piecewise form (Nocedal & Wright Sec. 17.4).
        """
        total = _finite_float(primary_objective, "primary_objective")
        eq, ineq = self._validated(equality_residuals, inequality_residuals)
        mu = self.penalty_mu
        for name, c_val in eq.items():
            total += self.multipliers[name] * c_val + 0.5 * mu * c_val * c_val
        for name, g_val in ineq.items():
            lam = self.multipliers[name]
            if lam + mu * g_val > 0.0:
                total += lam * g_val + 0.5 * mu * g_val * g_val
            else:
                total -= lam * lam / (2.0 * mu)
        return total

    def gradient_weights(
        self,
        equality_residuals: Mapping[str, float] | None = None,
        inequality_residuals: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return ``w`` with ``grad_x L_A = grad f + sum_k w_k grad c_k(x)`` at the given residuals.

        ``w_i = lam_i + mu c_i`` (E) and ``w_j = max(0, lam_j + mu g_j)`` (I): the unsafeguarded
        first-order multiplier estimates, for gradient-based inner solvers.
        """
        eq, ineq = self._validated(equality_residuals, inequality_residuals)
        return self._gradient_weights(eq, ineq)

    def kkt_violation(
        self,
        equality_residuals: Mapping[str, float] | None = None,
        inequality_residuals: Mapping[str, float] | None = None,
    ) -> float:
        """Return ``max(max_i |c_i|, max_j |max(g_j, -lam_j/mu)|)`` with the current ``lam, mu``."""
        eq, ineq = self._validated(equality_residuals, inequality_residuals)
        return self._kkt_violation(eq, ineq)

    def is_feasible(
        self,
        equality_residuals: Mapping[str, float] | None = None,
        inequality_residuals: Mapping[str, float] | None = None,
    ) -> bool:
        """Return True if :meth:`kkt_violation` is within ``constraint_tol``.

        Call it *before* :meth:`update_outer_loop` for the iterate just produced by the inner
        solve (the measure uses the multipliers of that subproblem).
        """
        return self.kkt_violation(equality_residuals, inequality_residuals) <= self.constraint_tol

    @property
    def converged(self) -> bool:
        """Feasibility + complementarity within ``constraint_tol`` at the last outer update."""
        return self.prev_max_violation <= self.constraint_tol

    def update_outer_loop(
        self,
        equality_residuals: Mapping[str, float] | None = None,
        inequality_residuals: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Apply the safeguarded PHR multiplier updates, then grow ``mu`` if progress stalls.

        The violation measure ``v_k`` (:meth:`kkt_violation`, taken with the subproblem's
        ``lam, mu`` before the update) is stored in ``prev_max_violation``. The penalty grows by
        ``penalty_growth`` (capped at ``penalty_max``) only if ``v_k > constraint_tol`` and
        ``v_k > improvement_ratio * v_{k-1}`` (Andreani et al. 2008; Nocedal & Wright Sec. 17.4).
        Returns a copy of all multipliers.
        """
        eq, ineq = self._validated(equality_residuals, inequality_residuals)
        violation = self._kkt_violation(eq, ineq)
        weights = self._gradient_weights(eq, ineq)
        for name in self.constraint_names:
            self.multipliers[name] = max(
                -self.multiplier_max, min(self.multiplier_max, weights[name])
            )
        for name in self.inequality_names:
            self.multipliers[name] = min(self.multiplier_max, weights[name])

        if (
            violation > self.constraint_tol
            and violation > self.improvement_ratio * self.prev_max_violation
        ):
            self.penalty_mu = min(self.penalty_max, self.penalty_mu * self.penalty_growth)
        self.prev_max_violation = violation
        return dict(self.multipliers)
