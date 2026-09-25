"""Declared operator-work proxy and exact call ledger, not a measured FLOP counter."""

from collections import Counter
from math import isfinite, prod
from numbers import Integral, Real

from mcss.dynamic.types import Action


def _policy_units(policy) -> int:
    """Return one-step policy work, preserving fixed-policy cost one."""

    if policy is None:
        return 1
    value = getattr(policy, "declared_work_units", 1)
    if callable(value):
        value = value()
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("policy declared_work_units must be a positive integer")
    return int(value)


class WorkBudget:
    def __init__(self, config, image_size, n_samples, max_units=1e12, *, policy=None):
        if not isfinite(max_units) or max_units <= 0:
            raise ValueError("Budget must be finite and positive")
        self.config = config
        self.max_units = float(max_units)
        self.used_units = 0.0
        self.calls = Counter()
        self.encode_units = prod(image_size) * config.feature_dim
        self.render_units = prod(image_size) * n_samples
        self.policy_units = _policy_units(policy)

    def materialize_units(self, views):
        c = self.config
        return prod(c.grid_size) * (
            views * c.feature_dim + 2 * c.hidden_dim * c.expansion_dim + 27 * c.hidden_dim**2
        )

    @property
    def remaining(self):
        return self.max_units - self.used_units

    def record(self, operation, units, *, views=0):
        if isinstance(units, bool) or not isinstance(units, Real) or not isfinite(units):
            raise ValueError("units must be finite and nonnegative")
        if units < 0:
            raise ValueError("units must be finite and nonnegative")
        if isinstance(views, bool) or not isinstance(views, Integral) or views < 0:
            raise ValueError("views must be a nonnegative integer")
        units = float(units)
        views = int(views)
        if units > self.remaining + 1e-6:
            raise ValueError("Required operation exceeds declared budget")
        self.calls[operation] += 1
        if views:
            self.calls[f"{operation}_views"] += views
        self.used_units += units

    def base_future_units(self, current_views, remaining_steps, query_count):
        return (
            sum(
                self.encode_units
                + self.render_units
                + self.policy_units
                + self.materialize_units(current_views + i)
                for i in range(1, remaining_steps + 1)
            )
            + query_count * self.render_units
        )

    def write_units(self, views):
        c = self.config
        # Both proposals are computed from the same trace, including single-location actions.
        outer = 2 * c.token_count * c.hidden_dim * c.expansion_dim
        return self.materialize_units(views) + outer

    def feasible_actions(self, views_after_append, remaining_steps, query_count):
        reserve = self.materialize_units(views_after_append) + self.base_future_units(
            views_after_append, remaining_steps, query_count
        )
        if self.remaining + 1e-6 < reserve:
            raise ValueError("Budget cannot cover mandatory future observations")
        if self.remaining + 1e-6 >= reserve + self.write_units(views_after_append):
            return tuple(Action)
        return (Action.OFF,)

    def report(self):
        return {
            "units_kind": "declared_operator_work_proxy_not_measured_flops",
            "max_units": self.max_units,
            "used_units": self.used_units,
            "remaining_units": self.remaining,
            "policy_units": self.policy_units,
            "calls": dict(self.calls),
        }
