from math import inf, nan

import pytest

from mcss.dynamic.budget import WorkBudget
from mcss.dynamic.config import CarrierConfig
from mcss.dynamic.policy import LearnedActionPolicy


def _budget() -> WorkBudget:
    return WorkBudget(CarrierConfig(), (8, 10), n_samples=8, max_units=10_000_000)


def _ledger(budget: WorkBudget) -> tuple[float, dict[str, int], float]:
    return budget.used_units, dict(budget.calls), budget.remaining


@pytest.mark.parametrize("units", [nan, inf, -inf, -1.0])
def test_record_rejects_nonfinite_or_negative_units_without_mutating_ledger(units: float) -> None:
    budget = _budget()
    budget.record("encode", 10.0, views=2)
    before = _ledger(budget)

    with pytest.raises(ValueError, match="units"):
        budget.record("bad", units, views=1)

    assert _ledger(budget) == before


@pytest.mark.parametrize("views", [nan, inf, -inf, -1, 1.5, True])
def test_record_rejects_invalid_view_counts_without_mutating_ledger(views: object) -> None:
    budget = _budget()
    budget.record("encode", 10.0, views=2)
    before = _ledger(budget)

    with pytest.raises(ValueError, match="views"):
        budget.record("bad", 3.0, views=views)

    assert _ledger(budget) == before


def test_record_accepts_nonnegative_integer_views() -> None:
    budget = _budget()

    budget.record("encode", 10.0, views=2)

    assert budget.used_units == 10.0
    assert budget.calls == {"encode": 1, "encode_views": 2}


def test_learned_policy_declared_work_is_reserved_beyond_fixed_unit() -> None:
    config = CarrierConfig(
        feature_dim=4, hidden_dim=4, expansion_dim=8, grid_size=(4, 4, 4), token_count=16
    )
    policy = LearnedActionPolicy(hidden_dim=8)
    learned = WorkBudget(config, (8, 10), n_samples=8, policy=policy)
    fixed = WorkBudget(config, (8, 10), n_samples=8)

    assert learned.policy_units == policy.declared_work_units
    assert learned.policy_units > fixed.policy_units == 1
    assert learned.base_future_units(2, 1, 0) - fixed.base_future_units(2, 1, 0) == (
        learned.policy_units - 1
    )
