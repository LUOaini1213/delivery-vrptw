from dataclasses import replace

import numpy as np
import pytest

from vrptw.alns import solve
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.instance import make_instance, subset
from vrptw.routes import Route, Solution


def zero_instance():
    return make_instance("zero", [[0, 0], [1, 0]], [0, 0], [0, 0], [10, 10], [0, 0], 0)


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_zero_capacity_and_empty_instances(hierarchical, empty):
    inst = zero_instance()
    if empty:
        inst = subset(inst, 0)
    result, _ = solve(inst, best_i1(inst), iters=20, seed=0, hierarchical=hierarchical)
    ev = evaluate(inst, result.as_lists())
    assert ev.feasible
    assert ev.cost == (0 if empty else 2)


@pytest.mark.parametrize("missing", [[], [1, 1], [True], [-1], [2]])
def test_invalid_partial_solution_is_rejected(missing):
    inst = zero_instance()
    with pytest.raises(ValueError):
        solve(inst, Solution([], missing), 0, 0)


def test_starting_cache_rebuilt_for_current_instance():
    inst = zero_instance()
    start = best_i1(inst)
    inst.dist = inst.dist * 3
    result, _ = solve(inst, start, 0, 0)
    assert result.cost() == 6
    assert evaluate(inst, result.as_lists()).cost == result.cost()


def test_infeasible_start_is_rejected_even_with_zero_iterations():
    inst = zero_instance()
    start = best_i1(inst)
    inst.time = np.array([[0., 20.], [20., 0.]])
    with pytest.raises(ValueError, match="starting solution"):
        solve(inst, start, 0, 0)


def test_positive_demand_zero_capacity_stays_unassigned():
    inst = replace(zero_instance(), demand=np.array([0., 1.]))
    result, _ = solve(inst, best_i1(inst), 20, 0, hierarchical=True)
    assert result.as_lists() == []
    assert result.unassigned == [1]


def test_valid_partial_start_is_repaired():
    inst = zero_instance()
    result, _ = solve(inst, Solution([], [1]), 10, 0)
    assert evaluate(inst, result.as_lists()).feasible


def test_duplicate_route_in_start_is_rejected():
    inst = zero_instance()
    route = Route(inst, [0, 1, 0])
    with pytest.raises(ValueError, match="starting solution"):
        solve(inst, Solution([route, route]), 0, 0)
