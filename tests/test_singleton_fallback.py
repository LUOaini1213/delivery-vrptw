"""Singleton completion prefers feasible depots but never drops late deliveries."""
from dataclasses import replace

import numpy as np
import pytest

from vrptw.checker import evaluate
from vrptw.drone import complete_with_singletons
from vrptw.instance import EnergyModel, make_instance
from vrptw.routes import Route, Solution
from vrptw.sim import Scenario, execute_trip


def two_depots(**kwargs):
    return make_instance("singleton depots", [(0, 0), (10, 0), (1, 0)], [0, 0, 1],
                         [0] * 3, [100] * 3, [0] * 3, 2, n_depots=2, **kwargs)


@pytest.mark.parametrize("constraint", ["depot_close", "depot_open", "battery", "duration"])
def test_nearest_infeasible_depot_does_not_hide_a_feasible_singleton(constraint):
    inst = two_depots()
    if constraint == "depot_close":
        inst = replace(inst, due=np.array([1, 100, 100], dtype=float))
    elif constraint == "depot_open":
        inst = replace(inst, ready=np.array([30, 0, 0], dtype=float),
                       due=np.array([100, 100, 20], dtype=float))
    elif constraint == "battery":
        inst = replace(inst, energy=EnergyModel(1, 0, 0, 5, 0),
                       energy_dist=np.array([[0, 10, 5], [10, 0, 1], [5, 1, 0]], dtype=float))
    else:
        inst = replace(inst, max_duration=5,
                       time=np.array([[0, 10, 5], [10, 0, 1], [5, 1, 0]], dtype=float))
    assert not evaluate(inst, [[0, 2, 0]]).feasible
    assert evaluate(inst, [[1, 2, 1]]).feasible
    routes = complete_with_singletons(inst, Solution([], [2]))
    assert routes == [[1, 2, 1]]
    assert evaluate(inst, routes).feasible


def test_existing_route_consumes_a_vehicle_without_being_changed():
    inst = make_instance("existing route", [(0, 0), (10, 0), (2, 0), (1, 0)], [0, 0, 1, 1],
                         [0] * 4, [100] * 4, [0] * 4, 2, n_depots=2, max_vehicles=1)
    existing = Route(inst, [0, 2, 0])
    solution = Solution([existing], [3])
    routes = complete_with_singletons(inst, solution)
    assert routes == [[0, 2, 0], [1, 3, 1]]
    assert evaluate(inst, routes).feasible
    assert solution.routes == [existing] and solution.unassigned == [3]
    assert existing.nodes == [0, 2, 0]
    # Returning independent lists also protects the already dispatched route.
    routes[0][1] = 3
    assert existing.nodes == [0, 2, 0]


def test_each_appended_singleton_reserves_its_vehicle_for_following_customers():
    inst = make_instance("sequential slots", [(0, 0), (10, 0), (1, 0), (2, 0)], [0, 0, 1, 1],
                         [0] * 4, [100] * 4, [0] * 4, 2, n_depots=2, max_vehicles=1)
    routes = complete_with_singletons(inst, Solution([], [2, 3]))
    assert routes == [[0, 2, 0], [1, 3, 1]]
    assert evaluate(inst, routes).feasible


def test_all_fleets_full_still_dispatches_a_nearest_depot_fallback():
    inst = make_instance("full fleets", [(0, 0), (10, 0), (2, 0), (9, 0), (1, 0)],
                         [0, 0, 1, 1, 1], [0] * 5, [100] * 5, [0] * 5, 2,
                         n_depots=2, max_vehicles=1)
    existing = [Route(inst, [0, 2, 0]), Route(inst, [1, 3, 1])]
    routes = complete_with_singletons(inst, Solution(existing, [4]))
    assert routes == [[0, 2, 0], [1, 3, 1], [0, 4, 0]]
    result = evaluate(inst, routes)
    assert not result.feasible
    assert any("depot 0" in violation for violation in result.violations)


def test_zero_vehicle_limit_preserves_execution_fallback():
    inst = two_depots(max_vehicles=0)
    routes = complete_with_singletons(inst, Solution([], [2]))
    assert routes == [[0, 2, 0]]
    assert not evaluate(inst, routes).feasible


def test_all_customer_windows_missed_still_serves_and_records_lateness():
    inst = two_depots(energy=EnergyModel(1, 0, 0, 100, 0))
    inst = replace(inst, ready=np.array([20, 20, 0], dtype=float),
                   due=np.array([100, 100, 5], dtype=float))
    routes = complete_with_singletons(inst, Solution([], [2]))
    assert routes == [[0, 2, 0]]
    assert not evaluate(inst, routes).feasible
    calm = Scenario(0, np.zeros(1), np.zeros(1), np.ones((3, 3)), np.ones(3), np.ones((1, 3, 3)))
    _, _, served = execute_trip(inst, calm, routes[0], dep=20)
    assert served == {2: (21, 16)}


def test_capacity_infeasibility_is_retained_as_explicit_execution_fallback():
    inst = replace(two_depots(), capacity=0)
    routes = complete_with_singletons(inst, Solution([], [2]))
    assert routes == [[0, 2, 0]]
    result = evaluate(inst, routes)
    assert not result.feasible and any("load" in violation for violation in result.violations)


def test_feasible_nearest_depot_policy_remains_unchanged():
    inst = make_instance("nearest second depot", [(0, 0), (10, 0), (9, 0)], [0, 0, 1],
                         [0] * 3, [100] * 3, [0] * 3, 2, n_depots=2)
    assert complete_with_singletons(inst, Solution([], [2])) == [[1, 2, 1]]


def test_reassigned_invalid_instance_is_rejected():
    inst = two_depots()
    inst.capacity = float("nan")
    with pytest.raises(ValueError):
        complete_with_singletons(inst, Solution([], [2]))
