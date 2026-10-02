"""I1 selects an executable depot with remaining vehicles, not only the cheapest."""
from dataclasses import replace

import numpy as np
import pytest

from vrptw.checker import evaluate
from vrptw.construct import best_i1, solomon_i1
from vrptw.instance import EnergyModel, make_instance


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_nearest_depot_with_closed_return_window_does_not_hide_feasible_alternative(constructor):
    inst = make_instance("depot closing", [(0, 0), (10, 0), (1, 0)], [0, 0, 1], [0, 0, 0],
                         [1, 100, 100], [0, 0, 0], 2, n_depots=2)
    assert not evaluate(inst, [[0, 2, 0]]).feasible
    assert evaluate(inst, [[1, 2, 1]]).feasible
    solution = constructor(inst)
    assert solution.unassigned == []
    assert solution.as_lists() == [[1, 2, 1]]
    assert evaluate(inst, solution.as_lists()).feasible


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_depot_opening_after_customer_due_uses_other_depot(constructor):
    inst = make_instance("depot opens late", [(0, 0), (10, 0), (1, 0)], [0, 0, 1], [30, 0, 0],
                         [100, 100, 20], [0, 0, 0], 2, n_depots=2)
    solution = constructor(inst)
    assert solution.unassigned == []
    assert solution.as_lists() == [[1, 2, 1]]
    assert evaluate(inst, solution.as_lists()).feasible


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_exhausted_nearest_depot_fleet_falls_back_to_a_depot_with_a_vehicle(constructor):
    inst = make_instance("two fleets", [(0, 0), (10, 0), (1, 0), (2, 0)], [0, 0, 1, 1],
                         [0] * 4, [100] * 4, [0] * 4, 1, n_depots=2, max_vehicles=1)
    solution = constructor(inst)
    assert solution.unassigned == []
    assert solution.n_routes() == 2
    assert sorted(route.depot for route in solution.routes) == [0, 1]
    assert evaluate(inst, solution.as_lists()).feasible


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_energy_feasibility_is_checked_separately_from_distance_preference(constructor):
    inst = make_instance("different energy distances", [(0, 0), (10, 0), (1, 0)], [0, 0, 1],
                         [0] * 3, [100] * 3, [0] * 3, 2, n_depots=2,
                         energy=EnergyModel(1, 0, 0, 5, 0), w_dist=1, w_energy=0)
    energy_dist = np.array([[0, 10, 5], [10, 0, 1], [5, 1, 0]], dtype=float)
    inst = replace(inst, energy_dist=energy_dist)
    solution = constructor(inst)
    assert solution.unassigned == []
    assert solution.as_lists() == [[1, 2, 1]]
    assert evaluate(inst, solution.as_lists()).feasible


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_customer_impossible_at_all_depots_remains_explicitly_unassigned(constructor):
    inst = make_instance("unreachable", [(0, 0), (10, 0), (5, 0)], [0, 0, 1], [0] * 3,
                         [100, 100, 1], [0] * 3, 2, n_depots=2)
    solution = constructor(inst)
    assert solution.as_lists() == [] and solution.unassigned == [2]
    assert evaluate(inst, solution.as_lists(), allow_unassigned=True).feasible


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_global_route_limit_does_not_create_extra_trips_to_use_alternate_fleet(constructor):
    inst = make_instance("route budget", [(0, 0), (10, 0), (1, 0), (2, 0)], [0, 0, 1, 1],
                         [0] * 4, [100] * 4, [0] * 4, 1, n_depots=2, max_vehicles=1)
    solution = constructor(inst, max_routes=1)
    assert solution.n_routes() == 1 and len(solution.unassigned) == 1
    assert evaluate(inst, solution.as_lists(), allow_unassigned=True).feasible


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_final_available_vehicle_can_still_take_multiple_customers(constructor):
    inst = make_instance("last vehicle", [(0, 0), (1, 0), (2, 0)], [0, 1, 1],
                         [0] * 3, [100] * 3, [0] * 3, 2, max_vehicles=1)
    solution = constructor(inst)
    assert solution.n_routes() == 1 and solution.unassigned == []
    assert evaluate(inst, solution.as_lists()).feasible


@pytest.mark.parametrize("constructor", [solomon_i1, best_i1])
def test_construct_validates_instances_mutated_since_construction(constructor):
    inst = make_instance("invalidated", [(0, 0), (1, 0)], [0, 1], [0, 0], [100, 100], [0, 0], 2)
    inst.capacity = float("nan")
    with pytest.raises(ValueError):
        constructor(inst)


def test_single_depot_existing_seed_and_insertion_choices_are_unchanged():
    inst = make_instance("single depot", [(0, 0), (2, 0), (1, 0), (3, 0)], [0, 1, 1, 1],
                         [0] * 4, [100] * 4, [0] * 4, 2)
    assert solomon_i1(inst, seed_rule="far").as_lists() == [[0, 1, 3, 0], [0, 2, 0]]
    assert solomon_i1(inst, seed_rule="due").as_lists() == [[0, 3, 1, 0], [0, 2, 0]]
    assert evaluate(inst, best_i1(inst).as_lists()).feasible
