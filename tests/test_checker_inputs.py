"""Malformed tours cannot be scored as valid travel or hide unserved customers."""
import numpy as np
import pytest

from vrptw.checker import evaluate, min_duration, route_energy, route_schedule
from vrptw.instance import EnergyModel, make_instance


@pytest.fixture
def inst():
    return make_instance("inputs", [(0, 0), (1, 0), (2, 0)], [0, 2, 1], [0, 0, 0], [100, 100, 100],
                         [0, 1, 1], 10, energy=EnergyModel(1, 0.5, 5, 1000, 0),
                         vehicle_cost=7, w_dist=2, w_energy=3, max_vehicles=2)


BAD_ROUTES = [
    [-1, 1, -1], [0, -1, 0], [3, 1, 3], [0, 3, 0], [0, 1, 3],
    [False, 1, False], [0, True, 0], [0, np.bool_(True), 0],
    [0.0, 1, 0.0], [0, 1.0, 0], [0, np.float64(1), 0], [0, "1", 0],
    [0, float("nan"), 0], [0, float("inf"), 0], [0, None, 0], [0, [1], 0],
    [0, 1, 0, 2, 0], [0, 1, 1, 0], [1, 2, 1], [0, 1, 2],
    None, 7, "0,1,0", {0, 1}, {0: 1}, np.array([[0, 1, 0]]),
]


@pytest.mark.parametrize("bad", BAD_ROUTES)
def test_bad_route_is_reported_without_corrupting_valid_route_statistics(inst, bad):
    expected = evaluate(inst, [[0, 1, 0]], allow_unassigned=True)
    actual = evaluate(inst, [bad, [0, 1, 0]], allow_unassigned=True)
    assert not actual.feasible
    assert any("route 0:" in text for text in actual.violations)
    assert not any("visited twice" in text for text in actual.violations)
    for name in ("n_routes", "distance", "energy", "cost", "route_energy", "route_end"):
        assert getattr(actual, name) == getattr(expected, name)


def test_negative_depot_is_not_a_valid_zero_distance_alias(inst):
    result = evaluate(inst, [[-1, 1, -1]])
    assert not result.feasible
    assert result.n_routes == 0
    assert result.cost == result.distance == result.energy == 0
    assert any("unserved" in text and "2" in text for text in result.violations)


@pytest.mark.parametrize("empty", [[], [0], [0, 0]])
def test_submitted_empty_tour_is_invalid_but_has_no_trip_charge(inst, empty):
    result = evaluate(inst, [empty], allow_unassigned=True)
    assert not result.feasible
    assert any("no customer" in text for text in result.violations)
    assert result.n_routes == result.distance == result.energy == result.cost == 0
    assert result.route_energy == result.route_end == []


@pytest.mark.parametrize("helper", [route_schedule, min_duration, route_energy])
@pytest.mark.parametrize("bad", [[-1, 1, -1], [0, 3, 0], [0, True, 0], [0, 1.0, 0],
                                 [0, 1, 0, 2, 0], [0, 1, 1, 0], [0, 1], [1], None])
def test_public_helpers_raise_value_error_for_invalid_tours(inst, helper, bad):
    with pytest.raises(ValueError):
        helper(inst, bad)


@pytest.mark.parametrize("nodes,times", [([], []), ([0], [0]), ([0, 0], [0, 0])])
def test_public_helpers_keep_neutral_empty_tour_contract(inst, nodes, times):
    assert route_schedule(inst, nodes) == times
    assert min_duration(inst, nodes) == 0
    assert route_energy(inst, nodes) == 0


def test_numpy_integer_nodes_and_both_levels_of_generators_are_supported(inst):
    lists = [[0, 1, 0], [0, 2, 0]]
    expected = evaluate(inst, lists)
    assert expected.feasible and expected.n_routes == 2
    assert expected.distance == 6
    assert expected.energy == 18
    assert expected.cost == 80
    generated = ((np.int64(node) for node in route) for route in lists)
    assert evaluate(inst, generated) == expected
    assert evaluate(inst, np.asarray(lists, dtype=np.int32)) == expected
    assert route_schedule(inst, (np.int64(node) for node in lists[0])) == [0, 1, 3]
    assert min_duration(inst, iter(lists[0])) == 3
    assert route_energy(inst, iter(lists[0])) == 8


def test_outer_generator_is_consumed_once_including_fleet_and_trip_cost(inst):
    class Once:
        def __init__(self):
            self.used = False

        def __iter__(self):
            assert not self.used, "route collection traversed twice"
            self.used = True
            yield [0, 1, 0]
            yield [0, 2, 0]

    result = evaluate(inst, Once())
    assert result.feasible and result.n_routes == 2 and result.cost == 80
    inst.max_vehicles = 1
    over_limit = evaluate(inst, (route for route in [[0, 1, 0], [0, 2, 0]]))
    assert not over_limit.feasible
    assert any("routes > max" in text for text in over_limit.violations)


@pytest.mark.parametrize("routes", [None, 1, True, "routes", {"route": [0, 1, 0]}])
def test_invalid_route_collection_is_reported_without_an_indexing_exception(inst, routes):
    result = evaluate(inst, routes)
    assert not result.feasible and result.n_routes == 0
    assert any("routes" in text for text in result.violations)


def test_empty_route_collection_is_valid_only_for_partial_evaluation(inst):
    assert not evaluate(inst, iter(())).feasible
    partial = evaluate(inst, iter(()), allow_unassigned=True)
    assert partial.feasible and partial.n_routes == partial.cost == 0


def test_well_formed_but_infeasible_routes_keep_their_actual_numeric_objective(inst):
    expected = evaluate(inst, [[0, 1, 2, 0]])
    inst.capacity = 1
    result = evaluate(inst, [[0, 1, 2, 0]])
    assert not result.feasible
    assert any("capacity" in text for text in result.violations)
    assert result.distance == expected.distance
    assert result.energy == expected.energy and result.cost == expected.cost


def test_different_legal_depots_cannot_start_and_end_one_route():
    inst = make_instance("two depots", [(0, 0), (10, 0), (1, 0)], [0, 0, 1], [0, 0, 0],
                         [100, 100, 100], [0, 0, 1], 10, n_depots=2)
    result = evaluate(inst, [[0, 2, 1]])
    assert not result.feasible and result.n_routes == 0
    assert result.distance == 0


def test_evaluate_rejects_an_instance_invalidated_after_construction(inst):
    bad_time = inst.time.copy()
    bad_time[0, 1] = float("nan")
    inst.time = bad_time
    with pytest.raises(ValueError):
        evaluate(inst, [[0, 1, 0], [0, 2, 0]])
