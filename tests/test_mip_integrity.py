"""Connectivity and independent validation of MIP incumbents, including forged ones."""

from types import SimpleNamespace

import numpy as np
import pytest

from vrptw import mip
from vrptw.brute import brute_force
from vrptw.checker import evaluate
from vrptw.instance import EnergyModel, make_instance


def instance(coords=((0, 0), (1, 0), (1, 0)), *, n_depots=1, demand=None,
             service=None, due=None, capacity=1, **kw):
    n = len(coords)
    return make_instance("integrity", coords, [0] * n if demand is None else demand,
                         [0] * n, [100] * n if due is None else due,
                         [0] * n if service is None else service, capacity,
                         n_depots=n_depots, **kw)


def assert_complete(inst, result):
    assert result["routes"] is not None, result
    seen = [node for route in result["routes"] for node in route[1:-1]]
    assert sorted(seen) == list(inst.customers)
    assert all(route[0] == route[-1] and 0 <= route[0] < inst.n_depots for route in result["routes"])
    checked = evaluate(inst, result["routes"])
    assert checked.feasible, checked.violations
    assert result["objective"] == pytest.approx(checked.cost, abs=1e-6)


def test_zero_demand_colocated_customers_require_a_real_depot_route():
    data = instance()
    result = mip.solve_mip(data, time_limit=5)
    assert_complete(data, result)
    assert result["optimal"]
    assert result["objective"] == pytest.approx(2)


def test_zero_demand_cycle_cannot_hide_beside_a_positive_demand_route():
    data = instance(((0, 0), (5, 0), (0, 5), (0, 5)), demand=[0, 1, 0, 0])
    result = mip.solve_mip(data, time_limit=5)
    assert_complete(data, result)
    assert result["objective"] == pytest.approx(10 + np.sqrt(50))


@pytest.mark.parametrize("n_depots, colocated, with_energy, positive_service", [
    (1, False, False, False), (1, True, False, False),
    (2, False, False, False), (2, True, False, False),
    (1, False, True, False), (2, True, True, False),
    (1, True, False, True), (2, False, True, True),
])
def test_degenerate_tiny_models_match_independent_enumeration(n_depots, colocated, with_energy, positive_service):
    depots = [(0, 0), (4, 0)][:n_depots]
    coords = depots + ([(0, 0)] * 3 if colocated else [(1, 0), (1, 0), (3, 0)])
    em = EnergyModel(a=1, b=0.25, e_trip=0.5, battery=30, reserve=0.2) if with_energy else None
    data = instance(coords, n_depots=n_depots, energy=em,
                    service=[0] * n_depots + [0.25 if positive_service else 0] * 3,
                    vehicle_cost=0.5, w_dist=1, w_energy=1 if em else 0,
                    max_vehicles=1, max_duration=20)
    best, routes = brute_force(data)
    assert routes is not None
    result = mip.solve_mip(data, time_limit=10)
    assert_complete(data, result)
    assert result["optimal"]
    assert result["objective"] == pytest.approx(best, abs=1e-6)


def test_no_customers_is_an_empty_optimum_without_calling_highs(monkeypatch):
    data = instance(((0, 0),), capacity=0, max_vehicles=0)
    monkeypatch.setattr(mip, "milp", lambda *a, **kw: pytest.fail("empty model reached HiGHS"))
    result = mip.solve_mip(data)
    assert result["valid"] and result["optimal"]
    assert result["routes"] == []
    assert result["objective"] == result["bound"] == result["gap"] == 0


def test_zero_capacity_can_serve_zero_demand_visits():
    data = instance(capacity=0)
    result = mip.solve_mip(data, time_limit=5)
    assert_complete(data, result)
    assert result["objective"] == pytest.approx(2)


@pytest.mark.parametrize("data", [
    instance(capacity=0, demand=[0, 1, 0]),
    instance(max_vehicles=0),
    instance(((0, 0), (1, 0)), service=[3, 0], due=[100, 2]),
    instance(((0, 0), (1, 0)), service=[3, 0], max_duration=4),
])
def test_impossible_capacity_fleet_and_depot_service_have_no_incumbent(data):
    best, routes = brute_force(data)
    assert routes is None and best == float("inf")
    result = mip.solve_mip(data, time_limit=5)
    assert not result["optimal"]
    assert result["status"] == 2
    assert result["routes"] is None
    assert result["objective"] is None


def test_depot_service_is_included_in_a_feasible_duration_boundary():
    data = instance(((0, 0), (1, 0)), service=[3, 0], due=[100, 4], max_duration=5)
    result = mip.solve_mip(data, time_limit=5)
    assert_complete(data, result)
    assert result["optimal"]


def test_explicit_fleet_override_is_used_by_model_and_independent_audit():
    data = instance(max_vehicles=0)
    result = mip.solve_mip(data, time_limit=5, max_routes_per_depot=1)
    assert result["valid"] and result["optimal"]
    assert len(result["routes"]) == 1
    assert sorted(result["routes"][0][1:-1]) == [1, 2]
    assert data.max_vehicles == 0
    assert mip.solve_mip(instance(), max_routes_per_depot=0)["routes"] is None


def forge_solver(monkeypatch, change):
    """Corrupt a real solver result; auxiliary dimensions follow the real model."""
    real = mip.milp

    def forged(*args, **kwargs):
        original = real(*args, **kwargs)
        assert original.x is not None
        result = SimpleNamespace(**dict(original))
        result.x = original.x.copy()
        change(result, args, kwargs)
        return result

    monkeypatch.setattr(mip, "milp", forged)


def assert_invalid(result):
    assert result["optimal"] is False
    assert result["valid"] is False
    assert result["status"] == 4
    assert result["routes"] is None
    assert result["objective"] is None
    assert "invalid solver result" in result["message"]


@pytest.mark.parametrize("value", [0.5, -1, 2, float("nan"), float("inf")])
def test_fractional_out_of_bounds_and_nonfinite_arc_values_are_not_rounded_into_routes(monkeypatch, value):
    forge_solver(monkeypatch, lambda result, *_: result.x.__setitem__(0, value))
    assert_invalid(mip.solve_mip(instance(((0, 0), (1, 0)))))


@pytest.mark.parametrize("kind", ["short", "long", "matrix", "nan_auxiliary"])
def test_solver_vector_shape_and_auxiliary_finiteness_are_checked(monkeypatch, kind):
    def change(result, *_):
        if kind == "short":
            result.x = result.x[:-1]
        elif kind == "long":
            result.x = np.r_[result.x, 0]
        elif kind == "matrix":
            result.x = result.x[:, None]
        else:
            result.x[-1] = np.nan
    forge_solver(monkeypatch, change)
    assert_invalid(mip.solve_mip(instance(((0, 0), (1, 0)))))


def test_multidepot_integer_labels_are_also_audited(monkeypatch):
    def change(result, args, kwargs):
        last_integer = np.flatnonzero(kwargs["integrality"])[-1]
        result.x[last_integer] = 0.5
    forge_solver(monkeypatch, change)
    assert_invalid(mip.solve_mip(instance(((0, 0), (4, 0), (1, 0)), n_depots=2)))


def select_edges(result, data, chosen):
    # Fixtures below allow every non-depot-to-depot arc; no preprocessing exclusion.
    arcs = [(i, j) for i in range(data.n_nodes) for j in range(data.n_nodes)
            if i != j and not (i < data.n_depots and j < data.n_depots)]
    result.x[:len(arcs)] = 0
    for edge in chosen:
        result.x[arcs.index(edge)] = 1


@pytest.mark.parametrize("case", ["disconnected", "missing", "branch", "cross_depot"])
def test_every_selected_edge_must_decompose_into_complete_same_depot_routes(monkeypatch, case):
    if case == "cross_depot":
        data = instance(((0, 0), (4, 0), (1, 0), (3, 0)), n_depots=2)
        edges = [(0, 2), (2, 1), (1, 3), (3, 0)]
    else:
        data = instance()
        edges = {"disconnected": [(1, 2), (2, 1)],
                 "missing": [(0, 1), (1, 0)],
                 "branch": [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)]}[case]
    forge_solver(monkeypatch, lambda result, *_: select_edges(result, data, edges))
    assert_invalid(mip.solve_mip(data))


@pytest.mark.parametrize("case", ["capacity", "time_window"])
def test_independent_checker_rejects_a_connected_but_physically_infeasible_route(monkeypatch, case):
    if case == "capacity":
        data = instance(((0, 0), (1, 0), (2, 0), (3, 0)), demand=[0, 1, 1, 1], capacity=2)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    else:
        data = instance(((0, 0), (1, 0), (2, 0), (3, 0)), due=[100, 4, 4, 4])
        edges = [(0, 3), (3, 2), (2, 1), (1, 0)]
    forge_solver(monkeypatch, lambda result, *_: select_edges(result, data, edges))
    result = mip.solve_mip(data)
    assert_invalid(result)
    assert "feasibility" in result["message"]


@pytest.mark.parametrize("objective", [float("nan"), float("inf"), -1, 3])
def test_solver_objective_must_match_independent_route_cost(monkeypatch, objective):
    forge_solver(monkeypatch, lambda result, *_: setattr(result, "fun", objective))
    assert_invalid(mip.solve_mip(instance(((0, 0), (1, 0)))))


@pytest.mark.parametrize("status", [2, 3, 4])
def test_failed_solver_status_with_a_vector_is_not_published_as_an_incumbent(monkeypatch, status):
    forge_solver(monkeypatch, lambda result, *_: setattr(result, "status", status))
    assert_invalid(mip.solve_mip(instance(((0, 0), (1, 0)))))


@pytest.mark.parametrize("status", [0, 1])
def test_valid_nonzero_gap_incumbent_is_not_called_proven_optimal(monkeypatch, status):
    def change(result, *_):
        result.status = status
        result.mip_gap = 0.1
        result.mip_dual_bound = result.fun * 0.9
    forge_solver(monkeypatch, change)
    data = instance(((0, 0), (1, 0)))
    result = mip.solve_mip(data, mip_rel_gap=0.1)
    assert result["valid"]
    assert not result["optimal"]
    assert_complete(data, result)
    assert result["gap"] == 0.1


def test_default_requests_zero_mip_gap(monkeypatch):
    real = mip.milp
    def checked(*args, **kwargs):
        assert kwargs["options"]["mip_rel_gap"] == 0
        return real(*args, **kwargs)
    monkeypatch.setattr(mip, "milp", checked)
    assert mip.solve_mip(instance(((0, 0), (1, 0))))["optimal"]


@pytest.mark.parametrize("bound", [1, 3])
def test_contradictory_zero_gap_certificate_is_rejected(monkeypatch, bound):
    forge_solver(monkeypatch, lambda result, *_: setattr(result, "mip_dual_bound", bound))
    result = mip.solve_mip(instance(((0, 0), (1, 0))))
    assert_invalid(result)
    assert result["solver_status"] == 0


@pytest.mark.parametrize("missing", ["mip_gap", "mip_dual_bound"])
def test_missing_proof_metadata_does_not_invalidate_route_or_prove_optimality(monkeypatch, missing):
    forge_solver(monkeypatch, lambda result, *_: delattr(result, missing))
    data = instance(((0, 0), (1, 0)))
    result = mip.solve_mip(data)
    assert result["valid"]
    assert not result["optimal"]
    assert_complete(data, result)


def test_solver_boundary_roundoff_does_not_reject_an_otherwise_valid_route(monkeypatch):
    forge_solver(monkeypatch, lambda result, *_: result.x.__setitem__(0, 1 - 1e-8))
    data = instance(((0, 0), (1, 0)))
    result = mip.solve_mip(data)
    assert result["valid"] and result["optimal"]
    assert_complete(data, result)
