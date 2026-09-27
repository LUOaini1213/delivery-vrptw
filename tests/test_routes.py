"""Search-side O(1) insertion checks must agree with the independent checker."""
import numpy as np
import pytest

from conftest import random_instance
from vrptw.checker import evaluate
from vrptw.instance import load_solomon, make_instance
from vrptw.routes import Route, UData, batch_insertion_costs, insertion_costs, insertion_eval, removal_savings


def test_time_window_propagation_hand_case():
    inst = make_instance("line", [(0, 0), (1, 0), (2, 0), (3, 0)], [0, 1, 1, 1], [0, 0, 5, 0],
                         [100, 10, 10, 20], [0, 1, 1, 1], 10)
    r = Route(inst, [0, 1, 2, 3, 0])
    assert r.a.tolist() == [0, 1, 5, 7, 11]
    # latest starts: depot 100; 3: min(20, 100-1-3)=20; 2: min(10, 20-1-1)=10; 1: min(10, 10-1-1)=8; 0: 8-0-1=7
    assert r.z.tolist() == [7, 8, 10, 20, 100]


def _check_all_insertions(inst, routes_nodes, rng):
    routes = [Route(inst, n) for n in routes_nodes]
    served = {c for n in routes_nodes for c in n[1:-1]}
    U = np.array([c for c in inst.customers if c not in served])
    if len(U) == 0:
        return 0
    n_checked = 0
    for r in routes:
        cost, ok, _, _ = insertion_eval(inst, r, U)
        base = evaluate(inst, [r.nodes], allow_unassigned=True).cost if len(r) else 0.0
        # insertion into an empty route excludes the fixed opening cost (added by the caller)
        open_cost = 0.0 if len(r) else inst.vehicle_cost + (inst.w_energy * inst.energy.e_trip if inst.energy else 0.0)
        for p in range(len(r.nodes) - 1):
            for k, u in enumerate(U):
                new = r.nodes[: p + 1] + [int(u)] + r.nodes[p + 1:]
                ev = evaluate(inst, [new], allow_unassigned=True)
                assert ok[p, k] == ev.feasible, (r.nodes, p, u, ev.violations)
                if ok[p, k]:
                    assert cost[p, k] == pytest.approx(ev.cost - base - open_cost, abs=1e-7)
                n_checked += 1
    return n_checked


@pytest.mark.parametrize("seed", range(12))
def test_insertion_matches_checker_random(seed):
    rng = np.random.default_rng(seed)
    inst = random_instance(seed, n=9, n_depots=1 + seed % 2, energy=seed % 3 != 0)
    cust = list(inst.customers)
    rng.shuffle(cust)
    routes = []
    for block in (cust[:3], cust[3:5]):
        d = int(rng.integers(inst.n_depots))
        # keep only routes that are feasible themselves (the precondition of the O(1) test)
        nodes = [d, *block, d]
        if evaluate(inst, [nodes], allow_unassigned=True).feasible:
            routes.append(nodes)
    routes.append([0, 0])  # the empty route used to open a new vehicle
    assert _check_all_insertions(inst, routes, rng) > 0


def test_insertion_matches_checker_solomon_subset():
    from vrptw.construct import solomon_i1
    inst = load_solomon("RC101", 25)
    sol = solomon_i1(inst)
    drop = {3, 8, 12, 17, 21, 24}
    routes = [[v for v in r.nodes if v not in drop] for r in sol.routes]
    routes = [r for r in routes if len(r) > 2] + [[0, 0]]
    for r in routes[:-1]:
        assert evaluate(inst, [r], allow_unassigned=True).feasible
    assert _check_all_insertions(inst, routes, np.random.default_rng(0)) > 100


def test_batch_equals_per_route():
    inst = random_instance(3, n=12, n_depots=2, energy=True)
    routes = [Route(inst, [0, 2, 3, 0]), Route(inst, [1, 5, 1]), Route(inst, [0, 0])]
    U = np.array([4, 6, 7, 8, 9, 10, 11, 12, 13])
    C, P = batch_insertion_costs(inst, routes, UData(inst, U))
    for k, r in enumerate(routes):
        c, p = insertion_costs(inst, r, U)
        assert np.array_equal(np.isfinite(c), np.isfinite(C[:, k]))
        f = np.isfinite(c)
        assert np.allclose(c[f], C[f, k]) and np.array_equal(p[f], P[f, k])


@pytest.mark.parametrize("seed", range(6))
def test_removal_savings_match_checker(seed):
    inst = random_instance(seed, n=6, energy=seed % 2 == 0, tight=False)
    inst.vehicle_cost = 7.0
    r = Route(inst, [0, 1, 2, 3, 4, 0])
    sav = removal_savings(inst, r)
    full = evaluate(inst, [r.nodes], allow_unassigned=True).cost
    for k, c in enumerate(r.customers):
        rest = [v for v in r.nodes if v != c]
        ev = evaluate(inst, [rest], allow_unassigned=True).cost
        assert sav[k] == pytest.approx(full - ev)
    single = Route(inst, [0, 1, 0])
    assert removal_savings(inst, single)[0] == pytest.approx(evaluate(inst, [[0, 1, 0]], allow_unassigned=True).cost)


def test_route_cache_matches_checker():
    inst = random_instance(5, n=8, energy=True, tight=False)
    r = Route(inst, [0, 3, 1, 4, 0])
    ev = evaluate(inst, [r.nodes], allow_unassigned=True)
    assert r.dist == pytest.approx(ev.distance)
    assert r.energy == pytest.approx(ev.energy)
    assert r.cost == pytest.approx(ev.cost)
    assert r.feasible(inst) == ev.feasible
