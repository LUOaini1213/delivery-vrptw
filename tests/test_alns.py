"""ALNS never returns infeasible solutions, finds known optima on tiny cases, and is deterministic."""
import numpy as np
import pytest

from conftest import random_instance
from vrptw.alns import DESTROY, DESTROY_FN, REPAIR, REPAIR_ARGS, Context, Params, repair, solve
from vrptw.brute import brute_force
from vrptw.checker import evaluate
from vrptw.construct import best_i1, solomon_i1
from vrptw.instance import load_solomon
from vrptw.routes import Solution


def _feasible_partial(inst, sol):
    ev = evaluate(inst, sol.as_lists(), allow_unassigned=True)
    served = {c for r in sol.as_lists() for c in r[1:-1]}
    assert served.isdisjoint(sol.unassigned)
    assert served | set(sol.unassigned) == set(inst.customers)
    return ev.feasible, ev.violations


@pytest.mark.parametrize("seed", range(6))
def test_every_destroy_repair_pair_keeps_feasibility(seed):
    inst = random_instance(300 + seed, n=25, n_depots=1 + seed % 2, energy=seed % 3 != 2, max_vehicles=6)
    ctx = Context(inst, Params())
    rng = np.random.default_rng(seed)
    sol = solomon_i1(inst)
    ok, v = _feasible_partial(inst, sol)
    assert ok, v
    for it in range(40):
        d = DESTROY[it % len(DESTROY)]
        r = REPAIR[(it // len(DESTROY)) % len(REPAIR)]
        cand = sol.copy()
        DESTROY_FN[d](ctx, cand, int(rng.integers(2, 10)), rng)
        ok, v = _feasible_partial(inst, cand)
        assert ok, (d, v)
        k, noise = REPAIR_ARGS[r]
        repair(ctx, cand, rng, k, noise, max_routes=None)
        ok, v = _feasible_partial(inst, cand)
        assert ok, (d, r, v)
        per_depot = {}
        for rt in cand.routes:
            per_depot[rt.depot] = per_depot.get(rt.depot, 0) + 1
        assert max(per_depot.values()) <= 6
        sol = cand


@pytest.mark.parametrize("name", ["C101", "R105", "RC202"])
def test_alns_solomon_subset_feasible_and_not_worse_than_start(name):
    inst = load_solomon(name, 25)
    start = best_i1(inst)
    sol, info = solve(inst, start, iters=150, seed=1, hierarchical=True)
    ev = evaluate(inst, sol.as_lists())
    assert ev.feasible, ev.violations
    ev0 = evaluate(inst, start.as_lists())
    assert (ev.n_routes, ev.distance) <= (ev0.n_routes, ev0.distance + 1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_alns_energy_multidepot_feasible(seed):
    inst = random_instance(400 + seed, n=30, n_depots=2, energy=True)
    sol, _ = solve(inst, best_i1(inst), iters=120, seed=seed)
    ev = evaluate(inst, sol.as_lists(), allow_unassigned=True)
    assert ev.feasible, ev.violations
    assert not sol.unassigned


@pytest.mark.parametrize("seed", range(6))
def test_alns_finds_brute_force_optimum_on_tiny(seed):
    inst = random_instance(500 + seed, n=6, n_depots=1 + seed % 2, energy=seed % 2 == 0)
    best, _ = brute_force(inst)
    sol, _ = solve(inst, best_i1(inst), iters=200, seed=seed)
    ev = evaluate(inst, sol.as_lists())
    assert ev.feasible
    assert ev.cost == pytest.approx(best, rel=1e-9, abs=1e-9)


def test_alns_respects_fleet_limit_and_reports_unassigned():
    inst = random_instance(7, n=20, energy=True, max_vehicles=2)
    sol, _ = solve(inst, Solution([], list(inst.customers)), iters=60, seed=0)
    ev = evaluate(inst, sol.as_lists(), allow_unassigned=True)
    assert ev.feasible and ev.n_routes <= 2
    assert sol.unassigned  # 20 customers cannot fit in 2 battery-limited trips


def test_alns_deterministic():
    inst = load_solomon("R101", 25)
    s0 = best_i1(inst)
    a, _ = solve(inst, s0, iters=80, seed=3, hierarchical=True)
    b, _ = solve(inst, s0, iters=80, seed=3, hierarchical=True)
    assert a.as_lists() == b.as_lists()


def test_hierarchical_reduces_vehicles_from_singleton_routes():
    """Stage A must cut the fleet: start from one route per customer."""
    from vrptw.routes import Route
    inst = load_solomon("R105", 25)
    start = Solution([Route(inst, [0, c, 0]) for c in inst.customers])
    i1 = best_i1(inst)
    sol, info = solve(inst, start, iters=300, seed=2, hierarchical=True)
    assert evaluate(inst, sol.as_lists()).feasible
    assert sol.n_routes() <= i1.n_routes() + 1   # 25 -> close to the I1 fleet within 300 iterations
    hist = info["route_history"]
    assert all(b < a for a, b in zip(hist, hist[1:]))
    assert info["route_history"][0] == 25 and info["route_history"][-1] == sol.n_routes()


def test_hierarchical_prefers_fewer_vehicles_than_distance_objective():
    inst = load_solomon("R201", 25)
    s0 = best_i1(inst)
    h, _ = solve(inst, s0, iters=300, seed=1, hierarchical=True)
    d, _ = solve(inst, s0, iters=300, seed=1, hierarchical=False)
    assert h.n_routes() < d.n_routes()
    assert d.distance() < h.distance()


def test_regret_repair_beats_greedy_on_constructed_case():
    """u (near the depot) is equally cheap in both routes; v is only cheap in route A, whose capacity
    allows one more parcel. Greedy inserts u into A first and pushes v into B (cost ~15); regret-2
    inserts v first (largest regret) and gets ~3.6."""
    from vrptw.instance import make_instance
    from vrptw.routes import Route
    coords = [(0, 0), (10, 0), (0, 10), (0.5, 0.5), (11, 2)]  # depot, a, b, u, v
    inst = make_instance("regret", coords, [0, 1, 1, 1, 1], [0] * 5, [1000] * 5, [0] * 5, capacity=2)
    ctx = Context(inst, Params())
    base = [Route(inst, [0, 1, 0]), Route(inst, [0, 2, 0])]
    out = {}
    for k in (1, 2):
        sol = Solution(list(base), [3, 4])
        repair(ctx, sol, np.random.default_rng(0), k, False, max_routes=2)
        ev = evaluate(inst, sol.as_lists())
        assert ev.feasible
        out[k] = (ev.cost, sol.as_lists())
    base_cost = evaluate(inst, [[0, 1, 0], [0, 2, 0]]).cost
    assert out[2][0] - base_cost == pytest.approx(3.64, abs=0.01)
    assert out[1][0] - base_cost > 14.0
    assert any(4 in r and 1 in r for r in out[2][1])  # v ends up in route A under regret-2
