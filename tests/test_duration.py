"""Maximum route duration with a delayable departure (used by the drone variant)."""
import dataclasses

import numpy as np
import pytest

from conftest import random_instance
from vrptw.alns import solve
from vrptw.brute import brute_force
from vrptw.checker import evaluate, min_duration
from vrptw.construct import best_i1
from vrptw.instance import make_instance
from vrptw.mip import solve_mip
from vrptw.routes import Route, insertion_eval


def test_min_duration_hand_case():
    # depot at 0; customer 1 at x=1 window [10, 20]; customer 2 at x=2 window [30, 40]; service 1
    inst = make_instance("dur", [(0, 0), (1, 0), (2, 0)], [0, 1, 1], [0, 10, 30], [100, 20, 40], [0, 1, 1], 10)
    # best: leave at 19, serve 1 at 20..21, reach 2 at 22, wait to 30, serve 30..31, back at 33 -> 14
    assert min_duration(inst, [0, 1, 2, 0]) == pytest.approx(14.0)
    assert min_duration(inst, [0, 2, 0]) == pytest.approx(5.0)
    assert Route(inst, [0, 1, 2, 0]).mindur == pytest.approx(14.0)
    lim = dataclasses.replace(inst, max_duration=10.0)
    assert not evaluate(lim, [[0, 1, 2, 0]]).feasible
    assert evaluate(lim, [[0, 1, 0], [0, 2, 0]]).feasible


def test_min_duration_latest_departure_binds():
    # customer 1 must be served by t=3 (due), customer 2 opens at 20: waiting cannot be absorbed
    inst = make_instance("dur2", [(0, 0), (1, 0), (2, 0)], [0, 1, 1], [0, 0, 20], [100, 3, 40], [0, 1, 1], 10)
    # latest departure 2 (arrive 1 at 3), then 2 at 5 -> wait to 20, serve, back at 23 -> 21
    assert min_duration(inst, [0, 1, 2, 0]) == pytest.approx(21.0)
    assert Route(inst, [0, 1, 2, 0]).mindur == pytest.approx(21.0)


@pytest.mark.parametrize("seed", range(10))
def test_route_mindur_and_insertion_match_checker(seed):
    inst = random_instance(600 + seed, n=8, n_depots=1 + seed % 2, energy=seed % 2 == 0)
    inst = dataclasses.replace(inst, max_duration=35.0)
    rng = np.random.default_rng(seed)
    cust = list(inst.customers)
    rng.shuffle(cust)
    checked = 0
    for block in (cust[:2], cust[2:3], []):
        nodes = [0, *block, 0]
        if block and not evaluate(inst, [nodes], allow_unassigned=True).feasible:
            continue
        r = Route(inst, nodes)
        if block:
            assert r.mindur == pytest.approx(min_duration(inst, nodes))
        U = np.array([c for c in cust if c not in block])
        _, ok, _, _ = insertion_eval(inst, r, U)
        for p in range(len(nodes) - 1):
            for k, u in enumerate(U):
                new = nodes[: p + 1] + [int(u)] + nodes[p + 1:]
                assert ok[p, k] == evaluate(inst, [new], allow_unassigned=True).feasible, (nodes, p, u)
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("seed", range(6))
def test_mip_equals_brute_force_with_max_duration(seed):
    inst = random_instance(700 + seed, n=6, n_depots=1 + seed % 2, energy=True)
    loose, _ = brute_force(inst)
    inst = dataclasses.replace(inst, max_duration=22.0)
    bc, br = brute_force(inst)
    assert br is not None
    m = solve_mip(inst, time_limit=60)
    assert m["optimal"]
    ev = evaluate(inst, m["routes"])
    assert ev.feasible, ev.violations
    assert ev.cost == pytest.approx(bc, rel=1e-6)
    assert bc >= loose - 1e-9


def test_duration_limit_binds_somewhere():
    changed = 0
    for seed in range(6):
        inst = random_instance(700 + seed, n=6, n_depots=1 + seed % 2, energy=True)
        a, _ = brute_force(inst)
        b, _ = brute_force(dataclasses.replace(inst, max_duration=22.0))
        changed += b > a + 1e-6
    assert changed >= 2


@pytest.mark.parametrize("seed", range(3))
def test_alns_respects_max_duration(seed):
    inst = dataclasses.replace(random_instance(800 + seed, n=25, n_depots=2, energy=True), max_duration=30.0)
    sol, _ = solve(inst, best_i1(inst), iters=100, seed=seed)
    ev = evaluate(inst, sol.as_lists(), allow_unassigned=True)
    assert ev.feasible, ev.violations
    assert not sol.unassigned
