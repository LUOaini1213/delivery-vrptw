"""Independent checker: hand-made cases with known answers, and the 56 published BKS solutions."""
import json

import numpy as np
import pytest

from vrptw.checker import evaluate, route_energy, route_schedule
from vrptw.instance import DATA_DIR, EnergyModel, SOLOMON_NAMES, load_solomon, make_instance, read_bks_solution


def line_instance(energy=None, capacity=10, **kw):
    # depot at x=0, customers at x=1,2,3 on a line; unit speed
    coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
    return make_instance("line", coords, demand=[0, 2, 3, 4], ready=[0, 0, 5, 0], due=[100, 10, 10, 20],
                         service=[0, 1, 1, 1], capacity=capacity, energy=energy, **kw)


def test_feasible_route_and_schedule():
    inst = line_instance()
    r = [0, 1, 2, 3, 0]
    # leave 0 at t=0, arrive 1 at t=1 (serve 1..2), arrive 2 at t=3 -> wait until 5, serve 5..6, arrive 3 at 7
    assert route_schedule(inst, r) == [0, 1, 5, 7, 11]
    ev = evaluate(inst, [r])
    assert ev.feasible and ev.n_routes == 1 and ev.distance == pytest.approx(6.0)


def test_time_window_violation_detected():
    inst = line_instance()
    ev = evaluate(inst, [[0, 3, 2, 1, 0]])  # 3 at t=3 ok, 2 at max(5,5)=5, 1 at 7 ok... then check
    assert ev.feasible
    inst2 = make_instance("tw", [(0, 0), (1, 0), (2, 0), (3, 0)], [0, 1, 1, 1], [0, 0, 0, 0],
                          [100, 3, 100, 2.999], [0, 1, 1, 1], 10)
    ev = evaluate(inst2, [[0, 1, 2, 3, 0]])  # 3 is reached at t=1+1+1+1+1=5 > 2.999
    assert not ev.feasible and any("node 3" in v for v in ev.violations)
    # exactly at the due date is allowed
    inst3 = make_instance("tw", [(0, 0), (1, 0)], [0, 1], [0, 0], [100, 1.0], [0, 1], 10)
    assert evaluate(inst3, [[0, 1, 0]]).feasible


def test_depot_return_deadline():
    inst = make_instance("dep", [(0, 0), (5, 0)], [0, 1], [0, 0], [10.9, 100], [0, 1], 10)
    assert not evaluate(inst, [[0, 1, 0]]).feasible  # back at 5+1+5 = 11 > 10.9


def test_capacity_violation():
    inst = line_instance(capacity=8)
    assert not evaluate(inst, [[0, 1, 2, 3, 0]]).feasible          # load 9
    assert evaluate(inst, [[0, 1, 2, 0], [0, 3, 0]]).feasible


def test_partition_errors():
    inst = line_instance()
    assert not evaluate(inst, [[0, 1, 2, 0]]).feasible             # 3 missing
    assert not evaluate(inst, [[0, 1, 2, 3, 0], [0, 1, 0]]).feasible  # 1 twice
    assert not evaluate(inst, [[0, 1, 0, 2, 3, 0]]).feasible        # depot inside
    assert not evaluate(inst, [[0, 1, 2, 3, 1]]).feasible           # does not end at depot


def test_fleet_limit():
    inst = line_instance(max_vehicles=2)
    assert not evaluate(inst, [[0, 1, 0], [0, 2, 0], [0, 3, 0]]).feasible
    assert evaluate(inst, [[0, 1, 2, 0], [0, 3, 0]]).feasible


def test_energy_hand_computed():
    em = EnergyModel(a=10.0, b=2.0, e_trip=5.0, battery=120.0, reserve=0.2)  # usable 96
    inst = line_instance(energy=em)
    # payload 9 on 0->1 (d=1), 7 on 1->2, 4 on 2->3, 0 on 3->0 (d=3)
    e = 5 + 1 * (10 + 18) + 1 * (10 + 14) + 1 * (10 + 8) + 3 * 10
    assert route_energy(inst, [0, 1, 2, 3, 0]) == pytest.approx(e)
    assert e == 105
    assert not evaluate(inst, [[0, 1, 2, 3, 0]]).feasible          # 105 > 96
    # splitting into two trips: 5+20+16+20 = 61 and 5+54+30 = 89, both <= 96
    ev = evaluate(inst, [[0, 1, 2, 0], [0, 3, 0]])
    assert ev.feasible and ev.route_energy == pytest.approx([61, 89])


def test_energy_increases_with_payload():
    em = EnergyModel(a=10.0, b=2.0, e_trip=0.0, battery=1e9, reserve=0.0)
    inst = line_instance(energy=em)
    heavy = make_instance("h", inst.coords, [0, 5, 5, 5], inst.ready, inst.due, inst.service, 99, energy=em)
    assert route_energy(heavy, [0, 1, 0]) > route_energy(inst, [0, 1, 0])
    # same arcs reversed: dropping the near parcels first carries less mass: 100 vs 128
    assert route_energy(inst, [0, 1, 2, 3, 0]) == pytest.approx(100)
    assert route_energy(inst, [0, 3, 2, 1, 0]) == pytest.approx(128)
    assert route_energy(inst, [0, 1, 0]) == pytest.approx(1 * (10 + 2 * 2) + 1 * 10)


def test_all_56_bks_solutions_reproduce_published_values():
    """Checker + distance convention reproduce every SINTEF BKS (vehicles, distance to 2 dp)."""
    bks = json.loads((DATA_DIR / "bks" / "solomon100_bks.json").read_text())["instances"]
    assert len(bks) == 56
    for name in SOLOMON_NAMES:
        inst = load_solomon(name)
        ev = evaluate(inst, read_bks_solution(name))
        assert ev.feasible, (name, ev.violations[:3])
        assert ev.n_routes == bks[name]["vehicles"], name
        assert abs(round(ev.distance, 2) - bks[name]["distance"]) < 0.006, name


def test_perturbed_bks_is_rejected():
    inst = load_solomon("R101")
    routes = read_bks_solution("R101")
    # moving the first customer of route 0 to the end of route 1 breaks a time window
    bad = [list(r) for r in routes]
    c = bad[0].pop(1)
    bad[1].insert(-1, c)
    assert not evaluate(inst, bad).feasible
