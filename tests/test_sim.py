"""Simulator: agrees with the checker when there is no uncertainty, is deterministic, and
trip chaining produces valid drone schedules."""
import numpy as np
import pytest

from vrptw.alns import solve
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.drone import DroneSpec, chain_trips, generate, trip_times, trip_window
from vrptw.sim import Scenario, bootstrap_ci, execute_plan, make_scenario, metrics, run_rolling, wind_factors

SPEC = DroneSpec()


@pytest.fixture(scope="module")
def planned():
    inst = generate(3, n=25, n_depots=2, spec=SPEC, horizon=180)
    sol, _ = solve(inst, best_i1(inst), iters=150, seed=0)
    routes = sol.as_lists()
    assert evaluate(inst, routes).feasible
    return inst, routes, chain_trips(inst, routes, SPEC.swap_min)


def calm(inst):
    n = inst.n_nodes
    return Scenario(0, np.zeros(10), np.zeros(10), np.ones((n, n)), np.ones(n), np.ones((10, n, n)))


def test_no_uncertainty_matches_checker(planned):
    inst, routes, drones = planned
    served, te = execute_plan(inst, calm(inst), routes, drones, SPEC.swap_min)
    m = metrics(inst, served, te, len(drones), 0.0)
    ev = evaluate(inst, routes)
    assert m["on_time"] == 1.0 and m["mean_late"] == 0.0
    assert m["reserve_viol"] == 0 and m["depleted"] == 0
    assert sum(te) == pytest.approx(ev.energy)
    assert sorted(te) == pytest.approx(sorted(ev.route_energy))
    assert set(served) == set(inst.customers)


def test_chaining_is_a_valid_schedule(planned):
    inst, routes, drones = planned
    seen = sorted(k for d in drones for k, _, _ in d["trips"])
    assert seen == list(range(len(routes)))
    for d in drones:
        prev_ret = -np.inf
        for k, dep, ret in d["trips"]:
            lo, hi = trip_window(inst, routes[k])
            assert routes[k][0] == d["depot"]
            assert dep >= prev_ret + SPEC.swap_min - 1e-9
            assert lo - 1e-9 <= dep <= hi + 1e-9
            assert ret == pytest.approx(trip_times(inst, routes[k], dep)[-1])
            prev_ret = ret
    assert len(drones) <= len(routes)


def test_trip_window_preferred_departure_is_shortest(planned):
    from vrptw.checker import min_duration
    inst, routes, _ = planned
    for r in routes:
        lo, hi = trip_window(inst, r)
        assert trip_times(inst, r, lo)[-1] - lo == pytest.approx(min_duration(inst, r))


def test_simulation_deterministic_and_seed_dependent(planned):
    inst, routes, drones = planned
    a = metrics(inst, *execute_plan(inst, make_scenario(inst, 5), routes, drones, 5.0), len(drones), 0.0)
    b = metrics(inst, *execute_plan(inst, make_scenario(inst, 5), routes, drones, 5.0), len(drones), 0.0)
    c = metrics(inst, *execute_plan(inst, make_scenario(inst, 6), routes, drones, 5.0), len(drones), 0.0)
    assert a == b
    assert a["energy"] != c["energy"]


def test_wind_factor_directions():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    f = wind_factors(coords, 5.0, 0.0, 15.0)  # wind blowing towards +x
    assert f[0, 1] == pytest.approx(15 / 20)          # tailwind
    assert f[1, 0] == pytest.approx(15 / 10)          # headwind
    assert f[0, 2] == pytest.approx(15 / np.sqrt(15 ** 2 - 25))  # crosswind
    assert np.allclose(wind_factors(coords, 0.0, 1.0, 15.0), 1.0)
    # a round trip in steady wind always costs more than in calm air
    assert f[0, 1] + f[1, 0] > 2.0


def test_bootstrap_ci():
    m, lo, hi = bootstrap_ci(np.full(50, 2.0))
    assert m == lo == hi == 2.0
    x = np.random.default_rng(0).normal(1.0, 1.0, 400)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and lo < 1.0 < hi


def test_rolling_serves_everyone_and_is_deterministic():
    inst = generate(4, n=15, n_depots=1, spec=SPEC, horizon=150)
    sc = make_scenario(inst, 1)
    a = run_rolling(inst, sc, SPEC.swap_min, iters=30, seed=0)
    b = run_rolling(inst, sc, SPEC.swap_min, iters=30, seed=0)
    assert set(a[0]) == set(inst.customers)
    assert a[1] == b[1] and a[2] == b[2]


def test_energy_model_matches_dandrea_formula():
    # P[kW] = m[kg] v[km/h] / (370 eta r) + p_av  ->  Wh per km = 1000 P / v
    sp = DroneSpec()
    em = sp.energy_model()
    for payload in (0.0, 1.0, 3.0):
        P = (sp.m_vehicle + payload) * sp.v_air_kmh / (370 * sp.eta * sp.lift_drag) + sp.p_avionics
        assert em.a + em.b * payload == pytest.approx(1000 * P / sp.v_air_kmh)
    assert em.usable == pytest.approx(sp.battery * (1 - sp.reserve))


def test_lateness_is_measured_exactly():
    """Depart so that the drone arrives 2.5 min after the window closes: lateness must be 2.5."""
    from vrptw.instance import make_instance
    em = SPEC.energy_model()
    inst = make_instance("late", [(0, 0), (0.9, 0)], [0, 1.0], [0, 0], [500, 10.0], [0, 2.0], 3.0,
                         speed=SPEC.speed_km_per_min, energy=em)  # 0.9 km at 0.9 km/min = 1 min
    drones = [{"depot": 0, "trips": [(0, 11.5, None)]}]
    served, _ = execute_plan(inst, calm(inst), [[0, 1, 0]], drones, SPEC.swap_min)
    assert served[1][1] == pytest.approx(2.5)
    m = metrics(inst, served, [10.0], 1, 0.0)
    assert m["on_time"] == 0.0 and m["mean_late"] == pytest.approx(2.5)
    drones = [{"depot": 0, "trips": [(0, 9.0, None)]}]
    served, _ = execute_plan(inst, calm(inst), [[0, 1, 0]], drones, SPEC.swap_min)
    assert served[1] == (pytest.approx(10.0), 0.0)
