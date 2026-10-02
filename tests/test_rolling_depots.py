"""Rolling replanning keeps original depot identities and calendar/service semantics."""
import numpy as np
import pytest

from vrptw import drone, sim
from vrptw.checker import evaluate, route_schedule
from vrptw.construct import best_i1
from vrptw.instance import EnergyModel, make_instance


def make(coords, ready, due, *, n_depots=1, service=None):
    n = len(coords)
    return make_instance("rolling-depots", coords, [0] * n_depots + [1] * (n - n_depots),
                         ready, due, [0] * n if service is None else service,
                         capacity=1, n_depots=n_depots, energy=EnergyModel(1, 0, 0, 1000, 0))


def calm(inst):
    n = inst.n_nodes
    return sim.Scenario(0, np.zeros(10), np.zeros(10), np.ones((n, n)),
                        np.ones(n), np.ones((10, n, n)))


def test_sub_instance_filters_closed_depot_and_preserves_future_opening():
    inst = make([[0, 0], [100, 0], [101, 0]], [0, 90, 100], [10, 200, 150], n_depots=2)
    sub, idx = sim.sub_instance(inst, [2], 60, 2)
    assert idx.tolist() == [1, 2]
    assert sub.n_depots == 1
    assert sub.ready.tolist() == [90, 100]
    assert sub.due.tolist() == [200, 150]
    assert sub.time.tolist() == [[0, 2], [2, 0]]
    assert inst.ready.tolist() == [0, 90, 100]
    assert inst.due.tolist() == [10, 200, 150]


def test_depot_is_still_open_at_its_exact_closing_time():
    inst = make([[0, 0], [0, 0]], [0, 0], [10, 10])
    sub, idx = sim.sub_instance(inst, [1], 10, 1)
    assert idx.tolist() == [0, 1]
    assert sub.ready.tolist() == [10, 0]
    assert evaluate(sub, [[0, 1, 0]]).feasible


def test_closed_unused_depot_does_not_abort_other_depot_delivery():
    inst = make([[0, 0], [100, 0], [101, 0]], [0, 0, 100], [10, 200, 150], n_depots=2)
    assert best_i1(inst).as_lists() == [[1, 2, 1]]
    assert evaluate(inst, [[1, 2, 1]]).feasible
    served, energies, used, distance, epochs = sim.run_rolling(inst, calm(inst), 0, iters=0)
    assert served == {2: (100, 0)}
    assert energies == [2]
    assert (used, distance, epochs) == (1, 2, 2)


@pytest.mark.parametrize("return_factor, expected_calls, expected_used", [
    (1, [[], [(0, 2)], [(0, 101)]], 2),
    (150, [[], [(0, 151)], [(0, 151), (0, 101)]], 3),
])
def test_existing_drones_keep_original_depot_and_availability_after_filtering(
        monkeypatch, return_factor, expected_calls, expected_used):
    inst = make([[0, 0], [100, 0], [1, 0], [101, 0], [101, 0], [101, 0]],
                [0, 0, 0, 0, 100, 170], [10, 300, 5, 5, 110, 200], n_depots=2)
    calls, flown = [], []
    chain, execute = drone.chain_trips, sim.execute_trip

    def record_chain(inst, trips, swap, available=None, keep_all=False):
        calls.append(list(available))
        return chain(inst, trips, swap, available=available, keep_all=keep_all)

    def record_execute(inst, sc, nodes, dep):
        flown.append((list(nodes), dep))
        return execute(inst, sc, nodes, dep)

    monkeypatch.setattr(drone, "chain_trips", record_chain)
    monkeypatch.setattr(sim, "execute_trip", record_execute)
    sc = calm(inst)
    sc.arc_noise[3, 1] = return_factor
    # A delayed first return requires a new drone after depot 0 has closed;
    # that new drone must retain original depot 1 at the following epoch.
    served, energies, used, distance, epochs = sim.run_rolling(inst, sc, 0, iters=0)
    assert calls == expected_calls
    assert sorted(nodes for nodes, _ in flown) == [[0, 2, 0], [1, 3, 1], [1, 4, 1], [1, 5, 1]]
    assert served == {2: (1, 0), 3: (1, 0), 4: (100, 0), 5: (170, 0)}
    assert energies == [2, 1 + return_factor, 2, 2]
    assert (used, distance, epochs) == (expected_used, 8, 3)


def test_future_opening_depot_is_never_dispatched_early(monkeypatch):
    inst = make([[0, 0], [1, 0]], [100, 0], [200, 200])
    departures = []
    execute = sim.execute_trip

    def record(inst, sc, nodes, dep):
        departures.append(dep)
        return execute(inst, sc, nodes, dep)

    monkeypatch.setattr(sim, "execute_trip", record)
    served, _, used, _, epochs = sim.run_rolling(inst, calm(inst), 0, iters=0)
    assert departures == [100]
    assert served == {1: (101, 0)}
    assert (used, epochs) == (1, 2)


def test_all_depots_closed_has_explicit_operational_error():
    inst = make([[0, 0], [1, 0]], [-10, 0], [-1, 100])
    with pytest.raises(RuntimeError, match="(?i)all depots.*closed"):
        sim.run_rolling(inst, calm(inst), 0, iters=0)


def test_late_customer_still_executes_fallback_from_open_depot():
    inst = make([[0, 0], [1, 0]], [100, 0], [200, 10])
    served, energies, used, _, epochs = sim.run_rolling(inst, calm(inst), 0, iters=0)
    assert served == {1: (101, 91)}
    assert energies == [2]
    assert (used, epochs) == (1, 2)


@pytest.mark.parametrize("customer_noise, expected_return", [(1, 11), (2, 13)])
def test_departure_depot_service_is_deterministic_and_customer_service_is_noisy(customer_noise, expected_return):
    inst = make([[0, 0], [1, 0]], [4, 0], [100, 100], service=[3, 2])
    sc = calm(inst)
    sc.service_noise[:] = [9, customer_noise]
    end, energy, served = sim.execute_trip(inst, sc, [0, 1, 0], 4)
    # 4 departure + 3 depot service + 1 flight = customer arrival 8;
    # then 2 * customer_noise service + 1 return flight.
    assert end == expected_return
    assert energy == 2
    assert served == {1: (8, 0)}
    if customer_noise == 1:
        assert route_schedule(inst, [0, 1, 0]) == [4, 8, 11]
