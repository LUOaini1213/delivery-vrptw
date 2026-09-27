"""The MIP optimum must equal exhaustive enumeration on tiny instances."""
import pytest

from conftest import random_instance
from vrptw.brute import brute_force
from vrptw.checker import evaluate
from vrptw.instance import load_solomon
from vrptw.mip import solve_mip


def _compare(inst):
    bc, br = brute_force(inst)
    m = solve_mip(inst, time_limit=60)
    assert br is not None, "test instance must be feasible"
    assert m["optimal"], m["message"]
    ev = evaluate(inst, m["routes"])
    assert ev.feasible, ev.violations
    assert ev.cost == pytest.approx(bc, rel=1e-6, abs=1e-6)
    assert m["objective"] == pytest.approx(bc, rel=1e-6, abs=1e-6)
    return bc


@pytest.mark.parametrize("name", ["C101", "R101", "RC105", "R202", "C206", "RC201"])
def test_mip_equals_brute_force_solomon6(name):
    assert _compare(load_solomon(name, 6)) is not None


@pytest.mark.parametrize("seed", range(8))
def test_mip_equals_brute_force_energy(seed):
    inst = random_instance(100 + seed, n=6, n_depots=1 + seed % 2, energy=True)
    _compare(inst)


@pytest.mark.parametrize("seed", range(4))
def test_mip_equals_brute_force_fleet_limit_and_vehicle_cost(seed):
    inst = random_instance(200 + seed, n=6, n_depots=2, energy=seed % 2 == 0, tight=True, max_vehicles=2)
    inst.vehicle_cost = 30.0
    _compare(inst)


def test_brute_force_energy_binds():
    """Sanity: in the energy instances the battery actually changes the optimum."""
    import dataclasses
    changed = 0
    for seed in range(8):
        inst = random_instance(100 + seed, n=6, energy=True)
        loose = dataclasses.replace(inst, energy=dataclasses.replace(inst.energy, battery=1e6))
        b1, _ = brute_force(inst)
        b2, _ = brute_force(loose)
        assert b2 <= b1 + 1e-9
        assert b1 < float("inf")
        changed += b2 < b1 - 1e-6
    assert changed >= 3
