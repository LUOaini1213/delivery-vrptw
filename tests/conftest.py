import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrptw.instance import EnergyModel, make_instance  # noqa: E402


def random_instance(seed, n=6, n_depots=1, energy=False, tight=True, max_vehicles=None, w_energy=None):
    """Small random instance; with energy=True the battery/payload limits bind."""
    rng = np.random.default_rng(seed)
    N = n_depots + n
    coords = rng.uniform(0, 10, size=(N, 2))
    demand = np.r_[np.zeros(n_depots), rng.uniform(0.3, 2.0, n)]
    horizon = 120.0
    ready = np.zeros(N)
    due = np.full(N, horizon)
    service = np.r_[np.zeros(n_depots), np.full(n, 2.0)]
    if tight:
        c = rng.uniform(10, 90, n)
        w = rng.uniform(15, 40, n)
        ready[n_depots:] = np.maximum(0, c - w / 2)
        due[n_depots:] = c + w / 2
    em = None
    if energy:
        # battery sized so every single-customer trip fits with 10% spare, but longer trips may not
        a, b, e0 = 1.0, 0.25, 1.0
        d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
        single = max(min(e0 + d[k, j] * (a + b * demand[j]) + d[j, k] * a for k in range(n_depots))
                     for j in range(n_depots, N))
        em = EnergyModel(a=a, b=b, e_trip=e0, battery=1.1 * single / 0.8, reserve=0.2)
    obj = {}
    if energy:
        obj = dict(vehicle_cost=0.0, w_dist=0.0, w_energy=1.0 if w_energy is None else w_energy)
    return make_instance(f"rand{seed}", coords, demand, ready, due, service, capacity=3.0,
                         n_depots=n_depots, speed=1.0, energy=em, max_vehicles=max_vehicles, **obj)


@pytest.fixture
def rand_instance():
    return random_instance
