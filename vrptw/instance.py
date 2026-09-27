"""Problem data shared by the classic VRPTW and the drone variant.

Node indexing: nodes 0..n_depots-1 are depots, nodes n_depots..N-1 are customers.
A route is a list of node ids that starts and ends at the same depot, e.g. [0, 5, 3, 0].

Distance/time conventions (Solomon): Euclidean distance in double precision,
travel time == distance (speed 1), no rounding. This matches the SINTEF
best-known-solution table (see data/bks/solomon100_bks.json).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class EnergyModel:
    """Battery energy of a trip = e_trip + sum over arcs of ed_ij * (a + b * payload_on_arc).

    a: energy per distance unit with zero payload (airframe + avionics), e.g. Wh/km
    b: extra energy per distance unit per unit of payload mass, e.g. Wh/(km*kg)
    e_trip: fixed energy per trip (take-off + landing)
    battery: battery capacity; reserve: fraction that must remain at landing.
    """
    a: float
    b: float
    e_trip: float
    battery: float
    reserve: float

    @property
    def usable(self) -> float:
        return self.battery * (1.0 - self.reserve)


@dataclass
class Instance:
    name: str
    n_depots: int
    coords: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray
    capacity: float
    dist: np.ndarray            # cost distance matrix (N x N)
    time: np.ndarray            # travel-time matrix (N x N)
    max_vehicles: int | None = None   # per depot; None = unlimited
    max_duration: float | None = None  # max route duration (return - departure), departure may be delayed
    energy: EnergyModel | None = None
    energy_dist: np.ndarray | None = None  # distance used by the energy model (defaults to dist)
    vehicle_cost: float = 0.0   # objective weight per route (trip)
    w_dist: float = 1.0         # objective weight per distance unit
    w_energy: float = 0.0       # objective weight per energy unit
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.energy_dist is None:
            self.energy_dist = self.dist

    @property
    def n_nodes(self) -> int:
        return len(self.demand)

    @property
    def customers(self) -> range:
        return range(self.n_depots, self.n_nodes)

    @property
    def n_customers(self) -> int:
        return self.n_nodes - self.n_depots

    def with_objective(self, vehicle_cost=None, w_dist=None, w_energy=None) -> "Instance":
        kw = {}
        if vehicle_cost is not None:
            kw["vehicle_cost"] = vehicle_cost
        if w_dist is not None:
            kw["w_dist"] = w_dist
        if w_energy is not None:
            kw["w_energy"] = w_energy
        return replace(self, **kw)


def euclid(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def parse_solomon(text: str, name: str | None = None) -> Instance:
    lines = [ln.strip() for ln in text.splitlines()]
    nm = name or lines[0].strip()
    k = next(i for i, ln in enumerate(lines) if ln.upper().startswith("NUMBER"))
    nveh, cap = (int(float(x)) for x in lines[k + 1].split()[:2])
    rows = []
    for ln in lines[k + 2:]:
        parts = ln.split()
        if len(parts) == 7 and all(p.replace(".", "", 1).lstrip("-").isdigit() for p in parts):
            rows.append([float(p) for p in parts])
    arr = np.array(rows)
    if not np.all(arr[:, 0] == np.arange(len(arr))):
        raise ValueError("customer ids are not 0..n in order")
    coords = arr[:, 1:3]
    d = euclid(coords)
    return Instance(name=nm, n_depots=1, coords=coords, demand=arr[:, 3], ready=arr[:, 4],
                    due=arr[:, 5], service=arr[:, 6], capacity=float(cap), dist=d, time=d.copy(),
                    max_vehicles=nveh, meta={"source": "Solomon (1987) via SINTEF TOP"})


def load_solomon(name: str, n_customers: int | None = None) -> Instance:
    inst = parse_solomon((DATA_DIR / "solomon" / f"{name.upper()}.txt").read_text(), name.upper())
    if n_customers is not None:
        inst = subset(inst, n_customers)
    return inst


def subset(inst: Instance, n: int) -> Instance:
    """First n customers (the convention used for the 25/50-customer Solomon sets)."""
    idx = np.arange(inst.n_depots + n)
    ed = inst.energy_dist[np.ix_(idx, idx)] if inst.energy_dist is not inst.dist else None
    return replace(inst, name=f"{inst.name}.{n}", coords=inst.coords[idx], demand=inst.demand[idx],
                   ready=inst.ready[idx], due=inst.due[idx], service=inst.service[idx],
                   dist=inst.dist[np.ix_(idx, idx)], time=inst.time[np.ix_(idx, idx)],
                   energy_dist=ed)


SOLOMON_NAMES = (
    [f"C1{i:02d}" for i in range(1, 10)] + [f"C2{i:02d}" for i in range(1, 9)]
    + [f"R1{i:02d}" for i in range(1, 13)] + [f"R2{i:02d}" for i in range(1, 12)]
    + [f"RC1{i:02d}" for i in range(1, 9)] + [f"RC2{i:02d}" for i in range(1, 9)]
)


def family(name: str) -> str:
    n = name.upper().split(".")[0]
    return n[:3] if n.startswith("RC") else n[:2]


def read_bks_solution(name: str) -> list[list[int]]:
    """Routes of the SINTEF detailed best-known solution, as [0, ..., 0] lists."""
    text = (DATA_DIR / "bks" / "solutions" / f"{name.upper()}.txt").read_text(errors="ignore")
    routes = []
    for ln in text.splitlines():
        if ln.strip().lower().startswith("route"):
            ids = [int(x) for x in ln.split(":", 1)[1].split()]
            if ids:
                routes.append([0] + ids + [0])
    return routes


def make_instance(name, coords, demand, ready, due, service, capacity, n_depots=1, speed=1.0,
                  energy: EnergyModel | None = None, max_vehicles=None, max_duration=None,
                  **objective) -> Instance:
    """Build an instance from raw arrays (Euclidean distances, travel time = distance / speed)."""
    coords = np.asarray(coords, dtype=float)
    d = euclid(coords)
    return Instance(name=name, n_depots=n_depots, coords=coords, demand=np.asarray(demand, float),
                    ready=np.asarray(ready, float), due=np.asarray(due, float),
                    service=np.asarray(service, float), capacity=float(capacity), dist=d,
                    time=d / speed, max_vehicles=max_vehicles, max_duration=max_duration, energy=energy,
                    **objective)
