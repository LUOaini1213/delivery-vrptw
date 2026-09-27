"""Independent feasibility checker and objective evaluator.

Deliberately written as plain sequential loops over the raw instance data and
shares NO code with the search (routes.py / alns.py / construct.py / mip.py).
Every result that is reported is re-checked here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EPS = 1e-6


@dataclass
class Evaluation:
    feasible: bool
    n_routes: int
    distance: float
    energy: float
    cost: float
    violations: list = field(default_factory=list)
    route_energy: list = field(default_factory=list)
    route_end: list = field(default_factory=list)


def route_schedule(inst, route):
    """Return (service start times, return time) following the route with waiting allowed."""
    t = float(inst.ready[route[0]])
    starts = [t]
    prev = route[0]
    for node in route[1:]:
        arr = t + float(inst.service[prev]) + float(inst.time[prev][node])
        t = max(arr, float(inst.ready[node]))
        starts.append(t)
        prev = node
    return starts


def _schedule_from(inst, route, dep):
    t = dep
    prev = route[0]
    for node in route[1:]:
        t = max(t + float(inst.service[prev]) + float(inst.time[prev][node]), float(inst.ready[node]))
        if t > float(inst.due[node]) + EPS:
            return None
        prev = node
    return t


def min_duration(inst, route):
    """Shortest (return - departure) over all feasible departure times, or None if infeasible.

    Evaluated by simulation at every candidate departure where some node is
    reached exactly at its ready time or at its due time with no waiting before
    it (one of these is optimal), plus the earliest departure.
    """
    d = route[0]
    e0 = float(inst.ready[d])
    cands = {e0}
    pt = 0.0
    prev = d
    for node in route[1:]:
        pt += float(inst.service[prev]) + float(inst.time[prev][node])
        cands.add(max(e0, float(inst.ready[node]) - pt))
        cands.add(max(e0, float(inst.due[node]) - pt))
        prev = node
    best = None
    for dep in cands:
        end = _schedule_from(inst, route, dep)
        if end is not None and (best is None or end - dep < best):
            best = end - dep
    return best


def route_energy(inst, route):
    em = inst.energy
    if em is None:
        return 0.0
    payload = sum(float(inst.demand[c]) for c in route[1:-1])
    e = em.e_trip
    for i, j in zip(route[:-1], route[1:]):
        e += float(inst.energy_dist[i][j]) * (em.a + em.b * payload)
        if j >= inst.n_depots:
            payload -= float(inst.demand[j])
    return e


def evaluate(inst, routes, allow_unassigned: bool = False) -> Evaluation:
    viol = []
    seen = {}
    dist = 0.0
    energy = 0.0
    per_depot = {}
    r_energy, r_end = [], []
    for k, r in enumerate(routes):
        if len(r) < 3:
            viol.append(f"route {k}: has no customer")
            continue
        d0 = r[0]
        if d0 >= inst.n_depots or r[-1] != d0:
            viol.append(f"route {k}: must start and end at the same depot")
        if any(v < inst.n_depots for v in r[1:-1]):
            viol.append(f"route {k}: depot inside route")
        per_depot[d0] = per_depot.get(d0, 0) + 1
        for c in r[1:-1]:
            if c in seen:
                viol.append(f"customer {c} visited twice (routes {seen[c]} and {k})")
            seen[c] = k
        load = sum(float(inst.demand[c]) for c in r[1:-1])
        if load > inst.capacity + EPS:
            viol.append(f"route {k}: load {load} > capacity {inst.capacity}")
        starts = route_schedule(inst, r)
        for pos, node in enumerate(r):
            if starts[pos] > float(inst.due[node]) + EPS:
                viol.append(f"route {k}: node {node} starts at {starts[pos]:.4f} > due {inst.due[node]}")
        for i, j in zip(r[:-1], r[1:]):
            dist += float(inst.dist[i][j])
        if getattr(inst, "max_duration", None) is not None:
            md = min_duration(inst, r)
            if md is not None and md > inst.max_duration + EPS:
                viol.append(f"route {k}: duration {md:.3f} > max {inst.max_duration}")
        e = route_energy(inst, r)
        energy += e
        r_energy.append(e)
        r_end.append(starts[-1])
        if inst.energy is not None and e > inst.energy.usable + EPS:
            viol.append(f"route {k}: energy {e:.3f} > usable {inst.energy.usable:.3f}")
    missing = [c for c in inst.customers if c not in seen]
    if missing and not allow_unassigned:
        viol.append(f"{len(missing)} customers unserved: {missing[:10]}")
    if inst.max_vehicles is not None:
        for d0, cnt in per_depot.items():
            if cnt > inst.max_vehicles:
                viol.append(f"depot {d0}: {cnt} routes > max {inst.max_vehicles}")
    n = sum(1 for r in routes if len(r) >= 3)
    cost = inst.vehicle_cost * n + inst.w_dist * dist + inst.w_energy * energy
    return Evaluation(not viol, n, dist, energy, cost, viol, r_energy, r_end)
