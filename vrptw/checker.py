"""Independent feasibility checker and objective evaluator.

Deliberately written as plain sequential loops over the raw instance data and
shares NO code with the search (routes.py / alns.py / construct.py / mip.py).
Every result that is reported is re-checked here.
"""
from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from numbers import Integral

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


def _route_nodes(inst, route):
    """Validate/materialize one tour before any indexing; empty depot tours are neutral."""
    if isinstance(route, (str, bytes, Mapping, Set)):
        raise ValueError("expected an ordered iterable of node IDs")
    try:
        nodes = list(route)
    except TypeError as exc:
        raise ValueError("expected an ordered iterable of node IDs") from exc
    for pos, node in enumerate(nodes):
        if isinstance(node, bool) or not isinstance(node, Integral):
            raise ValueError(f"node at position {pos} must be an integer ID, not {node!r}")
        if not 0 <= node < inst.n_nodes:
            raise ValueError(f"node {node} at position {pos} is outside 0..{inst.n_nodes - 1}")
    nodes = [int(node) for node in nodes]
    if not nodes:
        return nodes
    if nodes[0] >= inst.n_depots or nodes[-1] != nodes[0]:
        raise ValueError("must start and end at the same depot")
    if any(node < inst.n_depots for node in nodes[1:-1]):
        raise ValueError("depot inside route")
    if len(set(nodes[1:-1])) != len(nodes[1:-1]):
        raise ValueError("repeated customer inside route")
    return nodes


def route_schedule(inst, route):
    """Return service-start times (including depot return) with waiting allowed.

    Invalid tours raise ValueError. [] has no schedule; [d] and [d, d] stay at
    depot d's ready time and incur no travel/service. Helpers assume valid instance data.
    """
    route = _route_nodes(inst, route)
    if len(route) < 3:
        return [float(inst.ready[route[0]])] * len(route) if route else []
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
    Empty depot tours have duration zero; invalid tours raise ValueError.
    """
    route = _route_nodes(inst, route)
    if len(route) < 3:
        return 0.0
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
    """Trip energy for a valid tour; no trip charge for []/[d]/[d, d]."""
    route = _route_nodes(inst, route)
    em = inst.energy
    if em is None or len(route) < 3:
        return 0.0
    payload = sum(float(inst.demand[c]) for c in route[1:-1])
    e = em.e_trip
    for i, j in zip(route[:-1], route[1:]):
        e += float(inst.energy_dist[i][j]) * (em.a + em.b * payload)
        if j >= inst.n_depots:
            payload -= float(inst.demand[j])
    return e


def evaluate(inst, routes, allow_unassigned: bool = False) -> Evaluation:
    """Check route iterables once; malformed tours contribute no customer or cost data.

    Instance data errors raise ValueError. Route input errors are reported as
    violations. Structurally valid but infeasible tours retain their numeric cost.
    """
    inst.validate()
    try:
        if isinstance(routes, (str, bytes, Mapping)):
            raise TypeError
        route_iter = iter(routes)
    except TypeError:
        return Evaluation(False, 0, 0.0, 0.0, 0.0, ["routes must be an iterable of tours"])
    viol = []
    seen = {}
    dist = 0.0
    energy = 0.0
    per_depot = {}
    r_energy, r_end = [], []
    n = 0
    for k, raw in enumerate(route_iter):
        try:
            r = _route_nodes(inst, raw)
        except ValueError as exc:
            viol.append(f"route {k}: {exc}")
            continue
        if len(r) < 3:
            viol.append(f"route {k}: has no customer")
            continue
        d0 = r[0]
        n += 1
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
    cost = inst.vehicle_cost * n + inst.w_dist * dist + inst.w_energy * energy
    return Evaluation(not viol, n, dist, energy, cost, viol, r_energy, r_end)
