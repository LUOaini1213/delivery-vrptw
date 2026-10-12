"""Solomon (1987) I1 sequential insertion heuristic — the greedy baseline.

Route-by-route: start a route with a seed customer, then repeatedly insert the
unrouted customer u maximising  c2 = lambda * c0(u) - c1(u), where
  c1 = alpha1 * c11 + alpha2 * c12,
  c11 = insertion cost delta (distance, or the instance objective when energy is used)
        (mu = 1 as in Solomon's experiments),
  c12 = push-forward of the successor's service start,
  c0  = cost of serving u alone from the depot (round trip).
Several (seed rule, lambda, alpha) combinations are tried; the best by
(routes, cost) is returned, as Solomon did. Feasibility (time windows, capacity,
battery) uses the same O(1) checks as the ALNS (routes.py).
"""
from __future__ import annotations

import numpy as np

from .routes import Route, Solution, insertion_eval

PARAMS = [(1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0),
          (1.0, 0.5, 0.5), (2.0, 0.5, 0.5)]  # (lambda, alpha1, alpha2)


def _round_trip(inst, d, u):
    em = inst.energy
    c = inst.w_dist * (inst.dist[d, u] + inst.dist[u, d])
    if em is not None:
        e = em.e_trip + inst.energy_dist[d, u] * (em.a + em.b * inst.demand[u]) + inst.energy_dist[u, d] * em.a
        c += inst.w_energy * e
    return c


def _single_feasible(inst, d, u):
    r = Route(inst, [d, u, d])
    return r.feasible(inst)


def solomon_i1(inst, lam=1.0, alpha1=1.0, alpha2=0.0, seed_rule="far", max_routes=None):
    """Build routes greedily, choosing a feasible depot with a free vehicle.

    Depot selection is reconsidered before each new route. This is still a
    sequential heuristic, not a globally optimal customer-to-depot assignment.
    """
    inst.validate()
    unrouted = set(inst.customers)
    depots = list(range(inst.n_depots))
    feasible_depots = {
        u: sorted((d for d in depots if _single_feasible(inst, d, u)),
                  key=lambda d: (_round_trip(inst, d, u), d))
        for u in unrouted
    }
    # Keep the insertion reference cost even after all new-vehicle slots for a
    # customer fill up: it may still fit into the route currently being built.
    c0 = {u: _round_trip(inst, choices[0], u)
          for u, choices in feasible_depots.items() if choices}
    routes = []
    unassigned = []
    per_depot = {d: 0 for d in depots}
    for u in list(unrouted):
        if not feasible_depots[u]:
            unrouted.discard(u)
            unassigned.append(u)
    while unrouted:
        if max_routes is not None and len(routes) >= max_routes:
            break
        rt = {}
        for u in unrouted:
            for d in feasible_depots[u]:
                if inst.max_vehicles is None or per_depot[d] < inst.max_vehicles:
                    rt[u] = d
                    break
        cand = list(rt)
        if not cand:
            break
        if seed_rule == "far":
            s = max(cand, key=lambda u: (inst.dist[rt[u], u], -u))
        else:
            s = min(cand, key=lambda u: (inst.due[u], u))
        d = rt[s]
        route = Route(inst, [d, s, d])
        per_depot[d] += 1
        unrouted.discard(s)
        while unrouted:
            U = np.array(sorted(unrouted))
            cost, ok, _, new_succ = insertion_eval(inst, route, U)
            push = new_succ - route.a[1:][:, None]
            c1 = np.where(ok, alpha1 * np.where(ok, cost, 0.0) + alpha2 * push, np.inf)
            pos = c1.argmin(axis=0)
            c1b = c1[pos, np.arange(len(U))]
            feas = np.isfinite(c1b)
            if not feas.any():
                break
            c2 = np.where(feas, lam * np.array([c0[u] for u in U]) - c1b, -np.inf)
            k = int(np.argmax(c2))  # ties -> lowest customer id (U is sorted)
            u, p = int(U[k]), int(pos[k])
            nodes = route.nodes[: p + 1] + [u] + route.nodes[p + 1:]
            route = Route(inst, nodes)
            unrouted.discard(u)
        routes.append(route)
    unassigned.extend(sorted(unrouted))
    return Solution(routes, unassigned)


def best_i1(inst, max_routes=None):
    """Try all parameter combinations and both seed rules; return the best solution."""
    best = None
    for seed_rule in ("far", "due"):
        for lam, a1, a2 in PARAMS:
            sol = solomon_i1(inst, lam, a1, a2, seed_rule, max_routes)
            key = (len(sol.unassigned), sol.n_routes() if inst.vehicle_cost == 0 and inst.w_energy == 0 else 0,
                   sol.cost())
            if best is None or key < best[0]:
                best = (key, sol)
    return best[1]
