"""Brute-force enumeration of every routing plan for tiny instances (<= 6 customers).

Enumerates all set partitions of the customers into routes, every ordering of
each route and every depot for each route, and evaluates each plan with the
independent checker. Used only in tests, to certify the MIP and the ALNS.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import permutations

from .checker import evaluate


def _set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for k in range(len(part)):
            yield part[:k] + [[first] + part[k]] + part[k + 1:]
        yield [[first]] + part


def _routes_for_block(inst, block):
    """All feasible (cost-minimal per depot/order) single routes for one block of customers."""
    out = []
    free = replace(inst, max_vehicles=None)
    for d in range(inst.n_depots):
        for perm in permutations(block):
            r = [d, *perm, d]
            ev = evaluate(free, [r], allow_unassigned=True)
            if ev.feasible:
                out.append((ev.cost, r))
    return out


def brute_force(inst):
    """Return (best_cost, best_routes) over all feasible plans, or (inf, None)."""
    cust = list(inst.customers)
    if len(cust) > 7:
        raise ValueError("brute force is only for tiny instances")
    cache = {}
    best = (float("inf"), None)
    for part in _set_partitions(cust):
        choices = []
        ok = True
        for block in part:
            key = tuple(sorted(block))
            if key not in cache:
                cache[key] = _routes_for_block(inst, block)
            if not cache[key]:
                ok = False
                break
            choices.append(cache[key])
        if not ok:
            continue
        # per block, the best route per depot is enough except for the fleet limit per depot
        best = min(best, _best_combo(inst, choices), key=lambda t: t[0])
    return best


def _best_combo(inst, choices):
    best = (float("inf"), None)

    def rec(k, routes, cost):
        nonlocal best
        if cost >= best[0] - 1e-12:
            return
        if k == len(choices):
            ev = evaluate(inst, routes)
            if ev.feasible and ev.cost < best[0] - 1e-12:
                best = (ev.cost, [list(r) for r in routes])
            return
        # best route per depot for this block
        per_depot = {}
        for c, r in choices[k]:
            if r[0] not in per_depot or c < per_depot[r[0]][0]:
                per_depot[r[0]] = (c, r)
        for c, r in sorted(per_depot.values()):
            rec(k + 1, routes + [r], cost + c)

    rec(0, [], 0.0)
    return best
