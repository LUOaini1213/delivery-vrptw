"""Adaptive Large Neighbourhood Search for VRPTW (+ battery/payload constraints).

Follows Ropke & Pisinger (2006, Transportation Science 40(4)) in structure:
  destroy: random, worst, Shaw (related), route removal
  repair:  greedy, greedy+noise, regret-2, regret-3, and a "sequential" repair that
           inserts customers one by one in a random/demand/distance/deadline order at
           their best position (the recreate step of Christiaens & Vanden Berghe 2020,
           without blinks) - greedy/regret alone never open a route that only pays off
           once several customers share it
  adaptive roulette-wheel weights updated every `segment` iterations with
  scores (33, 9, 13) and reaction factor 0.1; simulated-annealing acceptance.
Hierarchical objective (vehicles, then distance) is handled with the usual
two-stage scheme: stage A repeatedly deletes a route and searches with one
route fewer (unassigned customers penalised) until it fails; stage B then
minimises cost with the fleet size fixed.

Repair only ever inserts at positions that pass the feasibility checks in
routes.py, and customers that cannot be inserted stay unassigned, so every
solution the search holds is feasible apart from possibly-unassigned
customers. Final answers are re-verified with checker.evaluate by callers.
Budgets are in iterations (deterministic for a given seed); wall time is
recorded by callers.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .routes import INF, Route, Solution, UData, batch_insertion_costs, empty_route, insertion_costs, removal_savings
from .checker import evaluate
from .instance import _integer, _number

DESTROY = ("random", "worst", "shaw", "route")
REPAIR = ("greedy", "greedy_noise", "regret2", "regret3", "sequential")


@dataclass
class Params:
    xi: float = 0.4            # max fraction of customers removed
    q_min: int = 4
    q_max_abs: int = 40
    p_worst: float = 3.0
    p_shaw: float = 6.0
    noise: float = 0.025       # noise amplitude relative to max distance/cost
    segment: int = 100
    reaction: float = 0.1
    scores: tuple = (33.0, 9.0, 13.0)
    t_start: float = 0.05      # a solution t_start worse is accepted w.p. 0.5 at the start
    t_end_ratio: float = 0.002
    route_min_frac: float = 0.7  # share of iterations used for route minimisation (hierarchical)
    penalty: float | None = None  # per unassigned customer in stage A (default: max arc cost)


@dataclass
class OpStats:
    w_d: np.ndarray = field(default_factory=lambda: np.ones(len(DESTROY)))
    w_r: np.ndarray = field(default_factory=lambda: np.ones(len(REPAIR)))
    s_d: np.ndarray = field(default_factory=lambda: np.zeros(len(DESTROY)))
    s_r: np.ndarray = field(default_factory=lambda: np.zeros(len(REPAIR)))
    n_d: np.ndarray = field(default_factory=lambda: np.zeros(len(DESTROY)))
    n_r: np.ndarray = field(default_factory=lambda: np.zeros(len(REPAIR)))
    used_d: np.ndarray = field(default_factory=lambda: np.zeros(len(DESTROY), dtype=int))
    used_r: np.ndarray = field(default_factory=lambda: np.zeros(len(REPAIR), dtype=int))


class Context:
    """Per-instance precomputation shared by the operators."""

    def __init__(self, inst, params: Params):
        self.inst = inst
        self.p = params
        n = inst.n_nodes
        C = np.arange(inst.n_depots, n)
        D = inst.dist
        maxd = D.max() if D.max() > 0 else 1.0
        span = max(inst.due.max() - inst.ready.min(), 1.0)
        maxq = max(inst.demand.max(), 1.0)
        # Shaw relatedness (lower = more related): distance, time windows, demand
        rel = (9.0 * D / maxd
               + 3.0 * (np.abs(inst.ready[:, None] - inst.ready[None, :])
                        + np.abs(inst.due[:, None] - inst.due[None, :])) / span
               + 2.0 * np.abs(inst.demand[:, None] - inst.demand[None, :]) / maxq)
        self.rel = rel
        self.customers = C
        self.empties = [empty_route(inst, d) for d in range(inst.n_depots)]
        em = inst.energy
        self.open_cost = inst.vehicle_cost + (inst.w_energy * em.e_trip if em is not None else 0.0)
        arc_cost = inst.w_dist * D
        if em is not None:
            arc_cost = arc_cost + inst.w_energy * inst.energy_dist * (em.a + em.b * inst.capacity)
        self.max_arc = float(arc_cost.max())
        self.noise_amp = params.noise * self.max_arc
        self.penalty = params.penalty if params.penalty is not None else max(self.max_arc, 1.0)


# ----------------------------------------------------------------------------- destroy

def _remove(ctx, sol: Solution, removed: set) -> None:
    new_routes = []
    for r in sol.routes:
        if any(c in removed for c in r.customers):
            nodes = [v for v in r.nodes if v not in removed or v < ctx.inst.n_depots]
            if len(nodes) > 2:
                new_routes.append(Route(ctx.inst, nodes))
        else:
            new_routes.append(r)
    sol.routes = new_routes
    sol.unassigned = sol.unassigned + sorted(removed)


def _assigned(sol):
    return [c for r in sol.routes for c in r.customers]


def destroy_random(ctx, sol, q, rng):
    cust = _assigned(sol)
    if not cust:
        return
    q = min(q, len(cust))
    pick = rng.choice(len(cust), size=q, replace=False)
    _remove(ctx, sol, {cust[i] for i in pick})


def destroy_worst(ctx, sol, q, rng):
    removed = set()
    q = min(q, len(_assigned(sol)))
    cache = {}
    while len(removed) < q:
        savs, custs = [], []
        for r in sol.routes:
            key = id(r)
            if key not in cache:
                cache[key] = (removal_savings(ctx.inst, r), r)  # keep r alive so id() stays unique
            savs.append(cache[key][0])
            custs.extend(r.customers)
        if not custs:
            break
        sv = np.concatenate(savs)
        cu = np.array(custs)
        order = np.lexsort((cu, -sv))
        k = int(math.floor(rng.random() ** ctx.p.p_worst * len(order)))
        c = int(cu[order[k]])
        removed.add(c)
        _remove(ctx, sol, {c})
        sol.unassigned = sol.unassigned[:-1]  # re-added once at the end
    sol.unassigned = sol.unassigned + sorted(removed)


def destroy_shaw(ctx, sol, q, rng):
    cust = _assigned(sol)
    if not cust:
        return
    q = min(q, len(cust))
    rest = np.array(sorted(cust))
    first = int(rest[int(rng.integers(len(rest)))])
    removed = [first]
    rest = rest[rest != first]
    while len(removed) < q and len(rest):
        r = removed[int(rng.integers(len(removed)))]
        order = np.lexsort((rest, ctx.rel[r, rest]))
        k = int(math.floor(rng.random() ** ctx.p.p_shaw * len(order)))
        c = int(rest[order[k]])
        removed.append(c)
        rest = rest[rest != c]
    _remove(ctx, sol, set(removed))


def destroy_route(ctx, sol, q, rng):
    if not sol.routes:
        return
    w = np.array([1.0 / len(r) for r in sol.routes])
    k = int(rng.choice(len(sol.routes), p=w / w.sum()))
    _remove(ctx, sol, set(sol.routes[k].customers))


DESTROY_FN = {"random": destroy_random, "worst": destroy_worst, "shaw": destroy_shaw, "route": destroy_route}


# ----------------------------------------------------------------------------- repair

def repair(ctx, sol: Solution, rng, k_regret: int = 1, noise: bool = False, max_routes: int | None = None):
    """Insert unassigned customers (greedy if k_regret==1, regret-k if >1, sequential if 0). In place."""
    inst = ctx.inst
    U = np.array(sorted(sol.unassigned), dtype=int)
    nU = len(U)
    if nU == 0:
        return
    routes = list(sol.routes)
    nD = inst.n_depots
    R0 = len(routes)
    ncol = R0 + nU + nD
    C = np.full((nU, ncol), INF)
    P = np.zeros((nU, ncol), dtype=int)

    def noisy(c):
        if not noise:
            return c
        eps = rng.uniform(-1.0, 1.0, size=c.shape) * ctx.noise_amp
        return np.where(np.isfinite(c), np.maximum(c + eps, 0.0), INF)

    ud = UData(inst, U)
    if R0:
        c, p = batch_insertion_costs(inst, routes, ud)
        C[:, :R0], P[:, :R0] = noisy(c), p
    per_depot = [sum(1 for r in routes if r.depot == d) for d in range(nD)]

    def can_open(d):
        if max_routes is not None and len(routes) >= max_routes:
            return False
        if inst.max_vehicles is not None and per_depot[d] >= inst.max_vehicles:
            return False
        return True

    empty_cost = []
    for d in range(nD):
        c, p = insertion_costs(inst, ctx.empties[d], ud, ctx.open_cost)
        empty_cost.append((noisy(c), p))

    def refresh_empty_cols():
        for d in range(nD):
            col = ncol - nD + d
            if can_open(d):
                C[:, col], P[:, col] = empty_cost[d]
            else:
                C[:, col] = INF

    refresh_empty_cols()
    col_route = list(range(R0)) + [None] * nU  # column -> index in routes
    next_col = R0
    active = np.ones(nU, dtype=bool)
    BIG = 1e12
    seq_rank = None
    failed = np.zeros(nU, dtype=bool)
    if k_regret == 0:
        mode = int(rng.integers(4))
        if mode == 0:
            key = rng.random(nU)
        elif mode == 1:
            key = -inst.demand[U]
        elif mode == 2:
            key = -inst.dist[np.ix_(U, np.arange(nD))].min(axis=1)
        else:
            key = inst.due[U]
        seq_rank = np.empty(nU)
        seq_rank[np.lexsort((U, key))] = np.arange(nU)
    while active.any():
        idx = np.flatnonzero(active)
        sub = C[idx]
        c1 = sub.min(axis=1)
        if k_regret == 0:
            pick = int(np.argmin(seq_rank[idx]))
        elif k_regret == 1:
            order = np.lexsort((U[idx], c1))
            pick = order[0]
        else:
            k = min(k_regret, sub.shape[1])
            part = np.sort(np.partition(sub, k - 1, axis=1)[:, :k], axis=1)
            fin = np.isfinite(part)
            with np.errstate(invalid="ignore"):
                diff = np.where(fin, part - part[:, :1], BIG)
            reg = diff[:, 1:].sum(axis=1)
            reg = np.where(np.isfinite(c1), reg, -INF)
            order = np.lexsort((U[idx], c1, -reg))
            pick = order[0]
        row = idx[pick]
        if not np.isfinite(c1[pick]):
            if k_regret != 0:
                break  # greedy/regret: nothing left is insertable
            active[row] = False  # sequential: skip this customer, keep going
            failed[row] = True
            continue
        u = int(U[row])
        col = int(np.argmin(C[row]))
        pos = int(P[row, col])
        active[row] = False
        if col >= ncol - nD:
            d = col - (ncol - nD)
            new = Route(inst, [d, u, d])
            routes.append(new)
            per_depot[d] += 1
            col_route[next_col] = len(routes) - 1
            tcol = next_col
            next_col += 1
            refresh_empty_cols()
        else:
            ri = col_route[col]
            old = routes[ri]
            new = Route(inst, old.nodes[: pos + 1] + [u] + old.nodes[pos + 1:])
            routes[ri] = new
            tcol = col
        rem = np.flatnonzero(active)
        if len(rem):
            c, p = insertion_costs(inst, new, U[rem] if len(rem) < nU else ud)
            C[rem, tcol], P[rem, tcol] = noisy(c), p
    sol.routes = routes
    sol.unassigned = [int(U[i]) for i in np.flatnonzero(active | failed)]


REPAIR_ARGS = {"greedy": (1, False), "greedy_noise": (1, True), "regret2": (2, False), "regret3": (3, False),
               "sequential": (0, False)}


# ----------------------------------------------------------------------------- search

def _roulette(w, rng):
    return int(rng.choice(len(w), p=w / w.sum()))


def _update_weights(st: OpStats, p: Params):
    for w, s, n in ((st.w_d, st.s_d, st.n_d), (st.w_r, st.s_r, st.n_r)):
        m = n > 0
        w[m] = (1 - p.reaction) * w[m] + p.reaction * s[m] / n[m]
        w[:] = np.maximum(w, 0.05)
        s[:] = 0
        n[:] = 0


def search(ctx, start: Solution, iters: int, rng, stats: OpStats, max_routes=None, penalty=0.0,
           stop_when_complete=False, deadline=None):
    """One SA-ALNS run. Returns (best_solution, iterations_used).

    'best' is lexicographic (fewest unassigned, then cost); acceptance uses
    cost + penalty*unassigned. With stop_when_complete the run
    stops at the first solution without unassigned customers.
    """
    p = ctx.p
    n = ctx.inst.n_customers
    q_hi = max(1, min(p.q_max_abs, int(math.ceil(p.xi * n))))
    q_lo = min(p.q_min, q_hi)
    cur = start.copy()
    cur_cost = cur.cost(penalty)
    best = cur.copy()
    best_key = (len(best.unassigned), best.cost(0.0))
    base = max(start.cost(0.0), 1e-9)
    T0 = p.t_start * base / math.log(2.0)
    cool = p.t_end_ratio ** (1.0 / max(iters, 1))
    T = T0
    it = 0
    for it in range(1, iters + 1):
        if deadline is not None and time.perf_counter() > deadline:
            it -= 1
            break
        di = _roulette(stats.w_d, rng)
        ri = _roulette(stats.w_r, rng)
        q = int(rng.integers(q_lo, q_hi + 1))
        cand = cur.copy()
        DESTROY_FN[DESTROY[di]](ctx, cand, q, rng)
        k, noise = REPAIR_ARGS[REPAIR[ri]]
        repair(ctx, cand, rng, k, noise, max_routes)
        c = cand.cost(penalty)
        key = (len(cand.unassigned), cand.cost(0.0))
        score = 0.0
        if key[0] < best_key[0] or (key[0] == best_key[0] and key[1] < best_key[1] - 1e-9):
            best, best_key = cand.copy(), key
            cur, cur_cost = cand, c
            score = p.scores[0]
        elif c < cur_cost - 1e-9:
            cur, cur_cost = cand, c
            score = p.scores[1]
        elif T > 0 and rng.random() < math.exp(-(c - cur_cost) / T):
            cur, cur_cost = cand, c
            score = p.scores[2]
        stats.s_d[di] += score
        stats.s_r[ri] += score
        stats.n_d[di] += 1
        stats.n_r[ri] += 1
        stats.used_d[di] += 1
        stats.used_r[ri] += 1
        if it % p.segment == 0:
            _update_weights(stats, p)
        T *= cool
        if stop_when_complete and not best.unassigned:
            break
    return best, it


def solve(inst, start: Solution, iters: int, seed: int, params: Params | None = None,
          hierarchical: bool = False, max_routes=None, time_limit: float | None = None):
    """ALNS entry point. Returns (solution, info dict)."""
    inst.validate()
    _integer(iters, "iters")
    if max_routes is not None:
        _integer(max_routes, "max_routes")
    if time_limit is not None:
        _number(time_limit, "time_limit", 0, positive=True)
    initial_routes = start.as_lists()
    checked = evaluate(inst, initial_routes, allow_unassigned=True)
    if not checked.feasible:
        raise ValueError("invalid starting solution: " + "; ".join(checked.violations))
    if max_routes is not None and len(initial_routes) > max_routes:
        raise ValueError("starting solution exceeds max_routes")
    for node in start.unassigned:
        _integer(node, "unassigned customer", inst.n_depots)
        if node >= inst.n_nodes:
            raise ValueError("unassigned customer exceeds node range")
    served = {node for route in initial_routes for node in route[1:-1]}
    if len(set(start.unassigned)) != len(start.unassigned) or set(start.unassigned) != set(inst.customers) - served:
        raise ValueError("unassigned customers must exactly cover missing customers, without duplicates")
    params = params or Params()
    ctx = Context(inst, params)
    rng = np.random.default_rng(seed)
    stats = OpStats()
    t0 = time.perf_counter()
    deadline = t0 + time_limit if time_limit else None
    # A caller may have changed the instance since creating its Route caches.
    sol = Solution([Route(inst, nodes) for nodes in initial_routes], start.unassigned)
    if sol.unassigned:
        repair(ctx, sol, rng, 2, False, max_routes)
    info = {"iters": iters, "seed": seed, "stage_a_iters": 0, "route_history": [sol.n_routes()]}
    used = 0
    if hierarchical:
        lb = (int(math.ceil(inst.demand.sum() / inst.capacity - 1e-9)) if inst.capacity > 0
              else (1 if inst.n_customers else 0))
        budget_a = int(iters * params.route_min_frac)
        feas = sol if not sol.unassigned else None
        while feas is not None and used < budget_a and feas.n_routes() > lb:
            target = feas.n_routes() - 1
            trial = feas.copy()
            sizes = np.array([len(r) for r in trial.routes])
            cands = np.flatnonzero(sizes == sizes.min())
            k = int(cands[int(rng.integers(len(cands)))])
            _remove(ctx, trial, set(trial.routes[k].customers))
            repair(ctx, trial, rng, 2, False, target)
            if not trial.unassigned:
                feas = trial
                info["route_history"].append(feas.n_routes())
                continue
            found, it = search(ctx, trial, budget_a - used, rng, stats, max_routes=target,
                               penalty=ctx.penalty, stop_when_complete=True, deadline=deadline)
            used += it
            if found.unassigned:
                break
            feas = found
            info["route_history"].append(feas.n_routes())
        info["stage_a_iters"] = used
        sol = feas if feas is not None else sol
        max_routes = sol.n_routes() if not sol.unassigned else max_routes
    best, it = search(ctx, sol, iters - used, rng, stats, max_routes=max_routes, penalty=ctx.penalty,
                      deadline=deadline)
    used += it
    info.update(iters_used=used, seconds=time.perf_counter() - t0,
                destroy_usage=dict(zip(DESTROY, stats.used_d.tolist())),
                repair_usage=dict(zip(REPAIR, stats.used_r.tolist())),
                destroy_weights=dict(zip(DESTROY, np.round(stats.w_d, 3).tolist())),
                repair_weights=dict(zip(REPAIR, np.round(stats.w_r, 3).tolist())))
    return best, info
