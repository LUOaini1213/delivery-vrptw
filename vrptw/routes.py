"""Search-side route representation with O(1)-per-position insertion checks.

For a route v_0 .. v_m (v_0 = v_m = depot) we cache
  a[k]    earliest service start at v_k (forward pass, waiting allowed)
  z[k]    latest service start at v_k that keeps the rest of the route feasible (backward pass)
  pay[k]  payload carried on arc (v_k -> v_k+1)
  ecum[k] energy-distance travelled from the depot up to v_k
so inserting customer u between v_p and v_p+1 is feasible iff
  max(e_u, a[p] + s[v_p] + t[v_p,u]) <= l_u,
  max(e_u, ...) + s_u + t[u,v_p+1] <= z[p+1],
  load + q_u <= Q,
  energy + dE <= usable,  dE = b*q_u*ecum[p] + ed[v_p,u](a+b(pay[p]+q_u)) + ed[u,v_p+1](a+b pay[p]) - ed[v_p,v_p+1](a+b pay[p])
The checks are vectorised over many customers and all positions with numpy.
This module is NOT used by checker.py; tests compare the two.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9
INF = float("inf")


class Route:
    __slots__ = ("nodes", "depot", "load", "dist", "energy", "a", "z", "pay", "ecum", "cost",
                 "ii", "jj", "dep", "zn", "dij", "eab", "ec", "edij", "ebase", "waitcum", "mindur")

    def __init__(self, inst, nodes):
        self.nodes = list(nodes)
        self.depot = self.nodes[0]
        self._refresh(inst)

    def _refresh(self, inst):
        v = self.nodes
        m = len(v)
        L = _lists(inst)
        q, S, T, D, E, Lt = L["q"], L["s"], L["t"], L["d"], L["e"], L["l"]
        self.load = float(sum(q[c] for c in v[1:-1]))
        a = [0.0] * m
        wc = [0.0] * m  # cumulative waiting at nodes 1..k
        t = E[v[0]]
        a[0] = t
        dist = 0.0
        w = 0.0
        for k in range(1, m):
            pk, vk = v[k - 1], v[k]
            t = t + S[pk] + T[pk][vk]
            if t < E[vk]:
                w += E[vk] - t
                t = E[vk]
            a[k] = t
            wc[k] = w
            dist += D[pk][vk]
        z = [0.0] * m
        zz = Lt[v[m - 1]]
        z[m - 1] = zz
        for k in range(m - 2, -1, -1):
            vk = v[k]
            zz = zz - S[vk] - T[vk][v[k + 1]]
            if Lt[vk] < zz:
                zz = Lt[vk]
            z[k] = zz
        self.a, self.z = np.array(a), np.array(z)
        self.waitcum = np.array(wc)
        # shortest duration: delay the departure by min(forward slack z0 - a0, total waiting)
        self.mindur = a[m - 1] - min(z[0], a[0] + w)
        self.dist = dist
        self.ii = np.array(v[:-1])
        self.jj = np.array(v[1:])
        self.dep = self.a[:-1] + inst.service[self.ii]
        self.zn = self.z[1:] + EPS
        self.dij = np.array([D[v[k]][v[k + 1]] for k in range(m - 1)])
        em = inst.energy
        if em is not None:
            ED = L["ed"]
            pay = [0.0] * (m - 1)
            ecum = [0.0] * m
            p = self.load
            e = em.e_trip
            for k in range(m - 1):
                pay[k] = p
                d = ED[v[k]][v[k + 1]]
                e += d * (em.a + em.b * p)
                ecum[k + 1] = ecum[k] + d
                if k + 1 < m - 1:
                    p -= q[v[k + 1]]
            self.pay, self.ecum, self.energy = np.array(pay), np.array(ecum), e
            self.eab = em.a + em.b * self.pay
            self.ec = self.ecum[:-1]
            self.edij = np.array([ED[v[k]][v[k + 1]] for k in range(m - 1)])
            self.ebase = e if m > 2 else em.e_trip
        else:
            self.pay = self.ecum = self.eab = self.ec = self.edij = None
            self.energy = 0.0
            self.ebase = 0.0
        n_c = m - 2
        self.cost = (inst.vehicle_cost if n_c else 0.0) + inst.w_dist * self.dist + inst.w_energy * (self.energy if n_c else 0.0)

    @property
    def customers(self):
        return self.nodes[1:-1]

    def __len__(self):
        return len(self.nodes) - 2

    def feasible(self, inst) -> bool:
        if self.load > inst.capacity + EPS:
            return False
        if np.any(self.a > inst.due[self.nodes] + EPS):
            return False
        if inst.energy is not None and len(self) and self.energy > inst.energy.usable + EPS:
            return False
        if inst.max_duration is not None and len(self) and self.mindur > inst.max_duration + EPS:
            return False
        return True


def empty_route(inst, depot):
    return Route(inst, [depot, depot])


def _lists(inst):
    """Python-list copies of the instance data for fast scalar access (cached on the instance)."""
    L = getattr(inst, "_lists_cache", None)
    if L is None or L["id"] is not inst.time or L["idd"] is not inst.dist or L["ide"] is not inst.energy_dist:
        L = {"id": inst.time, "idd": inst.dist, "ide": inst.energy_dist,
             "q": inst.demand.tolist(), "s": inst.service.tolist(), "e": inst.ready.tolist(),
             "l": inst.due.tolist(), "t": inst.time.tolist(), "d": inst.dist.tolist(),
             "ed": inst.energy_dist.tolist()}
        inst._lists_cache = L
    return L


class UData:
    """Customer-side arrays for a batch of candidate customers U (row vector layout)."""
    __slots__ = ("U", "Ur", "r", "l", "s", "q")

    def __init__(self, inst, U):
        U = np.asarray(U, dtype=int)
        self.U = U
        self.Ur = U[None, :]
        self.r = inst.ready[U]
        self.l = inst.due[U] + EPS
        self.s = inst.service[U]
        self.q = inst.demand[U]


class _Edges:
    """Concatenated edge arrays of several routes (for one batched evaluation)."""

    def __init__(self, routes, energy):
        self.ii = np.concatenate([r.ii for r in routes])
        self.jj = np.concatenate([r.jj for r in routes])
        self.dep = np.concatenate([r.dep for r in routes])
        self.zn = np.concatenate([r.zn for r in routes])
        self.dij = np.concatenate([r.dij for r in routes])
        lens = [len(r.ii) for r in routes]
        self.load = np.repeat([r.load for r in routes], lens)[:, None]
        if energy:
            self.eab = np.concatenate([r.eab for r in routes])
            self.ec = np.concatenate([r.ec for r in routes])
            self.edij = np.concatenate([r.edij for r in routes])
            self.ebase = np.repeat([r.ebase for r in routes], lens)[:, None]
        self.starts = np.cumsum([0] + lens[:-1])
        self.lens = lens


def _eval(inst, E, ud: UData, load, ebase, want_times=False):
    """cost[P,U] (inf where infeasible) of inserting each customer at each edge of E."""
    ii, jj, Ur = E.ii[:, None], E.jj[:, None], ud.Ur
    T = inst.time
    start = np.maximum(E.dep[:, None] + T[ii, Ur], ud.r)
    ok = start <= ud.l
    arr_j = start + ud.s + T[Ur, jj]
    ok &= arr_j <= E.zn[:, None]
    ok &= load + ud.q <= inst.capacity + EPS
    D = inst.dist
    cost = D[ii, Ur] + D[Ur, jj] - E.dij[:, None]
    if inst.w_dist != 1.0:
        cost *= inst.w_dist
    em = inst.energy
    if em is not None:
        ED = inst.energy_dist
        ab = E.eab[:, None]
        bq = em.b * ud.q
        dE = bq * E.ec[:, None] + ED[ii, Ur] * (ab + bq) + (ED[Ur, jj] - E.edij[:, None]) * ab
        ok &= ebase + dE <= em.usable + EPS
        cost += inst.w_energy * dE
    if inst.max_duration is not None:
        ok &= _duration_ok(inst, E, ud, start)
    cost[~ok] = INF
    if want_times:
        return cost, ok, start, arr_j
    return cost


def _duration_ok(inst, route, ud: UData, start):
    """Shortest route duration after each insertion <= max_duration. O(m^2) vector ops per route.

    Uses: min duration = end - min(latest departure, earliest departure + total waiting).
    """
    if not isinstance(route, Route):
        raise TypeError("max_duration needs per-route evaluation")
    v = route.nodes
    m = len(v)
    T, S, E, L = inst.time, inst.service, inst.ready, inst.due
    u = ud.U
    out = np.empty(start.shape, dtype=bool)
    for p in range(m - 1):
        su = start[p]
        arr_u = route.dep[p] + T[v[p], u]
        W = route.waitcum[p] + np.maximum(0.0, ud.r - arr_u)
        t, prev = su, u
        for k in range(p + 1, m):
            arr = t + (S[prev] if k > p + 1 else ud.s) + T[prev, v[k]]
            W = W + np.maximum(0.0, E[v[k]] - arr)
            t = np.maximum(arr, E[v[k]])
            prev = v[k]
        zu = np.minimum(L[u], route.z[p + 1] - ud.s - T[u, v[p + 1]])
        zz = np.minimum(L[v[p]], zu - S[v[p]] - T[v[p], u])
        for k in range(p - 1, -1, -1):
            zz = np.minimum(L[v[k]], zz - S[v[k]] - T[v[k], v[k + 1]])
        out[p] = t - np.minimum(zz, route.a[0] + W) <= inst.max_duration + EPS
    return out


def insertion_eval(inst, route: Route, U):
    """(cost[P,U], ok[P,U], start_u[P,U], arrival_at_successor[P,U]) for one route."""
    ud = U if isinstance(U, UData) else UData(inst, U)
    return _eval(inst, route, ud, route.load, route.ebase, want_times=True)


def insertion_costs(inst, route: Route, U, open_cost: float = 0.0):
    """Best insertion (cost delta, position) of each customer in U into `route`.

    Returns (best_cost[len(U)], best_pos[len(U)]); cost is +inf where infeasible.
    Position p means: insert between nodes[p] and nodes[p+1].
    `open_cost` is added when the route is currently empty (fixed cost of using it).
    """
    ud = U if isinstance(U, UData) else UData(inst, U)
    cost = _eval(inst, route, ud, route.load, route.ebase)
    pos = cost.argmin(axis=0)
    best = cost[pos, np.arange(len(ud.U))]
    if open_cost:
        best = best + open_cost
    return best, pos


def batch_insertion_costs(inst, routes, ud: UData):
    """insertion_costs for many routes with one vectorised evaluation. Returns (C[U,R], P[U,R])."""
    nU = len(ud.U)
    if not routes:
        return np.empty((nU, 0)), np.empty((nU, 0), dtype=int)
    if inst.max_duration is not None:
        cols = [insertion_costs(inst, r, ud) for r in routes]
        return np.stack([c for c, _ in cols], axis=1), np.stack([p for _, p in cols], axis=1)
    E = _Edges(routes, inst.energy is not None)
    cost = _eval(inst, E, ud, E.load, E.ebase if inst.energy is not None else 0.0)
    Cm = np.minimum.reduceat(cost, E.starts, axis=0)          # (R, U)
    P = np.empty((len(routes), nU), dtype=int)
    for k, (st, ln) in enumerate(zip(E.starts, E.lens)):
        P[k] = cost[st:st + ln].argmin(axis=0)
    return Cm.T.copy(), P.T.copy()


def removal_savings(inst, route: Route) -> np.ndarray:
    """Objective decrease from removing each customer (positions 1..m-2) from the route."""
    v = np.asarray(route.nodes)
    prev, cur, nxt = v[:-2], v[1:-1], v[2:]
    D = inst.dist
    sav = inst.w_dist * (D[prev, cur] + D[cur, nxt] - D[prev, nxt])
    em = inst.energy
    if em is not None:
        ED = inst.energy_dist
        k = np.arange(1, len(v) - 1)
        p_in = route.pay[k - 1]
        p_out = route.pay[k]
        qc = inst.demand[cur]
        dE = (em.b * qc * route.ecum[k - 1] + ED[prev, cur] * (em.a + em.b * p_in)
              + ED[cur, nxt] * (em.a + em.b * p_out) - ED[prev, nxt] * (em.a + em.b * p_out))
        sav = sav + inst.w_energy * dE
    if len(cur) == 1:
        sav = sav + inst.vehicle_cost + (inst.w_energy * em.e_trip if em is not None else 0.0)
    return sav


class Solution:
    """A set of routes plus unassigned customers. Routes are treated as immutable."""
    __slots__ = ("routes", "unassigned")

    def __init__(self, routes, unassigned=()):
        self.routes = [r for r in routes if len(r) > 0]
        self.unassigned = list(unassigned)

    def copy(self):
        return Solution(list(self.routes), list(self.unassigned))

    def n_routes(self):
        return len(self.routes)

    def routes_at(self, depot):
        return sum(1 for r in self.routes if r.depot == depot)

    def cost(self, penalty: float = 0.0):
        return sum(r.cost for r in self.routes) + penalty * len(self.unassigned)

    def distance(self):
        return sum(r.dist for r in self.routes)

    def as_lists(self):
        return [list(r.nodes) for r in self.routes]
