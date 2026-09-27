"""Exact 2-index MIP for (multi-depot) VRPTW with payload-dependent battery energy.

Solved with HiGHS through scipy.optimize.milp. Only meant for small instances
(up to ~25 customers); it gives provable optima (or a certified gap when the
time limit is hit) against which the heuristics are measured.

Variables
  x_ij  in {0,1}  arc i->j used (no depot->depot arcs)
  t_j   service start at customer j
  r_j   payload on board when arriving at j (includes q_j)
  g_j   battery energy used on arrival at j   (energy model only)
  f_j   energy of the trip if j is its last customer, else 0 (objective, energy model only)
  y_jd  customer j served from depot d        (only when there are >= 2 depots)
  tau_j departure time of j's route          (only with a max route duration)
Constraints (M's are tight big-M constants computed from the data)
  in/out degree 1 for customers; flow balance and fleet limit at each depot
  time:    t_j >= t_i + s_i + T_ij - M(1-x_ij);  depot start / return windows
  payload: r_i >= r_j + q_i - Q(1-x_ij), q_j <= r_j <= Q      (MTZ-style)
  energy:  g_j >= g_i + ED_ij (a + b r_j) - M(1-x_ij);  g_j >= e_trip + ED_dj (a + b r_j) - M(1-x_dj)
           g_j + ED_jd a <= usable + M(1-x_jd)
  duration: tau_i = tau_j on used arcs; t_j >= tau_j + T_dj; t_j + s_j + T_jd - tau_j <= D_max
The payload on arc i->j equals r_j, so ED_ij*b*r_j is linear once x_ij = 1
(it is a lower bound on r that the minimisation drives to the true payload).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class _Builder:
    def __init__(self):
        self.rows, self.cols, self.vals, self.lo, self.hi = [], [], [], [], []

    def add(self, coefs: dict, lo, hi):
        r = len(self.lo)
        for c, v in coefs.items():
            self.rows.append(r)
            self.cols.append(c)
            self.vals.append(v)
        self.lo.append(lo)
        self.hi.append(hi)

    def constraint(self, nvar):
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.lo), nvar)).tocsr()
        return LinearConstraint(A, np.array(self.lo), np.array(self.hi))


def solve_mip(inst, time_limit: float = 60.0, mip_rel_gap: float = 1e-6, max_routes_per_depot=None, verbose=False):
    """Returns dict(status, optimal, objective, bound, gap, routes, seconds)."""
    nD = inst.n_depots
    N = inst.n_nodes
    Cs = list(range(nD, N))
    Ds = list(range(nD))
    T, S, D = inst.time, inst.service, inst.dist
    e, l, q, Q = inst.ready, inst.due, inst.demand, inst.capacity
    em = inst.energy
    ED = inst.energy_dist
    K = max_routes_per_depot if max_routes_per_depot is not None else inst.max_vehicles
    if K is None:
        K = len(Cs)
    K = min(K, len(Cs))

    # --- arcs, with simple preprocessing (arcs that can never be feasible are dropped)
    arcs = []
    for i in range(N):
        for j in range(N):
            if i == j or (i < nD and j < nD):
                continue
            if i >= nD and j >= nD:
                if e[i] + S[i] + T[i, j] > l[j] + 1e-9 or q[i] + q[j] > Q + 1e-9:
                    continue
            if i < nD and j >= nD and e[i] + T[i, j] > l[j] + 1e-9:
                continue
            arcs.append((i, j))
    nx = len(arcs)
    idx = {}
    k = nx
    tv = {j: k + n for n, j in enumerate(Cs)}
    k += len(Cs)
    rv = {j: k + n for n, j in enumerate(Cs)}
    k += len(Cs)
    gv = fv = yv = None
    if em is not None:
        gv = {j: k + n for n, j in enumerate(Cs)}
        k += len(Cs)
        if inst.w_energy:
            fv = {j: k + n for n, j in enumerate(Cs)}
            k += len(Cs)
    tauv = None
    if inst.max_duration is not None:
        tauv = {j: k + n for n, j in enumerate(Cs)}
        k += len(Cs)
    if nD > 1:
        yv = {}
        for j in Cs:
            for d in Ds:
                yv[j, d] = k
                k += 1
    nvar = k
    for a_, (i, j) in enumerate(arcs):
        idx[i, j] = a_

    c = np.zeros(nvar)
    for (i, j), a_ in idx.items():
        c[a_] = inst.w_dist * D[i, j] + (inst.vehicle_cost if i < nD else 0.0)
    if fv is not None:
        for j in Cs:
            c[fv[j]] = inst.w_energy

    lb = np.zeros(nvar)
    ub = np.ones(nvar)
    integ = np.zeros(nvar)
    integ[:nx] = 1
    for j in Cs:
        lb[tv[j]], ub[tv[j]] = e[j], l[j]
        lb[rv[j]], ub[rv[j]] = q[j], Q
        if gv is not None:
            lb[gv[j]], ub[gv[j]] = 0.0, em.usable
        if fv is not None:
            lb[fv[j]], ub[fv[j]] = 0.0, np.inf
    if yv is not None:
        for key, v in yv.items():
            integ[v] = 1
    if tauv is not None:
        for j in Cs:
            lb[tauv[j]], ub[tauv[j]] = float(e[Ds].min()), float(l[j])
    Hz = float(l.max() - e.min() + T.max() + S.max())

    B = _Builder()
    outs = {v: [] for v in range(N)}
    ins = {v: [] for v in range(N)}
    for (i, j), a_ in idx.items():
        outs[i].append(a_)
        ins[j].append(a_)
    for j in Cs:
        B.add({a_: 1.0 for a_ in outs[j]}, 1, 1)
        B.add({a_: 1.0 for a_ in ins[j]}, 1, 1)
    for d in Ds:
        co = {a_: 1.0 for a_ in outs[d]}
        for a_ in ins[d]:
            co[a_] = co.get(a_, 0.0) - 1.0
        B.add(co, 0, 0)
        B.add({a_: 1.0 for a_ in outs[d]}, 0, K)
    # at least ceil(total demand / Q) routes overall (valid inequality)
    lbr = int(np.ceil(q[Cs].sum() / Q - 1e-9))
    B.add({a_: 1.0 for d in Ds for a_ in outs[d]}, lbr, np.inf)

    for (i, j), a_ in idx.items():
        if i >= nD and j >= nD:
            # time: t_j - t_i - M x_ij >= s_i + T_ij - M
            M = l[i] + S[i] + T[i, j] - e[j]
            if M > 0:  # (M <= 0: the constraint is implied by the time-window bounds)
                B.add({tv[j]: 1.0, tv[i]: -1.0, a_: -M}, S[i] + T[i, j] - M, np.inf)
            if tauv is not None:  # same route -> same departure time
                B.add({tauv[j]: 1.0, tauv[i]: -1.0, a_: Hz}, -np.inf, Hz)
                B.add({tauv[i]: 1.0, tauv[j]: -1.0, a_: Hz}, -np.inf, Hz)
            # payload: r_i - r_j - Q x_ij >= q_i - Q
            B.add({rv[i]: 1.0, rv[j]: -1.0, a_: -Q}, q[i] - Q, np.inf)
            if gv is not None:
                # g_j - g_i - ED_ij b r_j - M x_ij >= ED_ij a - M
                M = em.usable + ED[i, j] * (em.a + em.b * Q)
                B.add({gv[j]: 1.0, gv[i]: -1.0, rv[j]: -ED[i, j] * em.b, a_: -M}, ED[i, j] * em.a - M, np.inf)
        elif i < nD:
            d, jj = i, j
            M = e[d] + T[d, jj] - e[jj]
            if M > 0:
                B.add({tv[jj]: 1.0, a_: -M}, e[d] + T[d, jj] - M, np.inf)
            if gv is not None:
                M = em.e_trip + ED[d, jj] * (em.a + em.b * Q)
                B.add({gv[jj]: 1.0, rv[jj]: -ED[d, jj] * em.b, a_: -M}, em.e_trip + ED[d, jj] * em.a - M, np.inf)
            if yv is not None:
                B.add({yv[jj, d]: 1.0, a_: -1.0}, 0, np.inf)
            if tauv is not None:
                # t_j >= tau_j + T_dj and tau_j >= e_d when x_dj = 1
                B.add({tv[jj]: 1.0, tauv[jj]: -1.0, a_: -Hz}, T[d, jj] - Hz, np.inf)
                B.add({tauv[jj]: 1.0, a_: -Hz}, e[d] - Hz, np.inf)
        else:
            jj, d = i, j
            # t_j + s_j + T_jd <= l_d + M(1 - x)
            M = l[jj] + S[jj] + T[jj, d] - l[d]
            if M > 0:
                B.add({tv[jj]: 1.0, a_: M}, -np.inf, l[d] - S[jj] - T[jj, d] + M)
            if gv is not None:
                M = ED[jj, d] * em.a
                B.add({gv[jj]: 1.0, a_: M}, -np.inf, em.usable - ED[jj, d] * em.a + M)
                if fv is not None:
                    # f_j >= g_j + ED_jd a - M(1 - x_jd)
                    M2 = em.usable + ED[jj, d] * em.a
                    B.add({fv[jj]: 1.0, gv[jj]: -1.0, a_: -M2}, ED[jj, d] * em.a - M2, np.inf)
            if yv is not None:
                B.add({yv[jj, d]: 1.0, a_: -1.0}, 0, np.inf)
            if tauv is not None:
                # t_j + s_j + T_jd - tau_j <= max_duration + M(1 - x_jd)
                M = Hz + S[jj] + T[jj, d]
                B.add({tv[jj]: 1.0, tauv[jj]: -1.0, a_: M}, -np.inf,
                      inst.max_duration - S[jj] - T[jj, d] + M)
    if yv is not None:
        for j in Cs:
            B.add({yv[j, d]: 1.0 for d in Ds}, 1, 1)
        for (i, j), a_ in idx.items():
            if i >= nD and j >= nD:
                for d in Ds:
                    B.add({yv[j, d]: 1.0, yv[i, d]: -1.0, a_: 1.0}, -np.inf, 1)
                    B.add({yv[i, d]: 1.0, yv[j, d]: -1.0, a_: 1.0}, -np.inf, 1)

    cons = B.constraint(nvar)
    t0 = time.perf_counter()
    res = milp(c, constraints=[cons], integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": verbose})
    secs = time.perf_counter() - t0
    out = {"status": int(res.status), "message": res.message, "seconds": secs, "n_arcs": nx,
           "n_vars": nvar, "n_cons": len(B.lo)}
    if res.x is None:
        out.update(optimal=False, objective=None, bound=getattr(res, "mip_dual_bound", None), gap=None, routes=None)
        return out
    x = np.round(res.x[:nx]).astype(int)
    succ = {}
    for (i, j), a_ in idx.items():
        if x[a_]:
            succ.setdefault(i, []).append(j)
    routes = []
    for d in Ds:
        for first in succ.get(d, []):
            r = [d, first]
            while r[-1] >= nD:
                r.append(succ[r[-1]][0])
                if len(r) > N + 2:
                    raise RuntimeError("subtour in MIP solution")
            routes.append(r)
    bound = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    out.update(optimal=res.status == 0, objective=float(res.fun), bound=bound, gap=gap, routes=routes)
    return out
