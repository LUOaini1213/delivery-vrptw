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
  h_ij  remaining customer visits carried on an arc into a customer (continuous)
Constraints (M's are tight big-M constants computed from the data)
  in/out degree 1 for customers; flow balance and fleet limit at each depot
  connectivity: 0 <= h_ij <= n_customers*x_ij; each customer consumes one visit
  time:    t_j >= t_i + s_i + T_ij - M(1-x_ij);  depot start / return windows
  payload: r_i >= r_j + q_i - Q(1-x_ij), q_j <= r_j <= Q      (MTZ-style)
  energy:  g_j >= g_i + ED_ij (a + b r_j) - M(1-x_ij);  g_j >= e_trip + ED_dj (a + b r_j) - M(1-x_dj)
           g_j + ED_jd a <= usable + M(1-x_jd)
  duration: tau_i = tau_j on used arcs; t_j >= tau_j + T_dj; t_j + s_j + T_jd - tau_j <= D_max
The payload on arc i->j equals r_j, so ED_ij*b*r_j is linear once x_ij = 1
(it is a lower bound on r that the minimisation drives to the true payload).
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from numbers import Integral, Real

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .checker import evaluate


# Solver-boundary tolerances only; they never turn arbitrary fractional arcs
# into routes. Independent route feasibility is checked separately below.
_INTEGER_TOL = 1e-6
_COST_TOL = 1e-6


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


def solve_mip(inst, time_limit: float = 60.0, mip_rel_gap: float = 0.0, max_routes_per_depot=None, verbose=False):
    """Return an independently audited incumbent, with a separate optimality flag.

    A positive requested gap can yield a valid incumbent without a proof of
    optimality. Invalid solver output is withheld with status 4, valid=False.
    max_routes_per_depot explicitly overrides the instance's per-depot limit.
    """
    inst.validate()
    for label, value in (("time_limit", time_limit), ("mip_rel_gap", mip_rel_gap)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be a finite nonnegative number")
    if max_routes_per_depot is not None and (isinstance(max_routes_per_depot, bool)
            or not isinstance(max_routes_per_depot, Integral) or max_routes_per_depot < 0):
        raise ValueError("max_routes_per_depot must be a nonnegative integer or None")
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
    if not Cs:
        return {"status": 0, "message": "empty customer set", "valid": True, "optimal": True,
                "objective": 0.0, "bound": 0.0, "gap": 0.0, "routes": [], "seconds": 0.0,
                "n_arcs": 0, "n_vars": 0, "n_cons": 0}
    if K == 0 or np.any(q[Cs] > Q):
        return {"status": 2, "message": "fleet or payload capacity cannot serve every customer",
                "valid": False, "optimal": False, "objective": None, "bound": None, "gap": None,
                "routes": None, "seconds": 0.0, "n_arcs": 0, "n_vars": 0, "n_cons": 0}

    # --- arcs, with simple preprocessing (arcs that can never be feasible are dropped)
    arcs = []
    for i in range(N):
        for j in range(N):
            if i == j or (i < nD and j < nD):
                continue
            if i >= nD and j >= nD:
                if e[i] + S[i] + T[i, j] > l[j] + 1e-9 or q[i] + q[j] > Q + 1e-9:
                    continue
            if i < nD and j >= nD and e[i] + S[i] + T[i, j] > l[j] + 1e-9:
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
    # A visit is consumed even when demand, travel and service are all zero.
    # Flow can originate only at depots; a customer-only cycle cannot supply it.
    flow = {}
    for arc, (_, j) in enumerate(arcs):
        if j >= nD:
            flow[arc] = k
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
    for v in flow.values():
        ub[v] = len(Cs)
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
    lbr = int(np.ceil((q[Cs] / Q).sum() - 1e-9)) if Q > 0 else 0
    B.add({a_: 1.0 for d in Ds for a_ in outs[d]}, lbr, np.inf)
    for j in Cs:
        net = {flow[a_]: 1.0 for a_ in ins[j]}
        for a_ in outs[j]:
            if a_ in flow:
                net[flow[a_]] = -1.0
        B.add(net, 1, 1)
    for a_, v in flow.items():
        B.add({v: 1.0, a_: -len(Cs)}, -np.inf, 0)

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
            M = e[d] + S[d] + T[d, jj] - e[jj]
            if M > 0:
                B.add({tv[jj]: 1.0, a_: -M}, e[d] + S[d] + T[d, jj] - M, np.inf)
            if gv is not None:
                M = em.e_trip + ED[d, jj] * (em.a + em.b * Q)
                B.add({gv[jj]: 1.0, rv[jj]: -ED[d, jj] * em.b, a_: -M}, em.e_trip + ED[d, jj] * em.a - M, np.inf)
            if yv is not None:
                B.add({yv[jj, d]: 1.0, a_: -1.0}, 0, np.inf)
            if tauv is not None:
                # Departure includes the depot's service, as in the route checker.
                B.add({tv[jj]: 1.0, tauv[jj]: -1.0, a_: -Hz}, S[d] + T[d, jj] - Hz, np.inf)
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
    status = res.status if isinstance(res.status, Integral) and not isinstance(res.status, bool) else 4
    out = {"status": int(status), "message": res.message, "seconds": secs, "n_arcs": nx,
           "n_vars": nvar, "n_cons": len(B.lo)}
    if res.x is None:
        out.update(valid=False, optimal=False, objective=None,
                   bound=_finite_metadata(getattr(res, "mip_dual_bound", None)), gap=None, routes=None)
        return out
    try:
        if status not in (0, 1):
            raise ValueError("solver status does not support an incumbent")
        routes, cost = _audit_incumbent(inst, res, arcs, nvar, integ, lb, ub, K)
        bound = _finite_metadata(getattr(res, "mip_dual_bound", None))
        gap = _finite_metadata(getattr(res, "mip_gap", None))
        if gap is not None and gap < 0:
            raise ValueError("negative MIP gap")
        if bound is not None and bound > cost and not _same_cost(bound, cost):
            raise ValueError("dual bound exceeds independently checked objective")
        if gap == 0 and bound is not None and not _same_cost(bound, cost):
            raise ValueError("zero MIP gap disagrees with the dual bound")
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        out.update(status=4, solver_status=int(status), message=f"invalid solver result: {exc}",
                   valid=False, optimal=False, objective=None, bound=None, gap=None, routes=None)
        return out
    optimal = status == 0 and gap == 0 and bound is not None and _same_cost(bound, cost)
    out.update(valid=True, optimal=optimal, objective=cost, solver_objective=float(res.fun),
               bound=bound, gap=gap, routes=routes)
    return out


def _same_cost(left, right):
    return math.isclose(left, right, rel_tol=1e-7, abs_tol=_COST_TOL)


def _finite_metadata(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    return float(value)


def _audit_incumbent(inst, result, arcs, nvar, integrality, lower, upper, fleet_limit):
    """Check numeric output, decompose all selected arcs, then use the independent checker."""
    values = np.asarray(result.x, dtype=float)
    if values.shape != (nvar,) or not np.isfinite(values).all():
        raise ValueError("solver vector has incorrect shape or nonfinite entries")
    if np.any(values < lower - _INTEGER_TOL) or np.any(values > upper + _INTEGER_TOL):
        raise ValueError("solver vector is outside variable bounds")
    discrete = values[integrality != 0]
    if np.any(np.abs(discrete - np.rint(discrete)) > _INTEGER_TOL):
        raise ValueError("solver returned noninteger discrete decisions")
    selected = [arc for arc, used in zip(arcs, np.rint(values[:len(arcs)]).astype(int)) if used]
    successors = {node: [] for node in range(inst.n_nodes)}
    predecessors = {node: [] for node in range(inst.n_nodes)}
    for i, j in selected:
        successors[i].append(j)
        predecessors[j].append(i)
    for customer in inst.customers:
        if len(successors[customer]) != 1 or len(predecessors[customer]) != 1:
            raise ValueError(f"customer {customer} does not have exactly one incoming and outgoing arc")
    for depot in range(inst.n_depots):
        if len(successors[depot]) != len(predecessors[depot]) or len(successors[depot]) > fleet_limit:
            raise ValueError(f"depot {depot} has unbalanced flow or exceeds the fleet limit")

    routes, visited, used_edges = [], set(), set()
    for depot in range(inst.n_depots):
        for first in successors[depot]:
            route = [depot, first]
            used_edges.add((depot, first))
            while route[-1] >= inst.n_depots:
                node = route[-1]
                if node in visited:
                    raise ValueError("selected arcs contain a cycle or repeat a customer")
                visited.add(node)
                next_node = successors[node][0]
                route.append(next_node)
                used_edges.add((node, next_node))
            if route[-1] != depot:
                raise ValueError("route returns to a different depot")
            routes.append(route)
    if visited != set(inst.customers) or used_edges != set(selected):
        raise ValueError("selected arcs include a disconnected component or omit customers")

    # The argument is an explicit fleet-limit override, so audit that same model.
    checked = evaluate(replace(inst, max_vehicles=fleet_limit), routes)
    if not checked.feasible:
        raise ValueError("independent route feasibility failed: " + "; ".join(checked.violations))
    reported = _finite_metadata(result.fun)
    if reported is None or not math.isfinite(checked.cost) or not _same_cost(reported, checked.cost):
        raise ValueError("solver objective disagrees with independently computed route cost")
    return routes, float(checked.cost)
