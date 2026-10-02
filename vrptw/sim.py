"""Discrete-event execution of drone delivery plans under uncertainty.

A plan is a list of drones, each with an ordered list of trips and planned
departure times (from drone.chain_trips). Uncertainty per scenario (seeded):
  * wind: a day-level speed W0 ~ U(0, w_max) m/s and direction ~ U(0, 2pi), plus an
    hourly perturbation (speed +N(0, 1), direction +N(0, 0.35 rad)). For an arc with
    heading h the ground speed is  v_g = w.h + sqrt(v_air^2 - |w x h|^2); flight time
    and energy both scale with v_air / v_g (constant power at constant airspeed);
  * per-arc multiplicative noise LogNormal(0, 0.05) on time and energy;
  * customer service (drop-off) time multiplied by LogNormal(0, 0.3); depot
    preparation service remains deterministic and precedes the first flight.
Random numbers are drawn per scenario independently of the policy, so all
policies see exactly the same scenario (common random numbers -> paired comparison).
A drone departs a trip at max(planned departure, time it is free again), waits on
the ground if early, and flies the trip as planned (no in-flight re-routing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    seed: int
    wind_speed: np.ndarray      # m/s per hour
    wind_dir: np.ndarray        # rad per hour
    arc_noise: np.ndarray       # (N, N)
    service_noise: np.ndarray   # (N,)
    factors: np.ndarray         # (H, N, N) v_air / v_ground per hour


def wind_factors(coords, speed_ms, direction, v_air_ms):
    """v_air / v_ground for every arc under a uniform wind (speed_ms blowing towards `direction`)."""
    diff = coords[None, :, :] - coords[:, None, :]
    L = np.sqrt((diff ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(L[..., None] > 0, diff / L[..., None], 0.0)
    w = speed_ms * np.array([np.cos(direction), np.sin(direction)])
    along = h @ w
    cross2 = np.maximum(speed_ms ** 2 - along ** 2, 0.0)
    vg = along + np.sqrt(np.maximum(v_air_ms ** 2 - cross2, 1e-9))
    f = v_air_ms / np.maximum(vg, 1e-6)
    f[L == 0] = 1.0
    return f


def make_scenario(inst, seed, hours=10, w_max=7.0, v_air_ms=15.0, arc_sigma=0.05, service_sigma=0.3):
    rng = np.random.default_rng([seed, 7919])
    n = inst.n_nodes
    w0 = rng.uniform(0.0, w_max)
    d0 = rng.uniform(0.0, 2 * np.pi)
    ws = np.clip(w0 + rng.normal(0.0, 1.0, hours), 0.0, 0.8 * v_air_ms)
    wd = d0 + rng.normal(0.0, 0.35, hours)
    arc = rng.lognormal(0.0, arc_sigma, (n, n))
    srv = rng.lognormal(0.0, service_sigma, n)
    F = np.stack([wind_factors(inst.coords, ws[h], wd[h], v_air_ms) for h in range(hours)])
    return Scenario(seed, ws, wd, arc, srv, F)


def headwind_factor(speed_ms, v_air_ms=15.0):
    """Worst-heading slowdown for a wind speed (used by the rolling policy as its forecast)."""
    return v_air_ms / max(v_air_ms - speed_ms, 1e-6)


def execute_trip(inst, sc: Scenario, nodes, dep):
    """Fly one trip from `dep`. Returns (return_time, energy, {customer: (start, lateness)})."""
    em = inst.energy
    H = sc.factors.shape[0]
    t = dep
    payload = float(sum(inst.demand[c] for c in nodes[1:-1]))
    energy = em.e_trip
    out = {}
    for i, j in zip(nodes[:-1], nodes[1:]):
        if i >= inst.n_depots:
            t += inst.service[i] * sc.service_noise[i]
        else:
            t += inst.service[i]
        h = min(int(max(t, 0.0) // 60.0), H - 1)
        f = sc.factors[h, i, j] * sc.arc_noise[i, j]
        t += inst.time[i, j] * f
        energy += inst.energy_dist[i, j] * (em.a + em.b * payload) * f
        if j >= inst.n_depots:
            late = max(0.0, t - inst.due[j])
            t = max(t, inst.ready[j])
            out[j] = (t, late)
            payload -= inst.demand[j]
    return t, energy, out


def execute_plan(inst, sc: Scenario, trips, drones, swap):
    """Run a static plan. `drones`: output of drone.chain_trips."""
    served = {}
    trip_energy = []
    for dr in drones:
        free = -np.inf
        for k, dep, _ in dr["trips"]:
            d = max(dep, free)
            ret, e, out = execute_trip(inst, sc, trips[k], d)
            served.update(out)
            trip_energy.append(e)
            free = ret + swap
    return served, trip_energy


def metrics(inst, served, trip_energy, n_drones, distance):
    late = np.array([served[c][1] for c in inst.customers])
    te = np.array(trip_energy)
    em = inst.energy
    return {
        "on_time": round(float(np.mean(late <= 1e-9)), 6),
        "mean_late": round(float(late.mean()), 6),
        "p95_late": round(float(np.percentile(late, 95)), 6),
        "reserve_viol": int(np.sum(te > em.usable + 1e-9)),     # landed with less than the reserve
        "depleted": int(np.sum(te > em.battery + 1e-9)),        # would not have made it back
        "energy": round(float(te.sum()), 4),
        "drones": int(n_drones),
        "trips": int(len(te)),
        "distance": round(float(distance), 4),
    }


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean of paired differences x."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


# ----------------------------------------------------------------------------- rolling re-planning

def sub_instance(inst, customers, depot_ready, factor):
    """Planning copy on still-available depots and customers, plus original node IDs.

    A depot opens no earlier than its original ready time or the current epoch.
    Closed depots are excluded, not reopened; local depot IDs therefore change.
    Travel times and energy distances are multiplied by `factor`.
    """
    from dataclasses import replace
    inst.validate()
    depots = [d for d in range(inst.n_depots) if inst.due[d] >= max(depot_ready, inst.ready[d])]
    if not depots:
        raise RuntimeError(f"all depots are closed at time {depot_ready}; cannot dispatch remaining customers")
    idx = np.array(depots + list(customers), dtype=int)
    ix = np.ix_(idx, idx)
    ready = inst.ready[idx].copy()
    ready[:len(depots)] = np.maximum(ready[:len(depots)], depot_ready)
    sub = replace(inst, name=f"{inst.name}@{depot_ready:.0f}", n_depots=len(depots),
                  coords=inst.coords[idx], demand=inst.demand[idx],
                  ready=ready, due=inst.due[idx], service=inst.service[idx], dist=inst.dist[ix],
                  time=inst.time[ix] * factor, energy_dist=inst.energy_dist[ix] * factor, meta={})
    return sub, idx


def run_rolling(inst, sc: Scenario, swap, iters=200, seed=0, theta=0.0, period=60.0, v_air_ms=15.0,
                max_epochs=48):
    """Re-plan every `period` minutes: the not-yet-launched customers are re-optimised by ALNS
    with travel times and energy scaled by the worst-heading factor of the wind observed at
    that moment (times 1 + theta); trips starting before the next re-plan are launched.
    Returns (served, trip_energy, n_drones, planned_distance, n_replans)."""
    from .alns import solve
    from .construct import best_i1
    from .drone import chain_trips, complete_with_singletons

    unserved = set(inst.customers)
    drones = []
    served, trip_energy = {}, []
    distance = 0.0
    tau = 0.0
    H = len(sc.wind_speed)
    epochs = 0
    while unserved and epochs < max_epochs:
        epochs += 1
        h = min(int(tau // 60.0), H - 1)
        f = headwind_factor(sc.wind_speed[h], v_air_ms) * (1.0 + theta)
        sub, idx = sub_instance(inst, sorted(unserved), tau, f)
        sol, _ = solve(sub, best_i1(sub), iters=iters, seed=seed + epochs)
        trips = complete_with_singletons(sub, sol)
        depot_to_local = {int(idx[d]): d for d in range(sub.n_depots)}
        active_drones = [di for di, dr in enumerate(drones) if dr["depot"] in depot_to_local]
        available = [(depot_to_local[drones[di]["depot"]], drones[di]["free"]) for di in active_drones]
        # keep_all preserves this filtered prefix; new plan entries are new drones.
        plan = chain_trips(sub, trips, swap, available=available, keep_all=True)
        for pi, dr in enumerate(plan):
            if pi < len(active_drones):
                di = active_drones[pi]
            else:
                di = len(drones)
                drones.append({"depot": int(idx[dr["depot"]]), "free": -np.inf, "used": False})
            free = drones[di]["free"]
            for k, dep, _ in dr["trips"]:
                d = max(dep, free)
                if d >= tau + period:
                    break
                nodes = [int(idx[v]) for v in trips[k]]
                ret, e, out = execute_trip(inst, sc, nodes, d)
                served.update(out)
                trip_energy.append(e)
                distance += sum(inst.dist[a, b] for a, b in zip(nodes[:-1], nodes[1:]))
                unserved -= set(out)
                free = ret + swap
                drones[di]["used"] = True
            drones[di]["free"] = free
        tau += period
    if unserved:
        raise RuntimeError("rolling policy did not serve every customer")
    return served, trip_energy, sum(d["used"] for d in drones), distance, epochs
