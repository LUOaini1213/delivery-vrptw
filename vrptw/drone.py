"""Delivery-drone variant: energy parameters, a seeded urban-like instance generator,
and trip chaining (battery swap at the depot) that turns trips into drone schedules.

Energy model (linear in total mass), after D'Andrea (2014), "Guest editorial:
Can drones deliver?", IEEE T-ASE 11(3):647-648, as restated by Muli, Park & Liu
(2022, arXiv:2206.01609, eq. 1):
    P [kW] = (m_vehicle + m_payload) [kg] * v_air [km/h] / (370 * eta * r) + p_avionics [kW]
so energy per air-kilometre = P / v_air = (m_v + m_p) / (370 eta r) + p_av / v_air   [kWh/km],
i.e. a + b * payload with
    a = 1000 * m_v / (370 eta r) + 1000 * p_av / v_air      [Wh/km]
    b = 1000 / (370 eta r)                                  [Wh/(km kg)]
eta = 0.5 follows D'Andrea; r = 3, m_v = 7 kg, p_av = 0.1 kW, v_air = 54 km/h,
the fixed take-off/landing energy and the battery size are our own illustrative
choices (not calibrated to any real aircraft). Waiting (early arrival) is assumed
to happen on the ground and to cost no energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import EnergyModel, make_instance
from .routes import Route


@dataclass(frozen=True)
class DroneSpec:
    m_vehicle: float = 7.0       # kg, airframe + battery
    eta: float = 0.5             # power transfer efficiency (D'Andrea 2014)
    lift_drag: float = 3.0       # lift-to-drag ratio (assumption)
    p_avionics: float = 0.1      # kW (assumption)
    v_air_kmh: float = 54.0      # cruise airspeed, 15 m/s
    payload_cap: float = 3.0     # kg
    e_trip: float = 6.0          # Wh per take-off + landing (assumption)
    battery: float = 250.0       # Wh
    reserve: float = 0.2         # fraction that must remain on landing
    service_min: float = 2.0     # minutes to lower a parcel
    swap_min: float = 5.0        # minutes to swap the battery at the depot
    max_trip_min: float = 45.0   # max trip duration (a loaded drone should not sit idle for long)

    def energy_model(self) -> EnergyModel:
        k = 1000.0 / (370.0 * self.eta * self.lift_drag)
        a = k * self.m_vehicle + 1000.0 * self.p_avionics / self.v_air_kmh
        return EnergyModel(a=a, b=k, e_trip=self.e_trip, battery=self.battery, reserve=self.reserve)

    @property
    def speed_km_per_min(self) -> float:
        return self.v_air_kmh / 60.0


def generate(seed: int, n: int = 60, n_depots: int = 2, spec: DroneSpec = DroneSpec(),
             area: float = 12.0, n_clusters: int = 4, horizon: float = 360.0):
    """Urban-like clustered customers around `n_clusters` hot spots in an area x area km square.

    Time windows: ready e ~ U(0, horizon - 60) min, width ~ U{30, 45, 60} min.
    Demand ~ U(0.3, 2.5) kg. Every customer is guaranteed to be reachable on a
    single-customer trip from at least one depot (otherwise its position is resampled).
    """
    rng = np.random.default_rng(seed)
    em = spec.energy_model()
    v = spec.speed_km_per_min
    if n_depots == 1:
        depots = np.array([[area / 2, area / 2]])
    else:
        depots = np.array([[area * (k + 0.5) / n_depots, area / 2] for k in range(n_depots)])
    centres = rng.uniform(0.2 * area, 0.8 * area, size=(n_clusters, 2))
    pts, dem = [], []
    while len(pts) < n:
        if rng.random() < 0.8:
            p = centres[rng.integers(n_clusters)] + rng.normal(0, 0.08 * area, 2)
        else:
            p = rng.uniform(0, area, 2)
        p = np.clip(p, 0, area)
        q = float(np.round(rng.uniform(0.3, 2.5), 2))
        d = np.sqrt(((depots - p) ** 2).sum(1))
        e_single = em.e_trip + d * (em.a + em.b * q) + d * em.a
        if e_single.min() <= em.usable * 0.95:
            pts.append(p)
            dem.append(q)
    coords = np.vstack([depots, np.array(pts)])
    ready = np.r_[np.zeros(n_depots), np.round(rng.uniform(0, horizon - 60, n), 1)]
    width = np.r_[np.zeros(n_depots), rng.choice([30.0, 45.0, 60.0], n)]
    due = ready + width
    due[:n_depots] = horizon + 60.0
    # the window must be reachable from the depot: shift late ones if necessary
    d = np.sqrt(((coords[None, :n_depots, :] - coords[:, None, :]) ** 2).sum(-1)).min(1)
    ready = np.maximum(ready, 0.0)
    due = np.maximum(due, d / v + 1.0)
    service = np.r_[np.zeros(n_depots), np.full(n, spec.service_min)]
    demand = np.r_[np.zeros(n_depots), dem]
    inst = make_instance(f"drone-s{seed}-n{n}-d{n_depots}", coords, demand, ready, due, service,
                         capacity=spec.payload_cap, n_depots=n_depots, speed=v, energy=em,
                         max_duration=spec.max_trip_min, vehicle_cost=0.0, w_dist=0.0, w_energy=1.0)
    inst.meta = {"generator": "vrptw.drone.generate", "seed": seed, "spec": spec.__dict__.copy(),
                 "units": {"distance": "km", "time": "min", "energy": "Wh", "demand": "kg"}}
    return inst


def complete_with_singletons(inst, sol):
    """Preserve planned routes and add one trip per unassigned customer.

    Prefer the nearest depot whose singleton meets all route constraints and
    whose fleet still has a slot, counting both existing and newly added trips.
    If none qualifies, use the nearest depot anyway: this execution fallback
    deliberately retains late or otherwise infeasible deliveries for simulation
    and violation accounting. The returned routes are not guaranteed feasible.
    """
    inst.validate()
    routes = sol.as_lists()
    per_depot = {d: 0 for d in range(inst.n_depots)}
    for route in routes:
        per_depot[route[0]] += 1
    for c in sol.unassigned:
        depots = sorted(range(inst.n_depots), key=lambda d: (inst.dist[d, c], d))
        d = next((d for d in depots
                  if (inst.max_vehicles is None or per_depot[d] < inst.max_vehicles)
                  and Route(inst, [d, c, d]).feasible(inst)), depots[0])
        routes.append([d, c, d])
        per_depot[d] += 1
    return routes


def inflate(inst, time_factor: float = 1.0, energy_factor: float = 1.0):
    """Planning copy with travel times and energy distances scaled up (robust buffers)."""
    from dataclasses import replace
    return replace(inst, time=inst.time * time_factor, energy_dist=inst.energy_dist * energy_factor,
                   meta=dict(inst.meta, inflated=(time_factor, energy_factor)))


# ----------------------------------------------------------------------------- trip chaining

def trip_window(inst, nodes):
    """(preferred departure, latest departure) of a trip.

    Preferred = the departure giving the shortest trip (earliest departure delayed
    by min(forward slack, total waiting)); any departure up to the latest keeps
    all time windows.
    """
    r = Route(inst, nodes)
    pref = max(r.a[0], min(r.z[0], r.a[0] + r.waitcum[-1]))
    return float(pref), float(max(r.z[0], pref))  # a trip that cannot be on time leaves ASAP


def trip_times(inst, nodes, dep):
    """Planned service starts and return time when departing at `dep`."""
    t = dep
    out = [dep]
    for i, j in zip(nodes[:-1], nodes[1:]):
        t = max(t + inst.service[i] + inst.time[i, j], inst.ready[j])
        out.append(t)
    return out


def chain_trips(inst, trips, swap: float, available=None, keep_all=False):
    """Greedy assignment of trips to drones (per depot) with a battery swap between trips.

    Trips are taken in order of preferred departure; each goes to the drone at its
    depot that becomes free latest while still allowing a feasible departure
    (best fit), else to a new drone. Heuristic (not an optimal schedule).
    `available`: optional list of (depot, time) for drones that already exist.
    Returns list of drones: {"depot", "trips": [(trip_index, dep, ret)]}; with keep_all the
    first len(available) entries are the given drones (possibly without trips).
    """
    drones = [{"depot": d, "free": t, "trips": []} for d, t in (available or [])]
    order = sorted(range(len(trips)), key=lambda k: (trip_window(inst, trips[k])[0], k))
    for k in order:
        nodes = trips[k]
        lo, hi = trip_window(inst, nodes)
        best = None
        for dr in drones:
            if dr["depot"] != nodes[0]:
                continue
            dep = max(lo, dr["free"])
            if dep <= hi + 1e-9 and (best is None or dr["free"] > best["free"]):
                best = dr
        if best is None:
            best = {"depot": nodes[0], "free": -np.inf, "trips": []}
            drones.append(best)
        dep = max(lo, best["free"])
        ret = trip_times(inst, nodes, dep)[-1]
        best["trips"].append((k, float(dep), float(ret)))
        best["free"] = ret + swap
    return [{"depot": dr["depot"], "trips": dr["trips"]} for dr in drones if dr["trips"] or keep_all]
