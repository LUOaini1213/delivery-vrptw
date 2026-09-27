"""Stage 2: delivery-drone variant (payload, battery energy, time windows, max trip duration,
battery swap at the depot).

(a) small instances: exact MIP vs ALNS (5 seeds) vs I1 -> hits / gaps
(b) 60-customer, 2-depot instances: I1 vs ALNS (3 seeds): energy, trips, drones after chaining

    python scripts/run_drone.py --workers 6
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

from common import RESULTS, machine_info, write_json

from vrptw.alns import solve
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.drone import DroneSpec, chain_trips, generate
from vrptw.mip import solve_mip

SPEC = DroneSpec()
SMALL = [(n, nd, s) for n in (8, 10, 12) for s in range(1, 9) for nd in ((1,) if s % 2 else (2,))]
SMALL_HORIZON = 120.0
SMALL_MIP_LIMIT = 300.0
SMALL_ALNS_ITERS = 500
LARGE_SEEDS = [1, 2, 3, 4, 5]
LARGE_ALNS_ITERS = 2000
ALNS_SEEDS = [1, 2, 3]


def summarize(inst, routes):
    ev = evaluate(inst, routes)
    assert ev.feasible, ev.violations[:3]
    drones = chain_trips(inst, routes, SPEC.swap_min)
    return {"energy_wh": round(ev.energy, 3), "trips": ev.n_routes, "distance_km": round(ev.distance, 3),
            "drones": len(drones), "max_trip_energy_wh": round(max(ev.route_energy), 2),
            "customers_per_trip": round((inst.n_customers) / ev.n_routes, 3)}


def run_small(args):
    n, nd, seed = args
    inst = generate(seed, n=n, n_depots=nd, spec=SPEC, horizon=SMALL_HORIZON)
    m = solve_mip(inst, time_limit=SMALL_MIP_LIMIT)
    rec = {"n": n, "depots": nd, "seed": seed,
           "mip": {"optimal": bool(m["optimal"]), "seconds": round(m["seconds"], 2), "gap": m["gap"]}}
    if m["routes"] is not None:
        ev = evaluate(inst, m["routes"])
        assert ev.feasible, ev.violations[:3]
        rec["mip"]["objective"] = round(ev.cost, 6)
        rec["mip"]["bound"] = None if m["bound"] is None else round(float(m["bound"]), 6)
    s0 = best_i1(inst)
    rec["i1"] = round(evaluate(inst, s0.as_lists()).cost, 6)
    rec["alns"] = []
    for s in range(1, 6):
        sol, info = solve(inst, s0, iters=SMALL_ALNS_ITERS, seed=s)
        ev = evaluate(inst, sol.as_lists())
        assert ev.feasible, ev.violations[:3]
        rec["alns"].append({"seed": s, "objective": round(ev.cost, 6), "seconds": round(info["seconds"], 2)})
    return rec


def run_large(args):
    seed, alns_seed = args
    inst = generate(seed, n=60, n_depots=2, spec=SPEC)
    t0 = time.perf_counter()
    s0 = best_i1(inst)
    t_i1 = time.perf_counter() - t0
    sol, info = solve(inst, s0, iters=LARGE_ALNS_ITERS, seed=alns_seed)
    return {"instance_seed": seed, "alns_seed": alns_seed, "name": inst.name,
            "i1": dict(summarize(inst, s0.as_lists()), seconds=round(t_i1, 2)),
            "alns": dict(summarize(inst, sol.as_lists()), seconds=round(info["seconds"], 1),
                         iters=info["iters_used"]),
            "routes": sol.as_lists()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(RESULTS / "drone.json"))
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        large = list(ex.map(run_large, [(s, k) for s in LARGE_SEEDS for k in ALNS_SEEDS]))
        for r in large:
            print(f"{r['name']} alns seed {r['alns_seed']}: I1 {r['i1']['energy_wh']} Wh/{r['i1']['drones']} drones, "
                  f"ALNS {r['alns']['energy_wh']} Wh/{r['alns']['drones']} drones", flush=True)
        small = list(ex.map(run_small, sorted(SMALL, key=lambda t: -t[0])))
        for r in small:
            best = min(x["objective"] for x in r["alns"])
            print(f"small n={r['n']} d={r['depots']} s={r['seed']}: MIP {r['mip'].get('objective')} "
                  f"opt={r['mip']['optimal']} ({r['mip']['seconds']}s) ALNS {best}", flush=True)
    small.sort(key=lambda r: (r["n"], r["seed"]))
    write_json(a.out, {"experiment": "drone", "spec": SPEC.__dict__, "energy_model": SPEC.energy_model().__dict__,
                       "small": {"horizon": SMALL_HORIZON, "mip_time_limit": SMALL_MIP_LIMIT,
                                 "alns_iters": SMALL_ALNS_ITERS, "alns_seeds": [1, 2, 3, 4, 5], "runs": small},
                       "large": {"n": 60, "depots": 2, "alns_iters": LARGE_ALNS_ITERS, "alns_seeds": ALNS_SEEDS,
                                 "runs": large},
                       "wall_seconds": round(time.time() - t0, 1), "workers": a.workers,
                       "machine": machine_info()})


if __name__ == "__main__":
    main()
