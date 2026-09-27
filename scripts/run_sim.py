"""Stage 3: execute drone plans in a stochastic simulator and compare policies with paired seeds.

Policies (all plans built for the same 60-customer, 2-depot instances as stage 2):
  greedy      Solomon I1 plan, static
  alns        ALNS plan (nominal travel times / energy), static
  robust15    ALNS plan on times and energies inflated by 15% (buffers), static
  robust30    same with 30%
  rolling     re-plan every 60 min with ALNS using the wind observed at that moment
              (worst-heading factor), launch trips until the next re-plan
Every policy is run on the same scenarios (common random numbers). Differences to
`alns` are paired per (instance, scenario) and summarised with 95% percentile-
bootstrap CIs of the mean difference.

    python scripts/run_sim.py --workers 6
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import RESULTS, machine_info, write_json

from vrptw.alns import solve
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.drone import DroneSpec, chain_trips, complete_with_singletons, generate, inflate
from vrptw.sim import bootstrap_ci, execute_plan, make_scenario, metrics, run_rolling

SPEC = DroneSpec()
INSTANCES = [1, 2, 3, 4, 5]
N_STATIC = 200        # scenarios per instance for the static policies
N_ROLLING = 40        # first N scenarios per instance also run with the rolling policy
PLAN_ITERS = 2000
PLAN_SEED = 1
ROLL_ITERS = 200
METRICS = ["on_time", "mean_late", "p95_late", "reserve_viol", "depleted", "energy", "drones", "trips", "distance"]


def build_plans(seed):
    inst = generate(seed, n=60, n_depots=2, spec=SPEC)
    plans = {}
    s0 = best_i1(inst)
    plans["greedy"] = (inst, s0.as_lists(), [])
    sol, _ = solve(inst, s0, iters=PLAN_ITERS, seed=PLAN_SEED)
    plans["alns"] = (inst, sol.as_lists(), [])
    for tag, th in (("robust15", 0.15), ("robust30", 0.30)):
        pi = inflate(inst, 1 + th, 1 + th)
        s, _ = solve(pi, best_i1(pi), iters=PLAN_ITERS, seed=PLAN_SEED)
        plans[tag] = (pi, complete_with_singletons(pi, s), list(s.unassigned))
    out = {}
    for tag, (pinst, routes, singles) in plans.items():
        ev = evaluate(inst, routes)  # nominal feasibility of the executed routes
        drones = chain_trips(pinst, routes, SPEC.swap_min)
        out[tag] = {"routes": routes, "drones": drones, "distance": ev.distance,
                    "nominal_feasible": ev.feasible, "forced_single_trips": singles}
    return inst, out


def run_instance(seed):
    inst, plans = build_plans(seed)
    rows = []
    for sc_seed in range(N_STATIC):
        sc = make_scenario(inst, 1000 * seed + sc_seed)
        row = {"instance": seed, "scenario": sc_seed, "wind0": float(sc.wind_speed[0])}
        for tag, p in plans.items():
            served, te = execute_plan(inst, sc, p["routes"], p["drones"], SPEC.swap_min)
            row[tag] = metrics(inst, served, te, len(p["drones"]), p["distance"])
        rows.append(row)
    return seed, plans, rows


def run_roll(args):
    seed, sc_seed = args
    inst = generate(seed, n=60, n_depots=2, spec=SPEC)
    sc = make_scenario(inst, 1000 * seed + sc_seed)
    t0 = time.perf_counter()
    served, te, nd, dist, ep = run_rolling(inst, sc, SPEC.swap_min, iters=ROLL_ITERS, seed=sc_seed)
    m = metrics(inst, served, te, nd, dist)
    m["replans"] = ep
    m["seconds"] = round(time.perf_counter() - t0, 1)
    return seed, sc_seed, m


def summarize(rows, policies, base="alns"):
    summ = {}
    for p in policies:
        vals = [r[p] for r in rows if p in r]
        summ[p] = {"n": len(vals), **{m: round(float(np.mean([v[m] for v in vals])), 4) for m in METRICS}}
        if p != base:
            summ[p]["diff_vs_" + base] = {}
            for m in METRICS:
                d = [r[p][m] - r[base][m] for r in rows if p in r]
                mean, lo, hi = bootstrap_ci(d)
                summ[p]["diff_vs_" + base][m] = [round(mean, 4), round(lo, 4), round(hi, 4)]
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(RESULTS / "simulation.json"))
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_instance, INSTANCES))
        rows = [r for _, _, rr in res for r in rr]
        plans = {seed: {k: {kk: vv for kk, vv in v.items()} for k, v in p.items()} for seed, p, _ in res}
        print("static done", flush=True)
        roll = list(ex.map(run_roll, [(s, k) for s in INSTANCES for k in range(N_ROLLING)]))
    idx = {(r["instance"], r["scenario"]): r for r in rows}
    for seed, sc_seed, m in roll:
        idx[(seed, sc_seed)]["rolling"] = m
    static_pols = ["greedy", "alns", "robust15", "robust30"]
    sub = [r for r in rows if "rolling" in r]
    out = {
        "experiment": "simulation", "instances": INSTANCES, "n_static": N_STATIC, "n_rolling": N_ROLLING,
        "plan_iters": PLAN_ITERS, "plan_seed": PLAN_SEED, "rolling_iters_per_replan": ROLL_ITERS,
        "scenario_seed_rule": "make_scenario(inst, 1000 * instance_seed + scenario)",
        "summary_static": summarize(rows, static_pols),
        "summary_with_rolling": summarize(sub, static_pols + ["rolling"]),
        "plans": {str(s): {k: {"trips": len(v["routes"]), "drones": len(v["drones"]),
                               "distance_km": round(v["distance"], 3), "nominal_feasible": v["nominal_feasible"],
                               "forced_single_trips": v["forced_single_trips"]} for k, v in p.items()}
                  for s, p in plans.items()},
        "rolling_seconds_mean": round(float(np.mean([m["seconds"] for _, _, m in roll])), 1),
        "wall_seconds": round(time.time() - t0, 1), "workers": a.workers, "machine": machine_info(),
        "rows": rows,
    }
    write_json(a.out, out)
    for k, v in out["summary_static"].items():
        print(k, {m: v[m] for m in METRICS})
    for k, v in out["summary_with_rolling"].items():
        print("sub", k, {m: v[m] for m in METRICS})


if __name__ == "__main__":
    main()
