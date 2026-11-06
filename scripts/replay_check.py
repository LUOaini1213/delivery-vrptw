"""Replay a fixed subset of the committed experiments with the current code.

The large tables in README.md come from results/*.json, which were produced at commit
24d7d0f. This script reruns a fixed subset of those runs with whatever code is checked
out now and compares the outcome with the stored JSON. Every compared value must be
equal at the precision the runner stored it (the runners round before writing), and
every compared route must be identical. Wall-clock seconds are never compared.

    python scripts/replay_check.py --check                       # all cases, a few minutes
    python scripts/replay_check.py --check --only drone_i1       # one case
    python scripts/replay_check.py --check --output artifacts/replay.json

Exit code 1 if any value differs, 0 if everything matches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import RESULTS, machine_info, write_json

# ------------------------------------------------------------------ the replayed subset
# README.md's "replay" block is generated from these constants (scripts/check_readme.py).
SMALL_N = 10                                       # small Solomon subsets: first 10 customers
SMALL_INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]  # one per Solomon family
SMALL_ALNS_SEED = 1                                # of the 5 stored ALNS seeds, replay seed 1
SOLOMON_RUNS = [("R101", 1), ("C201", 1)]          # 10000 ALNS iterations each, as stored
DRONE_I1_SEEDS = [1, 2, 3, 4, 5]                   # I1 baseline of every 60-order drone instance
SIM_INSTANCE = 1                                   # drone instance used for the plan / simulation replay
SIM_STATIC_SCENARIOS = list(range(200))            # all stored static scenarios of SIM_INSTANCE: scenarios x 4 policies x 9 metrics
SIM_ROLLING_SCENARIOS = [0, 1, 2, 3]                # rolling re-planning scenarios
STATIC_POLICIES = ["greedy", "alns", "robust15", "robust30"]

CASES = ["small_mip", "solomon", "drone_i1", "drone_plans", "sim_static", "sim_rolling"]


def load(name, results_dir):
    return json.loads((Path(results_dir) / name).read_text(encoding="utf-8"))


def compare(stored, current, path="", ignore=("seconds",)):
    """List of human-readable differences between two JSON-like values (empty = identical)."""
    out = []
    if isinstance(stored, dict) and isinstance(current, dict):
        for k in sorted(set(stored) | set(current)):
            if k in ignore:
                continue
            if k not in stored or k not in current:
                out.append(f"{path}/{k}: only in {'current' if k in current else 'stored'}")
            else:
                out += compare(stored[k], current[k], f"{path}/{k}", ignore)
    elif isinstance(stored, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(stored) != len(current):
            out.append(f"{path}: length {len(stored)} stored vs {len(current)} current")
        else:
            for i, (a, b) in enumerate(zip(stored, current)):
                out += compare(a, b, f"{path}[{i}]", ignore)
    elif isinstance(stored, bool) or isinstance(current, bool):
        if stored is not current:
            out.append(f"{path}: {stored!r} stored vs {current!r} current")
    elif isinstance(stored, (int, float)) and isinstance(current, (int, float)):
        if stored != current:
            out.append(f"{path}: {stored!r} stored vs {current!r} current")
    elif stored != current:
        out.append(f"{path}: {stored!r} stored vs {current!r} current")
    return out


# ------------------------------------------------------------------ cases

def case_small_mip(results_dir):
    """Proven MIP optimum (objective, optimality flag), the I1 baseline and one ALNS seed per instance.

    The MIP's vehicle count is not compared: another optimal route set may use a different number of
    vehicles. Status, bound and gap are not compared either; the optimum's objective is what the
    README tables use."""
    from run_small import ALNS_ITERS, MIP_LIMIT

    from vrptw.alns import solve
    from vrptw.checker import evaluate
    from vrptw.construct import best_i1
    from vrptw.instance import load_solomon
    from vrptw.mip import solve_mip
    d = load("small_mip_vs_alns.json", results_dir)
    stored = {(r["instance"], r["n"]): r for r in d["runs"]}
    diffs = []
    for name in SMALL_INSTANCES:
        s = stored[(name, SMALL_N)]
        tag = f"small_mip_vs_alns.json[{name}.{SMALL_N}]"
        inst = load_solomon(name, SMALL_N)
        m = solve_mip(inst, time_limit=MIP_LIMIT[SMALL_N])
        ev = evaluate(inst, m["routes"]) if m["routes"] is not None else None
        cur_mip = {"optimal": bool(m["optimal"]), "objective": None if ev is None else round(ev.distance, 6)}
        diffs += compare({k: s["mip"].get(k) for k in cur_mip}, cur_mip, f"{tag}/mip")
        s0 = best_i1(inst)
        ev0 = evaluate(inst, s0.as_lists())
        diffs += compare(s["i1"], {"distance": round(ev0.distance, 6), "vehicles": ev0.n_routes}, f"{tag}/i1")
        stored_alns = next(a for a in s["alns"] if a["seed"] == SMALL_ALNS_SEED)
        sol, info = solve(inst, s0, iters=ALNS_ITERS[SMALL_N], seed=SMALL_ALNS_SEED)
        ev1 = evaluate(inst, sol.as_lists())
        diffs += compare(stored_alns, {"seed": SMALL_ALNS_SEED, "distance": round(ev1.distance, 6),
                                       "vehicles": ev1.n_routes, "iters": info["iters_used"]},
                         f"{tag}/alns[seed {SMALL_ALNS_SEED}]")
    return len(SMALL_INSTANCES), diffs


def case_solomon(results_dir):
    from run_solomon import run_one
    d = load("solomon100.json", results_dir)
    stored = {(r["instance"], r["seed"]): r for r in d["runs"]}
    diffs, n = [], 0
    for name, seed in SOLOMON_RUNS:
        s = stored[(name, seed)]
        cur = run_one((name, seed, d["iters"]))
        cur["bks"] = s["bks"]  # added by the runner from data/bks, not by the solver
        diffs += compare(s, cur, f"solomon100.json[{name} s{seed}]")
        n += 1
    return n, diffs


def drone_i1(seed, n, depots):
    from run_drone import SPEC, summarize

    from vrptw.construct import best_i1
    from vrptw.drone import generate
    inst = generate(seed, n=n, n_depots=depots, spec=SPEC)
    return summarize(inst, best_i1(inst).as_lists())


def case_drone_i1(results_dir):
    d = load("drone.json", results_dir)
    L = d["large"]
    diffs = []
    for seed in DRONE_I1_SEEDS:
        stored = [r for r in L["runs"] if r["instance_seed"] == seed]
        cur = drone_i1(seed, L["n"], L["depots"])
        for r in stored:  # the I1 baseline is repeated once per ALNS seed; all copies must match
            diffs += compare(r["i1"], cur, f"drone.json[s{seed} alns{r['alns_seed']}]/i1")
    return len(DRONE_I1_SEEDS), diffs


_PLANS = {}


def sim_plans():
    """build_plans(SIM_INSTANCE) runs four planners (I1, ALNS, ALNS on 15% / 30% inflated data)."""
    if SIM_INSTANCE not in _PLANS:
        from run_sim import build_plans
        _PLANS[SIM_INSTANCE] = build_plans(SIM_INSTANCE)
    return _PLANS[SIM_INSTANCE]


def case_drone_plans(results_dir):
    from run_drone import summarize
    sim = load("simulation.json", results_dir)
    drone = load("drone.json", results_dir)
    inst, plans = sim_plans()
    diffs = []
    stored_plans = sim["plans"][str(SIM_INSTANCE)]
    for tag, p in plans.items():
        cur = {"trips": len(p["routes"]), "drones": len(p["drones"]), "distance_km": round(p["distance"], 3),
               "nominal_feasible": p["nominal_feasible"], "forced_single_trips": p["forced_single_trips"]}
        diffs += compare(stored_plans[tag], cur, f"simulation.json/plans/{SIM_INSTANCE}/{tag}")
    # The simulator's ALNS plan is the same run as drone.json's (instance SIM_INSTANCE, ALNS seed 1).
    run = next(r for r in drone["large"]["runs"]
               if r["instance_seed"] == SIM_INSTANCE and r["alns_seed"] == sim["plan_seed"])
    assert drone["large"]["alns_iters"] == sim["plan_iters"]
    diffs += compare(run["routes"], plans["alns"]["routes"], f"drone.json[s{SIM_INSTANCE} alns{sim['plan_seed']}]/routes")
    diffs += compare(run["alns"], summarize(inst, plans["alns"]["routes"]),
                     f"drone.json[s{SIM_INSTANCE} alns{sim['plan_seed']}]/alns", ignore=("seconds", "iters"))
    return len(plans) + 1, diffs


def case_sim_static(results_dir):
    from run_sim import SPEC

    from vrptw.sim import execute_plan, make_scenario, metrics
    sim = load("simulation.json", results_dir)
    rows = {(r["instance"], r["scenario"]): r for r in sim["rows"]}
    inst, plans = sim_plans()
    diffs, n = [], 0
    for sc_seed in SIM_STATIC_SCENARIOS:
        sc = make_scenario(inst, 1000 * SIM_INSTANCE + sc_seed)
        row = rows[(SIM_INSTANCE, sc_seed)]
        diffs += compare(row["wind0"], float(sc.wind_speed[0]), f"simulation.json/rows[{SIM_INSTANCE},{sc_seed}]/wind0")
        for tag in STATIC_POLICIES:
            p = plans[tag]
            served, te = execute_plan(inst, sc, p["routes"], p["drones"], SPEC.swap_min)
            cur = metrics(inst, served, te, len(p["drones"]), p["distance"])
            diffs += compare(row[tag], cur, f"simulation.json/rows[{SIM_INSTANCE},{sc_seed}]/{tag}")
            n += 1
    return n, diffs


def case_sim_rolling(results_dir):
    from run_sim import run_roll
    sim = load("simulation.json", results_dir)
    rows = {(r["instance"], r["scenario"]): r for r in sim["rows"]}
    diffs = []
    for sc_seed in SIM_ROLLING_SCENARIOS:
        _, _, cur = run_roll((SIM_INSTANCE, sc_seed))
        diffs += compare(rows[(SIM_INSTANCE, sc_seed)]["rolling"], cur,
                         f"simulation.json/rows[{SIM_INSTANCE},{sc_seed}]/rolling")
    return len(SIM_ROLLING_SCENARIOS), diffs


RUNNERS = {"small_mip": case_small_mip, "solomon": case_solomon, "drone_i1": case_drone_i1, "drone_plans": case_drone_plans,
           "sim_static": case_sim_static, "sim_rolling": case_sim_rolling}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", required=True,
                    help="rerun the subset and exit 1 on any difference")
    ap.add_argument("--only", nargs="+", choices=CASES)
    ap.add_argument("--results-dir", default=str(RESULTS), help="directory holding the stored results JSON")
    ap.add_argument("--output", type=Path, help="optional JSON report path")
    a = ap.parse_args(argv)
    report = {"machine": machine_info(), "results_dir": Path(a.results_dir).name, "cases": []}
    failed = False
    for name in a.only or CASES:
        t0 = time.perf_counter()
        n, diffs = RUNNERS[name](a.results_dir)
        sec = round(time.perf_counter() - t0, 1)
        status = "match" if not diffs else "MISMATCH"
        failed |= bool(diffs)
        report["cases"].append({"case": name, "compared_runs": n, "status": status, "seconds": sec,
                                "differences": diffs[:50], "n_differences": len(diffs)})
        print(f"{name}: {n} stored runs replayed, {status} ({sec}s)", flush=True)
        for line in diffs[:20]:
            print("   ", line, flush=True)
    if a.output:
        write_json(a.output, report)
    if failed:
        print("REPLAY MISMATCH: current code does not reproduce the stored results above")
        return 1
    print("replay ok: current code reproduces every replayed stored result exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
