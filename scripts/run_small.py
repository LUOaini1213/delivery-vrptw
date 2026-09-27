"""Stage 1a: exact MIP (HiGHS) vs ALNS vs I1 on the first n customers of every Solomon instance.

Objective: total distance (vehicle count free, <= 25), double precision.
The MIP either proves optimality or reports its incumbent, dual bound and gap
at the time limit. ALNS is run with several seeds and a fixed iteration budget.

    python scripts/run_small.py --n 10 25 --seeds 1 2 3 4 5 --workers 6
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

from common import RESULTS, machine_info, write_json

from vrptw.alns import solve
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.instance import SOLOMON_NAMES, family, load_solomon
from vrptw.mip import solve_mip

MIP_LIMIT = {10: 60.0, 25: 300.0}
ALNS_ITERS = {10: 1000, 25: 2000}


def run_one(args):
    name, n, seeds = args
    inst = load_solomon(name, n)
    lim = MIP_LIMIT.get(n, 300.0)
    m = solve_mip(inst, time_limit=lim)
    rec = {"instance": name, "family": family(name), "n": n,
           "mip": {"status": m["status"], "optimal": bool(m["optimal"]), "seconds": round(m["seconds"], 2),
                   "n_vars": m["n_vars"], "n_cons": m["n_cons"], "time_limit": lim}}
    if m["routes"] is not None:
        ev = evaluate(inst, m["routes"])
        assert ev.feasible, (name, n, ev.violations[:3])
        rec["mip"].update(objective=round(ev.distance, 6), vehicles=ev.n_routes,
                          bound=None if m["bound"] is None else round(float(m["bound"]), 6),
                          gap=None if m["gap"] is None else float(m["gap"]))
    t0 = time.perf_counter()
    s0 = best_i1(inst)
    ev0 = evaluate(inst, s0.as_lists())
    assert ev0.feasible
    rec["i1"] = {"distance": round(ev0.distance, 6), "vehicles": ev0.n_routes,
                 "seconds": round(time.perf_counter() - t0, 2)}
    rec["alns"] = []
    for s in seeds:
        sol, info = solve(inst, s0, iters=ALNS_ITERS.get(n, 2000), seed=s)
        ev = evaluate(inst, sol.as_lists())
        assert ev.feasible, (name, n, s, ev.violations[:3])
        rec["alns"].append({"seed": s, "distance": round(ev.distance, 6), "vehicles": ev.n_routes,
                            "seconds": round(info["seconds"], 2), "iters": info["iters_used"]})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--instances", nargs="*", default=SOLOMON_NAMES)
    ap.add_argument("--out", default=str(RESULTS / "small_mip_vs_alns.json"))
    a = ap.parse_args()
    jobs = [(name, n, a.seeds) for n in a.n for name in a.instances]
    # hardest MIPs first so the pool stays busy
    jobs.sort(key=lambda j: (-j[1], not family(j[0]).endswith("2")))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)
            mo = r["mip"].get("objective")
            best = min(x["distance"] for x in r["alns"])
            tag = "opt" if r["mip"]["optimal"] else "lim"
            print(f"{r['instance']}.{r['n']}: MIP {tag} {mo} ({r['mip']['seconds']}s) "
                  f"ALNS best {best} I1 {r['i1']['distance']}", flush=True)
    rows.sort(key=lambda r: (r["n"], r["instance"]))
    write_json(a.out, {"experiment": "small_mip_vs_alns", "objective": "total distance",
                       "mip_time_limit": {str(k): v for k, v in MIP_LIMIT.items()},
                       "alns_iters": {str(k): v for k, v in ALNS_ITERS.items()},
                       "seeds": a.seeds, "workers": a.workers, "wall_seconds": round(time.time() - t0, 1),
                       "machine": machine_info(), "runs": rows})


if __name__ == "__main__":
    main()
