"""Cross-check the exact MIP with a second solver: HiGHS vs Gurobi on the same matrix model.

Both solvers receive the identical model built by vrptw.mip.solve_mip, and every
incumbent passes the same independent route audit. Gurobi runs under the
size-limited licence bundled with the gurobipy wheel (2,000 variables and
2,000 linear constraints), so models above that size are recorded as skipped.

    python scripts/run_gurobi_check.py --n 10 25
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

from common import RESULTS, machine_info, write_json

from vrptw.instance import SOLOMON_NAMES, family, load_solomon
from vrptw.mip import solve_mip

TIME_LIMIT = {10: 60.0, 25: 300.0}
SIZE_LIMIT = 2000


def run_one(args):
    name, n = args
    inst = load_solomon(name, n)
    lim = TIME_LIMIT.get(n, 300.0)
    h = solve_mip(inst, time_limit=lim)
    rec = {"instance": name, "family": family(name), "n": n, "n_vars": h["n_vars"], "n_cons": h["n_cons"],
           "highs": {"optimal": bool(h["optimal"]), "objective": h["objective"], "seconds": round(h["seconds"], 3)}}
    if h["n_vars"] > SIZE_LIMIT:
        rec["gurobi"] = {"skipped": "model above the size-limited licence"}
        return rec
    g = solve_mip(inst, time_limit=lim, solver="gurobi")
    rec["gurobi"] = {"optimal": bool(g["optimal"]), "objective": g["objective"], "seconds": round(g["seconds"], 3),
                     "valid": bool(g["valid"]), "message": g["message"]}
    if g["status"] == 4 and "10010" in str(g["message"]):
        rec["gurobi"] = {"skipped": "model above the size-limited licence"}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", nargs="*", default=SOLOMON_NAMES)
    ap.add_argument("--out", default=str(RESULTS / "gurobi_crosscheck.json"))
    a = ap.parse_args()
    jobs = [(name, n) for n in a.n for name in a.instances]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, jobs))
    rows.sort(key=lambda r: (r["n"], r["instance"]))
    summary = {}
    for n in a.n:
        rs = [r for r in rows if r["n"] == n]
        both = [r for r in rs if "skipped" not in r["gurobi"] and r["highs"]["optimal"] and r["gurobi"]["optimal"]]
        agree = [r for r in both if math.isclose(r["highs"]["objective"], r["gurobi"]["objective"], rel_tol=1e-7, abs_tol=1e-6)]
        summary[str(n)] = {"instances": len(rs),
                           "gurobi_run": sum("skipped" not in r["gurobi"] for r in rs),
                           "gurobi_skipped_size": sum("skipped" in r["gurobi"] for r in rs),
                           "highs_optimal": sum(r["highs"]["optimal"] for r in rs),
                           "gurobi_optimal": sum(r["gurobi"].get("optimal", False) for r in rs),
                           "both_optimal": len(both), "objectives_agree": len(agree),
                           "disagree": [r["instance"] for r in both if r not in agree]}
        print(n, summary[str(n)], flush=True)
    write_json(a.out, {"experiment": "gurobi_crosscheck", "objective": "total distance",
                       "time_limit": {str(k): v for k, v in TIME_LIMIT.items()}, "size_limit": SIZE_LIMIT,
                       "summary": summary, "wall_seconds": round(time.time() - t0, 1),
                       "machine": machine_info(), "runs": rows})


if __name__ == "__main__":
    main()
