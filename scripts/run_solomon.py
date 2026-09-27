"""Stage 1b: ALNS vs Solomon I1 on all 56 Solomon 100-customer instances, compared with the BKS.

Hierarchical objective (vehicles, then distance), double-precision Euclidean
distances, budget in ALNS iterations (deterministic per seed). Every solution
is re-checked with vrptw.checker before it is written.

    python scripts/run_solomon.py --iters 10000 --seeds 1 2 3 --workers 6
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

from common import RESULTS, ROOT, machine_info, write_json

from vrptw.alns import Params, solve
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.instance import SOLOMON_NAMES, family, load_solomon


def run_one(args):
    name, seed, iters = args
    inst = load_solomon(name)
    t0 = time.perf_counter()
    start = best_i1(inst)
    t_i1 = time.perf_counter() - t0
    ev0 = evaluate(inst, start.as_lists())
    assert ev0.feasible, (name, ev0.violations[:3])
    sol, info = solve(inst, start, iters=iters, seed=seed, params=Params(), hierarchical=True)
    ev = evaluate(inst, sol.as_lists())
    assert ev.feasible, (name, seed, ev.violations[:3])
    return {"instance": name, "family": family(name), "seed": seed, "iters": iters,
            "i1": {"vehicles": ev0.n_routes, "distance": round(ev0.distance, 4), "seconds": round(t_i1, 2)},
            "alns": {"vehicles": ev.n_routes, "distance": round(ev.distance, 4),
                     "seconds": round(info["seconds"], 1), "iters_used": info["iters_used"],
                     "stage_a_iters": info["stage_a_iters"], "route_history": info["route_history"],
                     "destroy_weights": info["destroy_weights"], "repair_weights": info["repair_weights"]},
            "routes": sol.as_lists()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--instances", nargs="*", default=SOLOMON_NAMES)
    ap.add_argument("--out", default=str(RESULTS / "solomon100.json"))
    a = ap.parse_args()
    jobs = [(n, s, a.iters) for n in a.instances for s in a.seeds]
    bks = json.loads((ROOT / "data" / "bks" / "solomon100_bks.json").read_text())
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_one, jobs):
            b = bks["instances"][r["instance"]]
            r["bks"] = {"vehicles": b["vehicles"], "distance": b["distance"], "ref": b["ref"]}
            rows.append(r)
            print(f"{r['instance']} s{r['seed']}: I1 {r['i1']['vehicles']}/{r['i1']['distance']:.1f} "
                  f"ALNS {r['alns']['vehicles']}/{r['alns']['distance']:.2f} "
                  f"BKS {b['vehicles']}/{b['distance']} ({r['alns']['seconds']}s)", flush=True)
    write_json(a.out, {
        "experiment": "solomon100", "objective": "hierarchical (vehicles, distance)",
        "distance_convention": "Euclidean, double precision, travel time = distance, no rounding",
        "bks_source": bks["source"], "bks_retrieved": bks["retrieved"],
        "iters": a.iters, "seeds": a.seeds, "workers": a.workers, "wall_seconds": round(time.time() - t0, 1),
        "machine": machine_info(), "runs": rows})


if __name__ == "__main__":
    main()
