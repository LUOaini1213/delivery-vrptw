"""Bounded current-code verification; never overwrites the historical experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from common import ROOT
import numpy as np
import scipy

from vrptw.brute import brute_force
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.instance import EnergyModel, load_solomon, make_instance
from vrptw.mip import solve_mip


def sha(path):
    data = path.read_bytes()
    if path.suffix == ".py":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def tiny(seed):
    rng = np.random.default_rng(seed)
    nd, nc = 1 + seed % 2, 4
    coords = rng.integers(0, 4, (nd + nc, 2)).astype(float)
    coords[-1] = coords[-2]  # coincide, with zero-demand customers in every problem
    demand = np.r_[np.zeros(nd), rng.integers(0, 3, nc)]
    demand[-2:] = 0
    em = EnergyModel(1, .2, 1, 40, .2) if seed % 3 else None
    return make_instance(f"tiny-{seed}", coords, demand, np.zeros(nd + nc),
                         np.full(nd + nc, 30), np.r_[np.ones(nd), np.zeros(nc)],
                         capacity=3, n_depots=nd, energy=em, max_vehicles=1,
                         max_duration=20, vehicle_cost=2, w_energy=.25 if em else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "review.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve evidence")
    sources = sorted((ROOT / "vrptw").glob("*.py")) + [Path(__file__).resolve()]
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256_lf": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sources},
        "data_manifest_sha256": sha(ROOT / "data" / "MANIFEST.json"),
        "scope": "12 tiny exhaustive comparisons and six Solomon-10 I1/MIP comparisons; not a full benchmark rerun",
        "tiny": [], "solomon": [],
    }
    for seed in range(12):
        inst = tiny(seed)
        exact, routes = brute_force(inst)
        solved = solve_mip(inst, time_limit=10, mip_rel_gap=0)
        if routes is None:
            assert solved["status"] == 2 and solved["routes"] is None, solved
        else:
            assert solved["valid"] and solved["optimal"], solved
            checked = evaluate(inst, solved["routes"])
            assert checked.feasible, checked.violations
            assert abs(checked.cost - exact) <= 1e-6 * max(1, abs(exact)), (seed, exact, solved)
        report["tiny"].append({"seed": seed, "brute_cost": exact if routes is not None else None,
                               "mip": solved})
    for name in ("C101", "C201", "R101", "R201", "RC101", "RC201"):
        inst = load_solomon(name, 10)
        solved = solve_mip(inst, time_limit=15, mip_rel_gap=0)
        assert solved["valid"], solved
        ev = evaluate(inst, solved["routes"])
        i1 = evaluate(inst, best_i1(inst).as_lists())
        assert ev.feasible and i1.feasible
        if solved["optimal"]:
            assert i1.cost + 1e-6 * max(1, abs(ev.cost)) >= ev.cost
        report["solomon"].append({"instance": name, "n_customers": 10, "mip": solved, "i1_cost": i1.cost})
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"review passed: {len(report['tiny'])} exhaustive cases, {len(report['solomon'])} Solomon cases; {args.output}")


if __name__ == "__main__":
    main()
