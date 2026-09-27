"""Mutation testing: apply deliberate bugs one at a time and check that the test suite fails.

Before the real mutants, the harness checks itself:
  * the unmodified copy must PASS (otherwise every mutant would look "killed");
  * an equivalent mutant (adds a comment) must SURVIVE (proves the harness can report survival).

    python scripts/mutation_check.py --workers 4          # writes MUTATION.md and results/mutation.json
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import RESULTS, ROOT, write_json

# (id, file, old, new, description) - `old` must occur exactly once in the file
MUTANTS = [
    ("M01", "vrptw/checker.py", "if starts[pos] > float(inst.due[node]) + EPS:", "if starts[pos] > float(inst.due[node]) + 1.0:",
     "checker: time-window check off by one time unit"),
    ("M02", "vrptw/checker.py", "if load > inst.capacity + EPS:", "if load > inst.capacity * 1.1:",
     "checker: capacity check 10% too loose"),
    ("M03", "vrptw/checker.py", "            payload -= float(inst.demand[j])", "            payload += float(inst.demand[j])",
     "checker: wrong sign when payload is dropped (energy)"),
    ("M04", "vrptw/checker.py", "e > inst.energy.usable + EPS:", "e > inst.energy.battery + EPS:",
     "checker: battery check ignores the reserve margin"),
    ("M05", "vrptw/checker.py", "            if c in seen:", "            if False:",
     "checker: duplicate visits not detected"),
    ("M06", "vrptw/checker.py", "        cands.add(max(e0, float(inst.due[node]) - pt))\n", "",
     "checker: min-duration misses latest-departure candidates"),
    ("M07", "vrptw/routes.py", "    ok = start <= ud.l\n", "    ok = start <= ud.l + 1.0\n",
     "search: insertion time-window check off by one"),
    ("M08", "vrptw/routes.py", "    ok &= arr_j <= E.zn[:, None]\n", "    ok &= arr_j <= E.zn[:, None] + 5.0\n",
     "search: successor latest-start (push-forward) check too loose"),
    ("M09", "vrptw/routes.py", "    ok &= load + ud.q <= inst.capacity + EPS\n", "    ok &= load <= inst.capacity + EPS\n",
     "search: insertion ignores the inserted customer's demand"),
    ("M10", "vrptw/routes.py", "+ ED[ii, Ur] * (ab + bq)", "+ ED[ii, Ur] * (ab - bq)",
     "search: wrong sign of payload energy on the new arc"),
    ("M11", "vrptw/routes.py", "dE = bq * E.ec[:, None]", "dE = 0.0 * E.ec[:, None]",
     "search: forgets extra payload carried on earlier arcs"),
    ("M12", "vrptw/routes.py", "            zz = zz - S[vk] - T[vk][v[k + 1]]", "            zz = zz - T[vk][v[k + 1]]",
     "search: latest-start backward pass forgets service time"),
    ("M13", "vrptw/routes.py", "            if t < E[vk]:\n                w += E[vk] - t\n                t = E[vk]",
     "            if t < E[vk]:\n                w += E[vk] - t",
     "search: forward pass forgets waiting for the window to open"),
    ("M14", "vrptw/routes.py", "        dE = (em.b * qc * route.ecum[k - 1]", "        dE = (0.0 * qc * route.ecum[k - 1]",
     "search: removal savings ignore payload on earlier arcs"),
    ("M15", "vrptw/routes.py", "self.mindur = a[m - 1] - min(z[0], a[0] + w)", "self.mindur = a[m - 1] - z[0]",
     "search: route min-duration formula wrong"),
    ("M16", "vrptw/routes.py", "            W = W + np.maximum(0.0, E[v[k]] - arr)", "            W = W",
     "search: insertion duration check ignores downstream waiting"),
    ("M17", "vrptw/alns.py", "            next_col += 1\n            refresh_empty_cols()", "            next_col += 1",
     "ALNS repair: fleet limit not re-checked after opening a route"),
    ("M18", "vrptw/alns.py", "            C[rem, tcol], P[rem, tcol] = noisy(c), p", "            pass",
     "ALNS repair: stale insertion costs after a route changes"),
    ("M19", "vrptw/alns.py", "            if len(nodes) > 2:\n                new_routes.append(Route(ctx.inst, nodes))",
     "            new_routes.append(Route(ctx.inst, nodes))",
     "ALNS destroy: keeps emptied routes"),
    ("M20", "vrptw/alns.py", "if key[0] < best_key[0] or", "if key[0] > best_key[0] or",
     "ALNS: best solution prefers MORE unassigned customers"),
    ("M21", "vrptw/alns.py", "            reg = diff[:, 1:].sum(axis=1)", "            reg = -diff[:, 1:].sum(axis=1)",
     "ALNS regret repair: inverted regret"),
    ("M22", "vrptw/alns.py", "            found, it = search(ctx, trial, budget_a - used, rng, stats, max_routes=target,",
     "            found, it = search(ctx, trial, budget_a - used, rng, stats, max_routes=None,",
     "ALNS stage A: route minimisation without the fleet limit"),
    ("M23", "vrptw/mip.py", "B.add({tv[j]: 1.0, tv[i]: -1.0, a_: -M}, S[i] + T[i, j] - M, np.inf)",
     "B.add({tv[j]: 1.0, tv[i]: -1.0, a_: -M}, T[i, j] - M, np.inf)",
     "MIP: time propagation ignores service time"),
    ("M24", "vrptw/mip.py", "B.add({rv[i]: 1.0, rv[j]: -1.0, a_: -Q}, q[i] - Q, np.inf)",
     "B.add({rv[i]: 1.0, rv[j]: -1.0, a_: -Q}, -Q, np.inf)",
     "MIP: payload (MTZ) constraint drops q_i"),
    ("M25", "vrptw/mip.py", "rv[j]: -ED[i, j] * em.b, a_: -M}", "rv[j]: ED[i, j] * em.b, a_: -M}",
     "MIP: wrong sign of payload energy"),
    ("M26", "vrptw/mip.py", "em.e_trip + ED[d, jj] * em.a - M, np.inf)", "ED[d, jj] * em.a - M, np.inf)",
     "MIP: forgets the take-off/landing energy"),
    ("M27", "vrptw/sim.py", "        energy += inst.energy_dist[i, j] * (em.a + em.b * payload) * f",
     "        energy += inst.energy_dist[i, j] * (em.a + em.b * payload)",
     "simulator: wind does not affect energy"),
    ("M28", "vrptw/sim.py", "            late = max(0.0, t - inst.due[j])", "            late = max(0.0, t - inst.due[j] - 1.0)",
     "simulator: lateness off by one minute"),
    ("M29", "vrptw/drone.py", "        best[\"free\"] = ret + swap", "        best[\"free\"] = ret",
     "trip chaining: forgets the battery-swap time"),
    ("M30", "vrptw/drone.py", "a = k * self.m_vehicle + 1000.0 * self.p_avionics / self.v_air_kmh",
     "a = k * self.m_vehicle",
     "energy model: drops avionics power"),
]
EQUIVALENT = ("M00", "vrptw/checker.py", "EPS = 1e-6\n", "EPS = 1e-6  # equivalent mutant\n",
              "equivalent mutant (comment only) - must SURVIVE")

HISTORY = ("First run on 2026-09-27: 28/30 killed. Survivors: M21 (inverted regret - no test looked at *which* "
           "customer regret inserts first) and M28 (lateness off by one - the simulator tests only covered zero "
           "lateness). Added `test_regret_repair_beats_greedy_on_constructed_case` and "
           "`test_lateness_is_measured_exactly`, then re-ran everything; the table below is that re-run.")

COPY = ["vrptw", "tests", "scripts", "data", "results", "README.md"]


def make_copy(dst: Path):
    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, dst / name, ignore=shutil.ignore_patterns("__pycache__", "*.log"))
        elif src.exists():
            shutil.copy2(src, dst / name)


def run_tests(workdir: Path, timeout: float):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "--deselect", "tests/test_readme.py::test_readme_matches_results"],
                           cwd=workdir, capture_output=True, text=True, timeout=timeout)
        rc, tail = p.returncode, (p.stdout.strip().splitlines() or [""])[-1]
    except subprocess.TimeoutExpired:
        rc, tail = -1, "timeout"
    return rc, tail, time.time() - t0


def run_mutant(m, timeout):
    mid, f, old, new, desc = m
    with tempfile.TemporaryDirectory(prefix="mut_") as td:
        d = Path(td)
        make_copy(d)
        if mid != "BASE":
            src = (d / f).read_text(encoding="utf-8")
            if src.count(old) != 1:
                return {"id": mid, "file": f, "desc": desc, "status": "INVALID",
                        "detail": f"pattern occurs {src.count(old)} times"}
            (d / f).write_text(src.replace(old, new), encoding="utf-8")
        rc, tail, secs = run_tests(d, timeout)
    status = "survived" if rc == 0 else ("timeout" if rc == -1 else "killed")
    return {"id": mid, "file": f, "desc": desc, "status": status, "detail": tail, "seconds": round(secs, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    base = run_mutant(("BASE", "", "", "", "unmodified copy - must PASS"), a.timeout)
    eq = run_mutant(EQUIVALENT, a.timeout)
    print("baseline:", base["status"], base["detail"])
    print("equivalent mutant:", eq["status"], eq["detail"])
    if base["status"] != "survived" or eq["status"] != "survived":
        sys.exit("harness self-check failed: baseline and equivalent mutant must both pass the tests")
    muts = [m for m in MUTANTS if not a.only or m[0] in a.only]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda m: run_mutant(m, a.timeout), muts))
    for r in res:
        print(r["id"], r["status"], "-", r["desc"], "|", r["detail"])
    killed = sum(r["status"] == "killed" for r in res)
    out = {"baseline": base, "equivalent": eq, "mutants": res, "killed": killed, "total": len(res), "history": HISTORY}
    if a.only:
        print(f"killed {killed}/{len(res)} (partial run, nothing written)")
        return
    write_json(RESULTS / "mutation.json", out)
    lines = ["# Mutation testing", "",
             "Generated by `python scripts/mutation_check.py` (do not edit by hand). Each mutant is one deliberate",
             "bug applied to a fresh copy of the repository; the `pytest` suite is then run (all tests except the",
             "README-table check, which does not depend on the mutated code and would need this file to exist).",
             "*killed* = at least one test failed.", "",
             "Harness self-check (run first):", "",
             f"- unmodified copy: tests **{'pass' if base['status'] == 'survived' else 'FAIL'}** ({base['detail']})",
             f"- equivalent mutant (comment only): **{eq['status']}** - so the harness can report a surviving mutant",
             "", HISTORY, "", f"**Killed: {killed} / {len(res)}**", "",
             "| id | file | mutation | result | first failing summary line |", "|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['id']} | `{r['file']}` | {r['desc']} | {r['status']} | {r['detail'].replace('|', '/')} |")
    with open(ROOT / "MUTATION.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"killed {killed}/{len(res)}")


if __name__ == "__main__":
    main()
