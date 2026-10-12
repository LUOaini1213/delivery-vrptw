"""Mutation testing: apply deliberate bugs one at a time and check that the test suite fails.

Before the real mutants, the harness checks itself:
  * the unmodified copy must PASS (otherwise every mutant would look "killed");
  * an equivalent mutant (adds a comment) must SURVIVE (proves the harness can report survival).

    python scripts/mutation_check.py --workers 4 --output-dir results/review-mutation
    python scripts/mutation_check.py --self-test

Only JUnit test failures kill a mutant. Collection/setup errors, skipped or missing
tests, crashes, timeouts and contradictory reports are errors. Product sources are
never edited: every run gets a fresh copy, and evidence is saved outside that copy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

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
    ("M31", "vrptw/checker.py", "if not 0 <= node < inst.n_nodes:", "if not node < inst.n_nodes:",
     "checker: accepts negative node IDs"),
    ("M32", "vrptw/checker.py", "if isinstance(node, bool) or not isinstance(node, Integral):",
     "if not isinstance(node, Integral):", "checker: accepts boolean node IDs"),
    ("M33", "vrptw/checker.py", "return Evaluation(not viol, n, dist, energy, cost, viol, r_energy, r_end)",
     "return Evaluation(not viol, sum(1 for route in routes if len(list(route)) >= 3), dist, energy, cost, viol, r_energy, r_end)",
     "checker: counts routes by consuming the input a second time"),
    ("M34", "vrptw/mip.py", "B.add(net, 1, 1)", "B.add(net, 0, 0)",
     "MIP: customers do not consume connectivity flow"),
    ("M35", "vrptw/mip.py", " or not _same_cost(reported, checked.cost)", "",
     "MIP: reported objective is not independently audited"),
    ("M36", "vrptw/instance.py", " or not np.isfinite(value)", "",
     "instance: accepts nonfinite scalar model fields"),
    ("M37", "vrptw/routes.py", "cacheable = all(not a.flags.writeable and a.flags.owndata for a in arrays)",
     "cacheable = True", "search: reuses stale vector caches after in-place data changes"),
    ("M38", "vrptw/construct.py", "for d in feasible_depots[u]:", "for d in feasible_depots[u][:1]:",
     "construction: ignores feasible alternative depots when the nearest fleet is full"),
    ("M39", "vrptw/drone.py", "d = next((d for d in depots\n", "d = next((d for d in depots[:1]\n",
     "singleton completion: ignores feasible alternative depots"),
    ("M40", "vrptw/alns.py", "sol = Solution([Route(inst, nodes) for nodes in initial_routes], start.unassigned)",
     "sol = start.copy()", "ALNS: keeps initial routes' stale caches after the model changes"),
]
EQUIVALENT = ("M00", "vrptw/checker.py", "EPS = 1e-6\n", "EPS = 1e-6  # equivalent mutant\n",
              "equivalent mutant (comment only) - must SURVIVE")

HISTORY = ("First run on 2026-09-27: 28/30 killed. Survivors: M21 (inverted regret - no test looked at *which* "
           "customer regret inserts first) and M28 (lateness off by one - the simulator tests only covered zero "
           "lateness). Added `test_regret_repair_beats_greedy_on_constructed_case` and "
           "`test_lateness_is_measured_exactly`, then re-ran everything; the table below is that re-run.")

COPY = ["vrptw", "tests", "scripts", "data", "results", "README.md", "MUTATION.md",
        "pyproject.toml", "pytest.ini", "setup.cfg", "conftest.py"]
BASELINE = ("BASE", "", "", "", "unmodified copy - must PASS")


class HarnessError(RuntimeError):
    """No valid evidence of a mutation kill was obtained."""


def replacement(root: Path, mutant):
    _, filename, old, new, _ = mutant
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()):
        raise HarnessError(f"mutation target escapes the source root: {filename}")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    old = old.replace("\r\n", "\n").replace("\n", newline)
    new = new.replace("\r\n", "\n").replace("\n", newline)
    if not old or old == new or text.count(old) != 1:
        raise HarnessError(f"{filename}: anchor must change and occur exactly once (found {text.count(old)})")
    changed = text.replace(old, new).encode("utf-8")
    try:
        compile(changed, filename, "exec")  # Only compile the changed file; do not import or write bytecode.
    except (SyntaxError, ValueError) as exc:
        raise HarnessError(f"{filename}: mutant does not compile; not a kill: {exc}") from exc
    return path, raw, changed


def preflight(root: Path, mutants):
    ids = [m[0] for m in mutants]
    if len(ids) != len(set(ids)):
        raise HarnessError("duplicate mutation IDs")
    for mutant in mutants:
        if not mutant[0] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in mutant[0]):
            raise HarnessError("mutation IDs must be safe directory names")
        replacement(root, mutant)


def make_copy(dst: Path, *, root=None, excluded_paths=()):
    root = Path(root or ROOT).resolve()
    excluded = [root / "results" / "mutation-runs", *(Path(p).resolve() for p in excluded_paths)]

    def ignore(directory, names):
        ignored = []
        for name in names:
            candidate = (Path(directory) / name).resolve()
            if (name in {"__pycache__", ".pytest_cache"} or name.endswith((".pyc", ".pyo", ".log"))
                    or any(candidate == p or candidate.is_relative_to(p) for p in excluded)):
                ignored.append(name)
        return ignored

    for name in COPY:
        src = root / name
        if name in ignore(root, [name]):
            continue
        if src.is_dir():
            shutil.copytree(src, dst / name, ignore=ignore)
        elif src.exists():
            shutil.copy2(src, dst / name)


def classify_report(report: Path, returncode: int):
    try:
        document = ET.parse(report).getroot()
    except (OSError, ET.ParseError) as exc:
        raise HarnessError(f"pytest exited {returncode} without a valid JUnit report; not a kill") from exc
    if document.tag not in {"testsuites", "testsuite"}:
        raise HarnessError("unrecognized JUnit report; not a kill")
    cases = list(document.iter("testcase"))
    errors = list(document.iter("error"))
    skipped = list(document.iter("skipped"))
    failures = list(document.iter("failure"))
    for suite in document.iter():
        if suite.tag not in {"testsuite", "testsuites"}:
            continue
        counts = {"tests": len(list(suite.iter("testcase"))), "errors": len(list(suite.iter("error"))),
                  "failures": len(list(suite.iter("failure"))), "skipped": len(list(suite.iter("skipped")))}
        for name, count in counts.items():
            if name in suite.attrib and suite.get(name) != str(count):
                raise HarnessError(f"JUnit {name} count disagrees with test records; not a kill")
    if errors or skipped or not cases or returncode not in (0, 1):
        raise HarnessError(f"pytest exit={returncode}, cases={len(cases)}, errors={len(errors)}, "
                           f"skipped={len(skipped)}; not a kill")
    if (returncode == 1) != bool(failures):
        raise HarnessError("pytest exit/report disagree; not a kill")
    identities, evidence = [], []
    for case in cases:
        if not case.get("name"):
            raise HarnessError("JUnit testcase has no identity; not a kill")
        identity = f"{case.get('classname', '')}::{case.get('name')}"
        identities.append(identity)
        failure = case.find("failure")
        if failure is not None:
            message = " ".join((failure.get("message") or failure.text or "test failure").split())
            evidence.append({"test": identity, "message": message[:1000]})
    if len(evidence) != len(failures):
        raise HarnessError("JUnit failure records are not associated with testcases; not a kill")
    return {"status": "killed" if evidence else "survived", "returncode": returncode,
            "tests": len(cases), "skipped": 0, "identities": sorted(identities), "failures": evidence,
            "detail": (f"{evidence[0]['test']}: {evidence[0]['message']}" if evidence else f"{len(cases)} passed, 0 skipped")}


def _text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")


def run_tests(workdir: Path, timeout: float, output_dir: Path, *, tests="tests", fail_fast=False):
    output_dir.mkdir(parents=True, exist_ok=False)
    report, log = output_dir / "pytest.xml", output_dir / "pytest.log"
    env = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(workdir), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPYCACHEPREFIX=str(output_dir / "pycache"), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               f"--junitxml={report}", str(tests)]
    if fail_fast:
        command.append("-x")
    try:
        result = subprocess.run(command, cwd=workdir, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        log.write_text(_text(exc.stdout) + _text(exc.stderr), encoding="utf-8")
        raise HarnessError(f"pytest timed out after {timeout}s; not a kill") from exc
    except OSError as exc:
        log.write_text(str(exc), encoding="utf-8")
        raise HarnessError(f"pytest could not start; not a kill: {exc}") from exc
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    verdict = classify_report(report, result.returncode)
    verdict.update(junit=str(report), log=str(log))
    return verdict


def run_mutant(m, timeout, *, root=None, output_dir=None, expected_identities=None, excluded_paths=()):
    root = Path(root or ROOT).resolve()
    mid, filename, _, _, desc = m
    started = time.monotonic()
    result = {"id": mid, "file": filename, "desc": desc, "status": "error"}
    # An omitted output directory is useful for tiny in-memory harness tests only.
    with tempfile.TemporaryDirectory(prefix="vrptw-mut-") as scratch:
        scratch = Path(scratch)
        workdir = scratch / "copy"
        workdir.mkdir()
        evidence_dir = Path(output_dir).resolve() if output_dir is not None else scratch / "evidence"
        result.update(junit=str(evidence_dir / "pytest.xml"), log=str(evidence_dir / "pytest.log"),
                      evidence_persisted=output_dir is not None)
        try:
            make_copy(workdir, root=root, excluded_paths=[evidence_dir, *excluded_paths])
            if mid != "BASE":
                path, raw, changed = replacement(workdir, m)
                result.update(original_sha256=hashlib.sha256(raw).hexdigest(),
                              mutated_sha256=hashlib.sha256(changed).hexdigest())
                path.write_bytes(changed)
            verdict = run_tests(workdir, timeout, evidence_dir, fail_fast=mid not in {"BASE", "M00"})
            if (expected_identities is not None and verdict["status"] == "survived"
                    and verdict["identities"] != expected_identities):
                raise HarnessError("passing test identities differ from the baseline; not a valid survivor")
            result.update(verdict)
        except (HarnessError, OSError, UnicodeError) as exc:
            result.update(status="error", detail=str(exc))
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def fingerprints(root):
    paths = sorted(path for directory in ("vrptw", "tests", "scripts")
                   for path in (root / directory).rglob("*.py"))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_batches(mutants, workers, run_one):
    """Finish the active batch, then stop after any survivor or harness error."""
    with ThreadPoolExecutor(workers) as executor:
        for start in range(0, len(mutants), workers):
            # Submit only this batch: map over the whole selection eagerly queues it all.
            futures = [executor.submit(run_one, m) for m in mutants[start:start + workers]]
            batch = [future.result() for future in futures]
            yield from batch
            if any(result["status"] != "killed" for result in batch):
                break


def save_report(directory: Path, report):
    target = directory / "mutation.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    lines = ["# Mutation testing", "", f"Outcome: **{report['outcome']}**; exit code {report['exit_code']}.",
             "Full pytest suite in fresh copies. Only actual JUnit test failures count as kills.",
             "Skipped/missing tests, collection/setup errors, timeouts and report disagreement are errors.",
             "Baseline and equivalent controls must pass with identical test identities.", "",
             f"Selected mutants: {len(report['selected_ids'])}; killed: {report['killed']}.",
             f"Completed: {report['completed']}; unrun: {len(report['unrun_ids'])}; incomplete: {report['incomplete']}.",
             f"Partial selection: {report['partial']}.", ""]
    if report.get("error"):
        lines.extend(["Error: " + report["error"], ""])
    lines.extend(["| id | result | evidence |", "|---|---|---|"])
    for row in [report.get("baseline"), report.get("equivalent"), *report["mutants"]]:
        if row:
            detail = row["detail"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {row['id']} | {row['status']} | {detail} |")
    (directory / "MUTATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test(timeout):
    with tempfile.TemporaryDirectory(prefix="vrptw-harness-self-test-") as td:
        result = run_tests(ROOT, timeout, Path(td) / "evidence", tests="tests/test_mutation_harness.py")
        if result["status"] != "survived":
            raise HarnessError("harness self-test failed: " + result["detail"])
        print(f"harness self-test: {result['tests']} passed, 0 skipped", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--output-dir", type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    a = ap.parse_args(argv)
    if a.workers <= 0 or not math.isfinite(a.timeout) or a.timeout <= 0:
        ap.error("workers and timeout must be positive and finite")
    known = {m[0] for m in MUTANTS}
    if a.only is not None and (len(a.only) != len(set(a.only)) or set(a.only) - known):
        ap.error("--only requires unique known mutant IDs")
    if a.self_test:
        if a.only or a.output_dir:
            ap.error("--self-test cannot be combined with --only or --output-dir")
        return self_test(a.timeout)
    mutants = [m for m in MUTANTS if a.only is None or m[0] in a.only]
    if not mutants:
        ap.error("no mutants selected")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (a.output_dir or ROOT / "results/mutation-runs" / f"{stamp}-{uuid4().hex[:8]}").resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        ap.error("output directory already exists; choose a new directory to preserve evidence")
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable, "source_root": str(ROOT.resolve()),
              "output_dir": str(output_dir), "selected_ids": [m[0] for m in mutants],
              "partial": len(mutants) != len(MUTANTS), "workers": a.workers, "timeout": a.timeout,
              "suite": "tests (all tests; no deselection)", "baseline": None, "equivalent": None,
              "mutants": [], "killed": 0, "total": len(mutants), "history": HISTORY,
              "completed": 0, "incomplete": True, "unrun_ids": [m[0] for m in mutants],
              "outcome": "error", "exit_code": 2}
    original_hashes = fingerprints(ROOT)
    report["source_and_test_sha256"] = original_hashes
    try:
        preflight(ROOT, [EQUIVALENT, *mutants])
        if a.preflight:
            report.update(outcome="preflight_passed", exit_code=0)
        else:
            base = run_mutant(BASELINE, a.timeout, output_dir=output_dir / "BASE", excluded_paths=[output_dir])
            report["baseline"] = base
            print("baseline:", base["status"], base["detail"], flush=True)
            if base["status"] != "survived":
                raise HarnessError("unmodified baseline must pass without skips")
            equivalent = run_mutant(EQUIVALENT, a.timeout, output_dir=output_dir / "M00",
                                    expected_identities=base["identities"], excluded_paths=[output_dir])
            report["equivalent"] = equivalent
            print("equivalent mutant:", equivalent["status"], equivalent["detail"], flush=True)
            if equivalent["status"] != "survived":
                raise HarnessError("equivalent control must pass the same tests as the baseline")
            results = run_batches(mutants, a.workers,
                                  lambda m: run_mutant(m, a.timeout, output_dir=output_dir / m[0],
                                                       expected_identities=base["identities"],
                                                       excluded_paths=[output_dir]))
            for result in results:
                report["mutants"].append(result)
                print(result["id"], result["status"], result["detail"], flush=True)
            errors = any(r["status"] == "error" for r in report["mutants"])
            survived = any(r["status"] == "survived" for r in report["mutants"])
            report.update(outcome="error" if errors else "survivors" if survived else "passed",
                          exit_code=2 if errors else 1 if survived else 0)
    except (HarnessError, OSError, UnicodeError) as exc:
        report["error"] = str(exc)
    finally:
        completed_ids = {r["id"] for r in report["mutants"]}
        report.update(completed=len(report["mutants"]),
                      incomplete=len(report["mutants"]) != len(mutants),
                      unrun_ids=[m[0] for m in mutants if m[0] not in completed_ids],
                      killed=sum(r["status"] == "killed" for r in report["mutants"]))
        if report["outcome"] == "passed" and report["incomplete"]:
            report.update(outcome="error", exit_code=2, error="selected mutants were not all executed")
        report["source_bytes_unchanged"] = fingerprints(ROOT) == original_hashes
        if not report["source_bytes_unchanged"]:
            report.update(outcome="error", exit_code=2, error="source/test bytes changed during the run")
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report(output_dir, report)
    print(f"killed {report['killed']}/{len(mutants)}; outcome={report['outcome']}; report={output_dir}", flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HarnessError, OSError) as exc:
        print(f"MUTATION ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
