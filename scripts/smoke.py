"""CI smoke run: a few small instances with short budgets; asserts feasibility and that no
heuristic solution beats a proven MIP optimum. Writes nothing."""
from __future__ import annotations

import time

from common import ROOT  # noqa: F401  (puts the repo on sys.path)

from vrptw.alns import solve
from vrptw.checker import evaluate
from vrptw.construct import best_i1
from vrptw.drone import DroneSpec, chain_trips, generate
from vrptw.instance import load_solomon
from vrptw.mip import solve_mip
from vrptw.sim import execute_plan, make_scenario, metrics


def check(inst, label, iters=200):
    t0 = time.perf_counter()
    m = solve_mip(inst, time_limit=60)
    s0 = best_i1(inst)
    sol, _ = solve(inst, s0, iters=iters, seed=1)
    ev = evaluate(inst, sol.as_lists())
    assert ev.feasible, (label, ev.violations[:3])
    assert evaluate(inst, s0.as_lists()).feasible
    if m["optimal"]:
        assert evaluate(inst, m["routes"]).feasible
        assert ev.cost >= m["objective"] - 1e-6 * max(1.0, abs(m["objective"])), label
    print(f"{label}: MIP {'opt' if m['optimal'] else 'limit'} {m['objective']:.3f}  ALNS {ev.cost:.3f}  "
          f"I1 {evaluate(inst, s0.as_lists()).cost:.3f}  ({time.perf_counter() - t0:.1f}s)")
    return sol


def main():
    for name in ("C101", "R101", "RC201"):
        check(load_solomon(name, 15), f"{name}.15")
    spec = DroneSpec()
    inst = generate(2, n=10, n_depots=2, spec=spec, horizon=120)
    check(inst, "drone n=10")
    big = generate(1, n=30, n_depots=2, spec=spec, horizon=240)
    sol, _ = solve(big, best_i1(big), iters=100, seed=1)
    routes = sol.as_lists()
    drones = chain_trips(big, routes, spec.swap_min)
    ms = [metrics(big, *execute_plan(big, make_scenario(big, s), routes, drones, spec.swap_min), len(drones), 0.0)
          for s in range(20)]
    on_time = sum(m["on_time"] for m in ms) / len(ms)
    assert 0.0 < on_time <= 1.0
    print(f"simulation: 20 scenarios, mean on-time {on_time:.3f}, drones {len(drones)}")
    print("smoke ok")


if __name__ == "__main__":
    main()
