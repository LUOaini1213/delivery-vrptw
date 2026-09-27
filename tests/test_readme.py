"""README tables must be exactly what scripts/check_readme.py generates from results/*.json,
and the committed result files must be internally consistent."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vrptw.checker import evaluate
from vrptw.instance import load_solomon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def test_readme_matches_results():
    p = subprocess.run([sys.executable, "scripts/check_readme.py", "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]


@pytest.mark.skipif(not (RES / "solomon100.json").exists(), reason="no Solomon results")
def test_committed_solomon_routes_are_feasible_and_match_reported_numbers():
    d = json.loads((RES / "solomon100.json").read_text(encoding="utf-8"))
    assert len({r["instance"] for r in d["runs"]}) == 56
    for r in d["runs"]:
        inst = load_solomon(r["instance"])
        ev = evaluate(inst, r["routes"])
        assert ev.feasible, (r["instance"], ev.violations[:2])
        assert ev.n_routes == r["alns"]["vehicles"]
        assert round(ev.distance, 4) == pytest.approx(r["alns"]["distance"], abs=1e-4)


@pytest.mark.skipif(not (RES / "drone.json").exists(), reason="no drone results")
def test_committed_drone_routes_are_feasible():
    from vrptw.drone import DroneSpec, generate
    d = json.loads((RES / "drone.json").read_text(encoding="utf-8"))
    assert d["spec"] == DroneSpec().__dict__
    for r in d["large"]["runs"]:
        inst = generate(r["instance_seed"], n=d["large"]["n"], n_depots=d["large"]["depots"])
        ev = evaluate(inst, r["routes"])
        assert ev.feasible, ev.violations[:2]
        assert round(ev.energy, 3) == pytest.approx(r["alns"]["energy_wh"], abs=1e-3)
