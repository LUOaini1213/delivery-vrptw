"""Fast part of the replay check: stored results must be reproduced by the current code.

The full replay (Solomon ALNS runs, drone plans, simulation rows) takes minutes and runs in CI as
`python scripts/replay_check.py --check`. Here only the cheap cases run, plus checks that the
comparison really reports a perturbed stored value."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_check  # noqa: E402

RES = ROOT / "results"


def test_compare_reports_value_route_and_type_differences():
    stored = {"distance": 828.9369, "routes": [[0, 5, 3, 0]], "ok": True, "seconds": 1.0}
    assert replay_check.compare(stored, dict(stored, seconds=99.0)) == []
    assert replay_check.compare(stored, dict(stored, distance=828.937))
    assert replay_check.compare(stored, dict(stored, routes=[[0, 3, 5, 0]]))
    assert replay_check.compare(stored, dict(stored, routes=[[0, 5, 3, 0], [0, 0]]))
    assert replay_check.compare(stored, dict(stored, ok=1))
    assert replay_check.compare(stored, {k: v for k, v in stored.items() if k != "ok"})


@pytest.mark.skipif(not (RES / "drone.json").exists(), reason="no drone results")
def test_drone_i1_baselines_replay_exactly():
    n, diffs = replay_check.case_drone_i1(RES)
    assert n == len(replay_check.DRONE_I1_SEEDS)
    assert diffs == []


@pytest.mark.skipif(not (RES / "drone.json").exists(), reason="no drone results")
def test_replay_fails_on_a_perturbed_stored_value(tmp_path):
    d = json.loads((RES / "drone.json").read_text(encoding="utf-8"))
    d["large"]["runs"][0]["i1"]["energy_wh"] += 0.001
    (tmp_path / "drone.json").write_text(json.dumps(d), encoding="utf-8")
    _, diffs = replay_check.case_drone_i1(tmp_path)
    assert len(diffs) == 1 and "energy_wh" in diffs[0]
