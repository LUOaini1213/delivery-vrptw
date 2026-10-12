"""Real, tiny pytest subprocesses verify mutation evidence without product mutations."""
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("mutation_harness_under_test", REPO / "scripts/mutation_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toy(tmp_path, harness, monkeypatch):
    root = tmp_path / "project"
    (root / "vrptw").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "vrptw/__init__.py").write_text("")
    source = root / "vrptw/subject.py"
    source.write_bytes(b"VALUE = 1\r\n")
    tests = root / "tests/test_subject.py"
    tests.write_text("from vrptw.subject import VALUE\ndef test_value():\n    assert VALUE == 1\n")
    monkeypatch.setattr(harness, "ROOT", root)
    return root, source, tests


def mutant(new="VALUE = 2"):
    return ("T01", "vrptw/subject.py", "VALUE = 1", new, "toy changed value")


def test_real_assertion_failure_is_a_kill_and_original_bytes_stay_intact(harness, toy):
    _, source, _ = toy
    original = source.read_bytes()
    result = harness.run_mutant(mutant(), 15)
    assert result["status"] == "killed"
    assert "test_value" in result["detail"]
    assert source.read_bytes() == original


def test_equivalent_behavior_survives(harness, toy):
    result = harness.run_mutant(mutant("VALUE = 1 # same behavior"), 15)
    assert result["status"] == "survived"


@pytest.mark.parametrize("phase", ["collection", "setup", "teardown"])
def test_infrastructure_errors_are_not_kills(harness, toy, phase):
    _, source, tests = toy
    if phase == "collection":
        change = "raise ImportError('synthetic import failure')"
    else:
        body = ("    raise RuntimeError('synthetic setup failure')\n" if phase == "setup" else
                "    yield\n    raise RuntimeError('synthetic teardown failure')\n")
        tests.write_text("import pytest\n@pytest.fixture(autouse=True)\ndef broken():\n" + body +
                         "def test_value():\n    assert True\n")
        change = "VALUE = 2"
    original = source.read_bytes()
    result = harness.run_mutant(mutant(change), 15)
    assert result["status"] == "error"
    assert source.read_bytes() == original


@pytest.mark.parametrize("contents", [
    "# no tests\n",
    "import pytest\n@pytest.mark.skip(reason='toy')\ndef test_skip():\n    assert False\n",
    "import pytest\ndef test_pass():\n    assert True\n@pytest.mark.skip(reason='toy')\ndef test_skip():\n    pass\n",
])
def test_missing_or_skipped_tests_are_errors(harness, toy, contents):
    toy[2].write_text(contents)
    assert harness.run_mutant(mutant(), 15)["status"] == "error"


def test_runtime_exception_in_a_test_body_is_a_real_failure(harness, toy):
    toy[2].write_text("def test_application():\n    raise ValueError('application failure')\n")
    assert harness.run_mutant(mutant(), 15)["status"] == "killed"


def test_expected_application_exception_is_a_passing_test(harness, toy):
    toy[2].write_text("import pytest\ndef test_application():\n    with pytest.raises(ValueError):\n        raise ValueError('expected')\n")
    assert harness.run_mutant(mutant(), 15)["status"] == "survived"


def test_compile_error_is_not_a_kill(harness, toy):
    result = harness.run_mutant(mutant("VALUE = ("), 15)
    assert result["status"] == "error"
    assert "compil" in result["detail"]


def test_timeout_is_not_a_kill(harness, toy):
    result = harness.run_mutant(mutant("import time; time.sleep(10); VALUE = 1"), 0.5)
    assert result["status"] == "error"
    assert "timeout" in result["detail"].lower() or "timed out" in result["detail"].lower()


def test_crash_without_junit_is_not_a_kill(harness, toy):
    result = harness.run_mutant(mutant("import os; os._exit(3)"), 15)
    assert result["status"] == "error"


def test_external_pytest_filter_cannot_remove_the_baseline(harness, toy, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nonexistent_test_name")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_synthetic_plugin")
    base = ("BASE", "", "", "", "unmodified")
    assert harness.run_mutant(base, 15)["status"] == "survived"


def cli_project(tmp_path):
    root = tmp_path / "cli-project"
    (root / "scripts").mkdir(parents=True)
    (root / "vrptw").mkdir()
    (root / "tests").mkdir()
    for filename in ("mutation_check.py", "common.py"):
        shutil.copy2(REPO / "scripts" / filename, root / "scripts" / filename)
    (root / "vrptw/__init__.py").write_text("")
    (root / "vrptw/checker.py").write_text(
        "EPS = 1e-6\ndef late(starts, pos, inst, node):\n"
        "    if starts[pos] > float(inst.due[node]) + EPS:\n        return True\n    return False\n")
    (root / "tests/test_toy.py").write_text("def test_pass():\n    assert True\n")
    return root


def cli(root, *args):
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run([sys.executable, str(root / "scripts/mutation_check.py"),
                           "--workers", "1", "--timeout", "15", *args],
                          cwd=root, env=env, capture_output=True, text=True, timeout=60)


def test_cli_survivor_has_a_nonzero_exit(tmp_path):
    root = cli_project(tmp_path)
    result = cli(root, "--only", "M01")
    assert result.returncode == 1, result.stdout + result.stderr
    reports = list((root / "results/mutation-runs").glob("*/mutation.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["outcome"] == "survivors"
    assert report["baseline"]["identities"] == report["equivalent"]["identities"]
    assert report["mutants"][0]["status"] == "survived"
    assert report["completed"] == 1 and not report["incomplete"] and report["unrun_ids"] == []


def test_cli_kill_keeps_logs_junit_and_historical_bytes(tmp_path):
    root = cli_project(tmp_path)
    (root / "tests/test_toy.py").write_text(
        "from types import SimpleNamespace\nfrom vrptw.checker import late\n"
        "def test_fractional_lateness():\n    assert late([0.25], 0, SimpleNamespace(due=[0]), 0)\n")
    (root / "results").mkdir()
    historical = {root / "results/mutation.json": b'{"historical": true}\r\n',
                  root / "MUTATION.md": b'historical markdown\r\n',
                  root / "vrptw/checker.py": (root / "vrptw/checker.py").read_bytes()}
    for path, raw in historical.items():
        path.write_bytes(raw)
    output = root / "results/new-evidence"
    result = cli(root, "--only", "M01", "--output-dir", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "mutation.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "passed" and report["partial"]
    assert report["killed"] == report["total"] == 1
    assert report["completed"] == 1 and not report["incomplete"] and report["unrun_ids"] == []
    assert report["source_bytes_unchanged"]
    assert report["baseline"]["identities"] == report["equivalent"]["identities"]
    for row in [report["baseline"], report["equivalent"], *report["mutants"]]:
        assert Path(row["log"]).is_file()
        cases = ET.parse(row["junit"]).findall(".//testcase")
        assert len(cases) == 1
        assert (cases[0].find("failure") is not None) == (row["id"] == "M01")
    for path, raw in historical.items():
        assert path.read_bytes() == raw
    prior = (output / "mutation.json").read_bytes()
    again = cli(root, "--only", "M01", "--output-dir", str(output))
    assert again.returncode == 2
    assert (output / "mutation.json").read_bytes() == prior


@pytest.mark.parametrize("args", [("--only", "UNKNOWN"), ("--only",), ("--only", "M01", "M01")])
def test_cli_invalid_selection_fails_before_baseline(tmp_path, args):
    result = cli(cli_project(tmp_path), *args)
    assert result.returncode == 2
    assert "baseline:" not in result.stdout


@pytest.mark.parametrize("new", ["VALUE = (", "VALUE = 1"])
def test_preflight_compile_or_noop_error_never_starts_pytest(harness, toy, monkeypatch, new):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid mutant reached pytest")
    monkeypatch.setattr(harness, "run_tests", unexpected)
    assert harness.run_mutant(mutant(new), 15)["status"] == "error"


@pytest.mark.parametrize("old", ["MISSING = 7", "VALUE"])
def test_anchor_must_occur_once_before_any_write(harness, toy, old):
    root, source, _ = toy
    if old == "VALUE":
        source.write_bytes(b"VALUE = 1\r\n# VALUE again\r\n")
    raw = source.read_bytes()
    with pytest.raises(harness.HarnessError, match="exactly once"):
        harness.preflight(root, [("T01", "vrptw/subject.py", old, "CHANGED", "toy")])
    assert source.read_bytes() == raw


def test_multiline_replacement_preserves_crlf_in_the_copy(harness, toy):
    root, source, _ = toy
    source.write_bytes(b"VALUE = 1\r\nOTHER = 2\r\n")
    _, raw, changed = harness.replacement(root, ("T01", "vrptw/subject.py", "VALUE = 1\nOTHER = 2",
                                              "VALUE = 2\nOTHER = 2", "toy"))
    assert changed == b"VALUE = 2\r\nOTHER = 2\r\n"
    assert source.read_bytes() == raw


def test_unchecked_stale_bytecode_is_not_used(harness, toy):
    py_compile.compile(str(toy[1]), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    assert harness.run_mutant(mutant(), 15)["status"] == "killed"


def test_equivalent_control_must_match_baseline_test_identities(harness, toy, tmp_path):
    result = harness.run_mutant(mutant("VALUE = 1 # equivalent"), 15,
                                expected_identities=["different::test_name"], output_dir=tmp_path / "evidence")
    assert result["status"] == "error"
    assert "identities" in result["detail"]


@pytest.mark.parametrize("returncode,body", [
    (0, '<testsuites><testsuite><testcase name="fake"><failure message="failure"/></testcase></testsuite></testsuites>'),
    (1, '<testsuites><testsuite><testcase name="fake"/></testsuite></testsuites>'),
    (2, '<testsuites><testsuite><testcase name="fake"><failure/></testcase></testsuite></testsuites>'),
    (0, '<testsuites tests="2"><testsuite><testcase name="fake"/></testsuite></testsuites>'),
    (0, '<testsuites><testsuite tests="2"><testcase name="fake"/></testsuite></testsuites>'),
    (0, '<not-junit/>'),
    (0, None),
])
def test_process_and_junit_disagreement_is_never_a_kill(harness, toy, monkeypatch, returncode, body):
    def fake_run(command, **kwargs):
        report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        if body is not None:
            report.write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, stdout="synthetic output", stderr="")
    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    result = harness.run_mutant(mutant(), 15)
    assert result["status"] == "error"
    assert "not a kill" in result["detail"]


def test_subprocess_launch_error_is_not_a_kill(harness, toy, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise OSError("synthetic launch failure")
    monkeypatch.setattr(harness.subprocess, "run", cannot_start)
    assert harness.run_mutant(mutant(), 15)["status"] == "error"


def test_report_directories_are_excluded_from_fresh_copies(harness, toy, tmp_path):
    root = toy[0]
    fresh = root / "results/review-output"
    default = root / "results/mutation-runs/old"
    for directory in (fresh, default):
        directory.mkdir(parents=True)
        (directory / "sentinel.json").write_text("{}")
    (root / "results/legacy.json").write_text("{}")
    destination = tmp_path / "copy"
    destination.mkdir()
    harness.make_copy(destination, root=root, excluded_paths=[fresh])
    assert (destination / "results/legacy.json").is_file()
    assert not (destination / "results/review-output").exists()
    assert not (destination / "results/mutation-runs").exists()


@pytest.mark.parametrize("status", ["survived", "error"])
@pytest.mark.parametrize("workers", [1, 2])
def test_bad_batch_stops_new_submissions_and_finishes_active_copies(harness, status, workers):
    started, finished = [], []
    barrier = threading.Barrier(workers)

    def run_one(index):
        started.append(index)
        barrier.wait(timeout=5)  # Every active copy must have started before this batch returns.
        finished.append(index)
        return {"id": index, "status": status if index == 0 else "killed"}

    results = list(harness.run_batches(list(range(6)), workers, run_one))
    assert sorted(started) == sorted(finished) == list(range(workers))
    assert [row["id"] for row in results] == list(range(workers))


def test_successful_batches_execute_every_selected_mutant_once(harness):
    calls = []

    def run_one(index):
        calls.append(index)
        return {"id": index, "status": "killed"}

    results = list(harness.run_batches(list(range(5)), 2, run_one))
    assert sorted(calls) == list(range(5))
    assert [row["id"] for row in results] == list(range(5))


def test_stopped_selection_reports_unrun_ids_and_cannot_pass(harness, toy, monkeypatch, tmp_path):
    selected = [mutant("VALUE = 2"), ("T02", "vrptw/subject.py", "VALUE = 1", "VALUE = 3", "second")]
    monkeypatch.setattr(harness, "MUTANTS", selected)
    monkeypatch.setattr(harness, "EQUIVALENT", ("M00", "vrptw/subject.py", "VALUE = 1", "VALUE = 1 # same", "equivalent"))
    toy[2].write_text("def test_pass():\n    assert True\n")
    output = tmp_path / "evidence"
    assert harness.main(["--workers", "1", "--timeout", "15", "--output-dir", str(output)]) == 1
    report = json.loads((output / "mutation.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "survivors" and report["incomplete"]
    assert report["completed"] == 1 and report["total"] == 2
    assert report["unrun_ids"] == ["T02"]
    assert not (output / "T02").exists()


def test_child_process_forces_utf8_even_with_an_ascii_parent(harness, toy, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("PYTHONUTF8", "0")
    toy[2].write_text("import os\ndef test_unicode():\n"
                      "    assert os.environ['PYTHONIOENCODING'] == 'utf-8'\n"
                      "    assert os.environ['PYTHONUTF8'] == '1'\n"
                      "    print('\\u6c49\\u5b57')\n")
    assert harness.run_mutant(mutant("VALUE = 1 # equivalent"), 15)["status"] == "survived"
