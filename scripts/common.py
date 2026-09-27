"""Shared helpers for experiment scripts."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


def machine_info() -> dict:
    import numpy
    import scipy
    cpu = platform.processor()
    try:
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=30)
            cpu = out.stdout.strip() or cpu
        elif Path("/proc/cpuinfo").exists():
            for ln in Path("/proc/cpuinfo").read_text().splitlines():
                if ln.startswith("model name"):
                    cpu = ln.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {"os": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
