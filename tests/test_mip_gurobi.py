"""Gurobi backend: same model, same audit, same optimum as HiGHS on small instances."""
import math

import pytest

from vrptw.instance import load_solomon
from vrptw.mip import solve_mip

gp = pytest.importorskip("gurobipy")


@pytest.mark.parametrize("name", ["C101", "R101", "RC201", "R202"])
def test_gurobi_matches_highs_on_small_solomon(name):
    inst = load_solomon(name, 10)
    h = solve_mip(inst, time_limit=60)
    g = solve_mip(inst, time_limit=60, solver="gurobi")
    assert h["optimal"] and g["optimal"] and g["valid"]
    assert g["solver"] == "gurobi"
    assert math.isclose(h["objective"], g["objective"], rel_tol=1e-7, abs_tol=1e-6)


def test_unknown_solver_rejected():
    with pytest.raises(ValueError):
        solve_mip(load_solomon("C101", 10), solver="cplex")
