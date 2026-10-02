"""Search caches must describe the current problem, not old measurements."""
from dataclasses import replace

import numpy as np
import pytest

from vrptw.checker import evaluate
from vrptw.instance import make_instance
from vrptw.routes import Route


def make(demand=None):
    return make_instance("cache", [[0, 0], [1, 0]], [0, 1] if demand is None else demand,
                         [0, 0], [10, 10], [0, 0], capacity=3)


def test_constructor_owns_snapshot_and_does_not_freeze_callers_array():
    source = np.array([0., 1.])
    inst = make(source)
    source[1] = 7
    assert inst.demand[1] == 1
    assert source.flags.writeable
    with pytest.raises(ValueError):
        inst.demand[1] = 7


@pytest.mark.parametrize("field,value", [
    ("demand", [0., 5.]), ("ready", [0., 9.5]), ("due", [10., .5]),
    ("service", [0., 10.]), ("time", [[0., 8.], [8., 0.]]),
])
def test_mutable_replacement_and_subsequent_edit_never_use_old_cache(field, value):
    inst = make()
    assert Route(inst, [0, 1, 0]).feasible(inst)
    original = np.array(getattr(inst, field), copy=True)
    setattr(inst, field, original)
    assert Route(inst, [0, 1, 0]).feasible(inst)
    original[:] = value
    route = Route(inst, [0, 1, 0])
    assert not route.feasible(inst)
    assert not evaluate(inst, [route.nodes]).feasible


def test_new_readonly_vector_invalidates_existing_cache():
    inst = make()
    assert Route(inst, [0, 1, 0]).load == 1
    demand = np.array([0., 5.])
    demand.flags.writeable = False
    inst.demand = demand
    assert Route(inst, [0, 1, 0]).load == 5


def test_readonly_view_with_mutable_backing_cannot_be_cached():
    inst = make()
    backing = np.array([0., 1.])
    inst.demand = backing.view()
    inst.demand.flags.writeable = False
    assert Route(inst, [0, 1, 0]).load == 1
    backing[1] = 5
    inst.validate()
    assert Route(inst, [0, 1, 0]).load == 5
    assert not Route(inst, [0, 1, 0]).feasible(inst)


def test_replace_preserves_original_and_refreshes_cached_data():
    inst = make()
    Route(inst, [0, 1, 0])
    changed = replace(inst, demand=np.array([0., 5.]))
    assert Route(changed, [0, 1, 0]).load == 5
    assert Route(inst, [0, 1, 0]).load == 1
