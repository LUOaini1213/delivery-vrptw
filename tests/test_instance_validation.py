"""Reject malformed problem data while retaining valid but infeasible instances."""
from dataclasses import replace

import numpy as np
import pytest

from vrptw.instance import EnergyModel, make_instance, subset


def instance(**updates):
    values = dict(name="validation", coords=[[0, 0], [1, 0]], demand=[0, 1],
                  ready=[0, 0], due=[10, 10], service=[0, 0], capacity=3)
    values.update(updates)
    return make_instance(**values)


@pytest.mark.parametrize("updates", [
    {"capacity": -1}, {"capacity": True}, {"capacity": float("nan")},
    {"speed": 0}, {"speed": -1}, {"speed": float("inf")}, {"speed": True},
    {"n_depots": 0}, {"n_depots": 3}, {"n_depots": 1.0}, {"n_depots": True},
    {"max_vehicles": -1}, {"max_vehicles": 1.5}, {"max_vehicles": True},
    {"max_duration": -1}, {"max_duration": float("nan")}, {"max_duration": False},
    {"vehicle_cost": -1}, {"w_dist": float("nan")}, {"w_energy": float("inf")},
    {"coords": [[0, 0], [float("nan"), 0]]}, {"coords": [0, 1]},
    {"demand": [0, float("nan")]}, {"demand": [0, -1]}, {"demand": [1, 1]},
    {"demand": [0]}, {"demand": [[0], [1]]}, {"demand": [0, True]},
    {"service": [0, -1]}, {"due": [10, float("inf")]},
    {"ready": [0, 11]}, {"ready": [0, "0"]},
])
def test_malformed_data_rejected(updates):
    with pytest.raises(ValueError):
        instance(**updates)


@pytest.mark.parametrize("field,bad", [
    ("dist", np.array([[0, -1], [1, 0]])),
    ("time", np.array([[0, float("nan")], [1, 0]])),
    ("energy_dist", np.zeros((3, 3))),
    ("service", np.array([0, float("inf")])),
    ("capacity", float("nan")),
    ("demand", np.array([0, True], dtype=object)),
])
def test_revalidation_catches_mutated_instance(field, bad):
    inst = instance()
    setattr(inst, field, bad)
    with pytest.raises(ValueError):
        inst.validate()


@pytest.mark.parametrize("updates", [
    {"a": -1}, {"b": float("nan")}, {"e_trip": -1}, {"battery": 0},
    {"battery": float("inf")}, {"reserve": -0.1}, {"reserve": 1.1}, {"a": True},
])
def test_energy_parameters_rejected(updates):
    values = dict(a=1, b=0, e_trip=0, battery=10, reserve=0.2)
    values.update(updates)
    with pytest.raises(ValueError):
        EnergyModel(**values)


def test_zero_demand_and_zero_distance_remain_valid():
    inst = instance(coords=[[0, 0], [0, 0]], demand=[0, 0], max_duration=0, capacity=0,
                    max_vehicles=0, energy=EnergyModel(0, 0, 0, 1, 1), w_dist=0)
    inst.validate()
    assert inst.n_customers == 1


def test_infeasible_demand_and_nonzero_depot_service_are_not_schema_errors():
    inst = instance(demand=[0, 100], service=[3, 0], max_vehicles=0)
    inst.validate()
    assert inst.demand[1] > inst.capacity


def test_asymmetric_travel_times_and_negative_time_origin_supported():
    inst = replace(instance(ready=[-10, -5]), time=np.array([[0, 2], [3, 0]]))
    inst.validate()
    assert inst.time[0, 1] != inst.time[1, 0]


def test_empty_customer_subset_supported():
    inst = subset(instance(), 0)
    assert inst.n_customers == 0
    inst.validate()


@pytest.mark.parametrize("n", [-1, 2, True, 0.5])
def test_subset_rejects_invalid_customer_count(n):
    with pytest.raises(ValueError):
        subset(instance(), n)
