import pandas as pd
from ruamel.yaml import YAML

from . import grid_data, utils


def read_parameters(yaml_file):
    yaml = YAML(typ="safe")
    with open(yaml_file, "r") as stream:
        data = yaml.load(stream)
    utils.validate_parameter_data(data)
    return data


def read_profiles(filename, timerange=None):
    profiles = pd.read_csv(filename, sep=None, engine="python")
    if timerange is not None:
        profiles = profiles.loc[timerange]
        profiles.index = range(len(timerange))
    return profiles


def read_grid(investment_years, nodes, branches, generators, consumers):
    """Read and validate grid data from input files

    time-series data may be used for
    consumer demand
    generator inflow (e.g. solar and wind)
    generator fuelcost (e.g. one generator with fuelcost = power price)
    """
    node = pd.read_csv(
        nodes,
        dtype={"id": str, "area": str},
    )
    # TODO use integer range index instead of id string, cf powergama
    node.set_index("id", inplace=True)
    node["id"] = node.index
    node.index.name = "index"
    branch = pd.read_csv(
        branches,
        dtype={"node_from": str, "node_to": str},
    )
    generator = pd.read_csv(
        generators,
        dtype={"node": str, "type": str},
    )
    consumer = pd.read_csv(
        consumers,
        dtype={"node": str},
    )

    grid = grid_data.GridData(
        investment_years=investment_years, node=node, branch=branch, generator=generator, consumer=consumer
    )
    grid.validate_grid_data()
    return grid
