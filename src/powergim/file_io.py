import yaml
import pandas as pd
from . import grid_data


def read_parameters(yaml_file):
    with open(yaml_file, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def read_profiles(filename, timerange):
    profiles = pd.read_csv(filename, sep=None, engine="python")
    profiles = profiles.loc[timerange]
    profiles.index = range(len(timerange))
    return profiles


def read_grid(nodes, branches, generators, consumers):
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

    grid = grid_data.GridData(node=node, branch=branch, generator=generator, consumer=consumer)
    grid.validate_grid_data()
    return grid
