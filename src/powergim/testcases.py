import numpy as np
import pandas as pd

from powergim.grid_data import GridData as grid

from . import utils


def create_case(investment_years, number_nodes, number_timesteps, base_MW=200):
    """Create test case of a given size"""

    grid_data = grid(investment_years, node=None, branch=None, generator=None, consumer=None)
    cols_node = grid_data.keys_sipdata["node"].keys()
    cols_branch = grid_data.keys_sipdata["branch"].keys()
    cols_generator = grid_data.keys_sipdata["generator"].keys()
    cols_consumer = grid_data.keys_sipdata["consumer"].keys()
    nodes = pd.DataFrame(columns=cols_node)
    radius = 5
    node_numbers = np.array(range(number_nodes))
    nodes["id"] = [f"n{i}" for i in node_numbers]
    nodes["area"] = "area"
    nodes["offshore"] = [i % 2 for i in node_numbers]
    nodes["type"] = "nodetype1"
    nodes["capacity"] = 0
    nodes["cost_scaling"] = 1
    nodes["lat"] = 60 + radius * np.sin(np.pi * 2 / number_nodes * node_numbers)
    nodes["lon"] = 5 + radius * np.cos(np.pi * 2 / number_nodes * node_numbers)
    nodes.set_index("id", drop=False, inplace=True)
    nodes.index.name = "index"  # to avoid having "id" both as index name and column name

    # num_odd = number_nodes // 2
    # num_even = number_nodes - num_odd  # = num_odd or num_odd+1

    branches = pd.DataFrame(columns=cols_branch)
    branches["node_from"] = [f"n{i}" for i in node_numbers] * 2
    branches["node_to"] = [f"n{(i+1)%number_nodes}" for i in node_numbers] + [
        f"n{(i+2)%number_nodes}" for i in node_numbers
    ]
    for year in investment_years:
        branches[f"capacity_{year}"] = 0
        branches[f"expand_{year}"] = 1
    branches["max_newCap"] = -1
    branches["distance"] = -1
    branches["cost_scaling"] = 1
    branches["type"] = "branchtype1"

    consumers = pd.DataFrame(columns=cols_consumer)
    # consumers at odd number nodes
    consumers["node"] = [f"n{i+1}" for i in range(0, number_nodes - 1, 2)]
    for year in investment_years:
        if year == investment_years[0]:
            consumers[f"demand_{year}"] = base_MW * len(investment_years)  # must be high enough
        else:
            consumers[f"demand_{year}"] = 0
    consumers["demand_ref"] = "demand1"

    generators = pd.DataFrame(columns=cols_generator)
    # generators at odd number nodes
    generators["node"] = [f"n{i}" for i in range(0, number_nodes, 2)]
    generators["type"] = [f"gentype{i%2}" for i in range(generators.shape[0])]
    generators["desc"] = "generator"
    # more than enough expensive generation:
    for year in investment_years:
        generators[f"expand_{year}"] = 0
        # res capacity increasing
        generators.loc[generators.index % 2 == 0, f"capacity_{year}"] = base_MW
        # expensive capacity not chaning
        clip = np.clip(1 + investment_years[0] - year, a_min=0, a_max=1)
        generators.loc[generators.index % 2 != 0, f"capacity_{year}"] = 3 * base_MW * len(investment_years) * clip

    generators["pmin"] = 0
    generators["cost_scaling"] = 1
    for year in investment_years:
        generators[f"fuelcost_{year}"] = [10 * (i % 2) for i in range(generators.shape[0])] + generators.index / 2
    generators["fuelcost_ref"] = "fuelcost_" + generators["type"]
    generators["pavg"] = 0
    generators["inflow_fac"] = 1
    generators["inflow_ref"] = "inflow_" + generators["type"]
    generators["allow_curtailment"] = 1

    timesteps = np.array(range(number_timesteps))
    profiles = pd.DataFrame(index=timesteps)
    profiles["demand1"] = 1.0
    all_gentypes = generators["type"].unique()
    i = 0
    for gentype in all_gentypes:
        # offset1 = 0.5 * i / len(all_gentypes)
        # offset2 = 0.5 + 0.5 * i / len(all_gentypes)
        i = i + 1
        profiles[f"fuelcost_{gentype}"] = 1  # + 0.5 * np.sin(2 * 2 * np.pi * (timesteps / number_timesteps + offset1))
        profiles[f"inflow_{gentype}"] = 1  # + 0.5 * np.sin(2 * 2 * np.pi * (timesteps / number_timesteps + offset2))

    grid_data.node = nodes
    grid_data.branch = branches
    grid_data.generator = generators
    grid_data.consumer = consumers
    grid_data.profiles = profiles
    grid_data.validate_grid_data()

    parameter_data = {
        "nodetype": {
            "nodetype1": {
                "L": 1e-6,
                "Lp": 0,
                "S": 50,
                "Sp": 0,
                "max_cap": 1e5,
            }
        },
        "branchtype": {
            "branchtype1": {
                "B": 5.000,
                "Bdp": 0.47,
                "Bd": 0.680,
                "CL": 20.280,
                "CLp": 118.28,
                "CS": 129.930,
                "CSp": 757.84,
                "max_cap": 2000,
                "max_num": 5,
                "loss_fix": 0.032,
                "loss_slope": 3e-5,
            }
        },
        "gentype": {
            "gentype0": {
                "Cp": 0,
                "CO2": 0,
                "max_cap": 1e5,
            },
            "gentype1": {
                "Cp": 0.100,
                "CO2": 0,
                "max_cap": 1e5,
            },
        },
        "parameters": {
            "investment_years": investment_years,
            "finance_interest_rate": 0.05,
            "finance_years": 40,
            "operation_maintenance_rate": 0.05,
            "CO2_price": 0,
            "CO2_cap": None,
            "load_shed_penalty": 10000,
            "profiles_period_suffix": False,
        },
    }
    utils.validate_parameter_data(parameter_data)

    return grid_data, parameter_data


def create_case_star(investment_years, number_nodes, number_timesteps, base_MW=200):
    """Create test case of a given size"""

    grid_data = grid(investment_years, node=None, branch=None, generator=None, consumer=None)
    cols_node = grid_data.keys_sipdata["node"].keys()
    cols_branch = grid_data.keys_sipdata["branch"].keys()
    cols_generator = grid_data.keys_sipdata["generator"].keys()
    cols_consumer = grid_data.keys_sipdata["consumer"].keys()
    nodes = pd.DataFrame(columns=cols_node)
    radius = 5
    lat0 = 50
    lon0 = -30
    node_numbers = np.array(range(number_nodes))
    nodes_outer = node_numbers[1:]
    nodes["id"] = [f"n{i}" for i in node_numbers]
    nodes["area"] = "area"
    nodes["offshore"] = [i % 2 for i in node_numbers]
    nodes["type"] = "nodetype1"
    for year in investment_years:
        nodes[f"capacity_{year}"] = 0
        nodes[f"expand_{year}"] = 1
    nodes["cost_scaling"] = 1
    nodes.loc[1:, "lat"] = [lat0] * (number_nodes - 1) + (radius * (1 + 0.5 * nodes_outer / number_nodes)) * np.sin(
        np.pi * 2 / (number_nodes - 1) * nodes_outer
    )
    nodes.loc[1:, "lon"] = [lon0] * (number_nodes - 1) + (radius * (1 + 0.5 * nodes_outer / number_nodes)) * np.cos(
        np.pi * 2 / (number_nodes - 1) * nodes_outer
    )
    nodes.loc[0, "lat"] = lat0
    nodes.loc[0, "lon"] = lon0
    nodes.set_index("id", drop=False, inplace=True)
    nodes.index.name = "index"  # to avoid having "id" both as index name and column name

    branches = pd.DataFrame(columns=cols_branch)
    branches["node_from"] = [f"n{i}" for i in nodes_outer] * 2
    branches["node_to"] = [f"n{1+i%(number_nodes-1)}" for i in nodes_outer] + ["n0"] * (number_nodes - 1)
    for year in investment_years:
        branches[f"capacity_{year}"] = 0
        branches[f"expand_{year}"] = 1
    branches["max_newCap"] = -1
    branches["distance"] = -1
    branches["cost_scaling"] = 1
    branches["type"] = "branchtype1"

    # consumers on every node on bottom half
    range_bottom_half = range((number_nodes - 1) // 2 + 1, number_nodes)
    consumers = pd.DataFrame(columns=cols_consumer)
    consumers["node"] = [f"n{i}" for i in range_bottom_half]
    for year in investment_years:
        if year == investment_years[0]:
            consumers[f"demand_{year}"] = base_MW * len(investment_years)  # must be high enough
        else:
            consumers[f"demand_{year}"] = 0
    consumers["demand_ref"] = "demand1"

    # generators on every second node on top half, at least 2
    if number_nodes >= 7:
        # range_top_half = range(1, (number_nodes - 1) // 2, 2)
        range_top_half = [1 + i * 2 for i in range((number_nodes - 1) // 2 - 1)]
    else:
        range_top_half = [1, 2]
    generators = pd.DataFrame(columns=cols_generator)
    # generators at odd number nodes
    generators["node"] = [f"n{i}" for i in range_top_half]
    generators["type"] = [f"gentype{i%2}" for i in generators.index]
    generators["desc"] = "generator"
    # more than enough expensive generation:
    for year in investment_years:
        generators[f"expand_{year}"] = 0
        # res capacity increasing with time, expensive generation fixed from beginning
        clip = np.clip(1 + investment_years[0] - year, a_min=0, a_max=1)
        generators[f"capacity_{year}"] = 1.5 * base_MW * (1 - clip)
        generators.loc[generators["type"] == "gentype1", f"capacity_{year}"] = (
            3 * base_MW * len(investment_years) * clip
        )
    generators["pmin"] = 0
    generators["cost_scaling"] = 1
    for year in investment_years:
        generators[f"fuelcost_{year}"] = 0
        generators.loc[generators["type"] == "gentype1", f"fuelcost_{year}"] = 100
    generators["fuelcost_ref"] = "fuelcost_" + generators["type"]
    generators["pavg"] = 0
    generators["inflow_fac"] = 1
    generators["inflow_ref"] = "inflow_" + generators["type"]
    generators["allow_curtailment"] = 1

    timesteps = np.array(range(number_timesteps))
    profiles = pd.DataFrame(index=timesteps)
    profiles["demand1"] = 1.0
    all_gentypes = generators["type"].unique()
    i = 0
    for gentype in all_gentypes:
        # offset1 = 0.5 * i / len(all_gentypes)
        # offset2 = 0.5 + 0.5 * i / len(all_gentypes)
        i = i + 1
        profiles[f"fuelcost_{gentype}"] = 1  # + 0.5 * np.sin(2 * 2 * np.pi * (timesteps / number_timesteps + offset1))
        profiles[f"inflow_{gentype}"] = 1  # + 0.5 * np.sin(2 * 2 * np.pi * (timesteps / number_timesteps + offset2))

    grid_data.node = nodes
    grid_data.branch = branches
    grid_data.generator = generators
    grid_data.consumer = consumers
    grid_data.profiles = profiles
    grid_data.validate_grid_data()

    parameter_data = {
        "nodetype": {
            "nodetype1": {
                "L": 1e-6,
                "Lp": 0,
                "S": 50,
                "Sp": 0,
                "max_cap": 1e5,
            }
        },
        "branchtype": {
            "branchtype1": {
                "B": 5.000,
                "Bdp": 0.47,
                "Bd": 0.680,
                "CL": 20.280,
                "CLp": 118.28,
                "CS": 129.930,
                "CSp": 757.84,
                "max_cap": 2000,
                "max_num": 5,
                "loss_fix": 0,  # 0.032,
                "loss_slope": 0,  # 3e-5,
            }
        },
        "gentype": {
            "gentype0": {
                "Cp": 0,
                "CO2": 0,
                "max_cap": 1e5,
            },
            "gentype1": {
                "Cp": 0.10,
                "CO2": 0,
                "max_cap": 1e5,
            },
        },
        "parameters": {
            "investment_years": investment_years,
            "finance_interest_rate": 0.05,
            "finance_years": 40,
            "operation_maintenance_rate": 0.05,
            "CO2_price": 0,
            "CO2_cap": None,
            "load_shed_penalty": 10000,
            "profiles_period_suffix": False,
        },
    }
    utils.validate_parameter_data(parameter_data)

    return grid_data, parameter_data


def create_case_star_loadflex(investment_years, number_nodes, number_timesteps, base_MW=200):
    grid_data, parameter_data = create_case_star(investment_years, number_nodes, number_timesteps, base_MW)

    # add load flexibility
    flex_data = {
        "load_flex_shift_frac": {y: 0.08 for y in investment_years},
        "load_flex_shift_max": {y: 2 for y in investment_years},
        "load_flex_price_frac": {y: 0.05 for y in investment_years},
        "load_flex_price_cap ": {y: 40 for y in investment_years},
    }
    parameter_data = {**parameter_data, **flex_data}
    return grid_data, parameter_data
