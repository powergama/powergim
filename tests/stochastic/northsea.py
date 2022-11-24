import logging
import sys
from pathlib import Path

import mpi4py
import mpisppy.opt.ph
import mpisppy.utils.sputils
import pandas as pd
import pyomo.environ as pyo

import powergim as pgim

TEST_DATA_ROOT_PATH = Path(__file__).absolute().parent / "data"
NUMERIC_THRESHOLD = 1e-3

NUM_SCENARIOS = 4

TMP_PATH = Path()


def my_scenario_creator(scenario_name, grid_data, parameter_data):
    """Create a scenario."""
    print("Scenario {}".format(scenario_name))

    # Adjust data according to scenario name
    num_scenarios = NUM_SCENARIOS
    probabilities = {f"scen{k}": 1 / num_scenarios for k in range(num_scenarios)}
    # probabilities = {"scen0": 0.3334, "scen1": 0.3333, "scen2": 0.3333,"scen"}
    if scenario_name == "scen0":
        pass
    elif scenario_name == "scen1":
        # Less wind at SN2
        grid_data.generator.loc[4, "capacity_2028"] = 1400
    elif scenario_name == "scen2":
        # More wind and SN2
        grid_data.generator.loc[4, "capacity_2028"] = 10000
        grid_data.generator.loc[3, "capacity_2028"] = 10000
    elif scenario_name == "scen3":
        # More wind, more demand
        grid_data.generator.loc[4, "capacity_2028"] = 8000
    elif scenario_name == "scen4":
        grid_data.generator.loc[4, "capacity_2028"] = 9000
    elif scenario_name == "scen5":
        grid_data.generator.loc[4, "capacity_2028"] = 10000
    else:
        raise ValueError("Invalid scenario name")

    # Create stochastic model:
    sip = pgim.SipModel(grid_data, parameter_data)

    model = sip.scenario_creator(scenario_name, probability=probabilities[scenario_name])
    return model


def my_scenario_denouement(rank, scenario_name, scenario):
    print(f"DENOUEMENT scenario={scenario_name} OBJ={pyo.value(scenario.OBJ)}")
    all_var_values_dict = pgim.SipModel.extract_all_variable_values(scenario)
    dfs = []
    for varname, data in all_var_values_dict.items():
        if data is None:
            logging.warning(f"northsea.py: Skipping variable with no data ({varname})")
            continue
        df = pd.DataFrame(data).reset_index()
        df.loc[:, "variable"] = varname
        dfs.append(df)
    pd.concat(dfs).to_csv(TMP_PATH / f"ph_res_ALL_{scenario_name}.csv")


def solve_ph(solver_name):

    # Read input data
    parameter_data = pgim.file_io.read_parameters(TEST_DATA_ROOT_PATH / "parameters_stoch.yaml")
    grid_data = pgim.file_io.read_grid(
        investment_years=parameter_data["parameters"]["investment_years"],
        nodes=TEST_DATA_ROOT_PATH / "nodes.csv",
        branches=TEST_DATA_ROOT_PATH / "branches.csv",
        generators=TEST_DATA_ROOT_PATH / "generators.csv",
        consumers=TEST_DATA_ROOT_PATH / "consumers.csv",
    )
    file_timeseries_sample = TEST_DATA_ROOT_PATH / "time_series_sample.csv"
    grid_data.profiles = pgim.file_io.read_profiles(filename=file_timeseries_sample)

    # Scenarios
    scenario_creator_kwargs = {"grid_data": grid_data, "parameter_data": parameter_data}
    # scenario_names = ["scen0", "scen1", "scen2", "scen3", "scen4", "scen5"]
    scenario_names = [f"scen{k}" for k in range(NUM_SCENARIOS)]

    # Solve via progressive hedging (PH)
    options = {
        "solvername": solver_name,
        "PHIterLimit": 5,
        "defaultPHrho": 10,
        "convthresh": 1e-7,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "linearize_proximal_terms": True,
        "proximal_linearization_tolerance ": 0.1,  # default =1e-1
        "initial_proximal_cut_count": 2,  # default = 2
        "iter0_solver_options": {},  # {"mipgap": 0.01},  # dict(),
        "iterk_solver_options": {},  # {"mipgap": 0.005},  # dict(),
    }
    ph = mpisppy.opt.ph.PH(
        options,
        scenario_names,
        scenario_creator=my_scenario_creator,
        scenario_denouement=my_scenario_denouement,  # post-processing and reporting
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # solve
    conv, obj, tbound = ph.ph_main()

    rank = mpi4py.MPI.COMM_WORLD.Get_rank()
    # These values are not very useful:
    # print(f"{rank}: conv = {conv}")
    # print(f"{rank}: obj = {obj}")
    # print(f"{rank}: tbound= {tbound}")

    # Extract results:
    res_ph = []
    variables = ph.gather_var_values_to_rank0()
    df_res = None
    if variables is not None:
        # this is true when rank is zero.
        for (scenario_name, variable_name) in variables:
            variable_value = variables[scenario_name, variable_name]
            res_ph.append({"scen": scenario_name, "var": variable_name, "value": variable_value})
        df_res = pd.DataFrame(data=res_ph)
        print(f"{rank}: Saving to file...ph_res_rank0.csv")
        df_res.to_csv(TMP_PATH / "ph_res_rank0.csv")
    return ph, df_res


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        TMP_PATH = Path(filepath)

    main_ph = solve_ph(solver_name="glpk")
