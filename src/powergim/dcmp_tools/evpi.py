"""
CALCULATION OF EVPI (EXPECTED VALUE OF PERFECT INFORMATION)

1) TAKE THE RESULT OF THE STOCHASTIC SOLUTION (RP)
2) SOLVE THE DETERMINISTIC CASES FOR THE DIFFERENT SCENARIOS ('default', 'A1', 'A2')
3) CALCULATE WS ('WAIT AND SEE' DECISION)
4) CALCULATE EVPI

@author: spyridonc
"""
import json
from pathlib import Path

import cloudpickle
import pyomo.environ as pyo

from .decomposition import MultiHorizonPgim
from .utils_spi import ReferenceModelCreator


def calc_evpi(case_settings, RP, path_in, path_out, lp_solver="gurobi"):

    GRID_CASE = case_settings["configuration"]["grid_case"]
    BRANCHES_FILE_NAME = case_settings["configuration"]["branches_file_used"]
    N_SAMPLES = case_settings["configuration"]["number_of_samples"]
    PROB = case_settings["configuration"]["scneario_probabilities"]
    MH_USE_BIN_EXPANS = case_settings["configuration"]["using_binary_expansion"]
    MH_USE_FIXED_CAP = case_settings["configuration"]["using_fixed_capacity_lines"]

    mh_obj_vals = {}
    # Set Gurobi solver parameters
    solver = pyo.SolverFactory(lp_solver)
    solver.options["TimeLimit"] = 60 * 60 * 2.5  # seconds
    solver.options["MIPGap"] = 0.00001

    for scen in PROB.keys():

        match scen:
            case "scen0":
                current_scenario = "default"
            case "scen1":
                current_scenario = "A1"
            case "scen2":
                current_scenario = "A2"
            case _:
                raise ValueError("Non-identified scenario.")

        pgim_model = ReferenceModelCreator(
            grid_case=GRID_CASE, branches_file_name=BRANCHES_FILE_NAME, s_sample_size=N_SAMPLES, probabilities=PROB
        )

        pgim_model.create_reference_pgim_model(path_in, path_out, scenario_selection=current_scenario)

        mh_model = MultiHorizonPgim(pgim_model, is_stochastic=False)
        mh_model.create_ancestors_struct()
        mh_model.get_pgim_params_per_scenario()
        mh_model.create_multi_horizon_problem(USE_BIN_EXPANS=MH_USE_BIN_EXPANS, USE_FIXED_CAP_LINES=MH_USE_FIXED_CAP)

        rstl = solver.solve(mh_model.non_decomposed_model, tee=False, keepfiles=False, symbolic_solver_labels=True)

        if str(rstl.solver.termination_condition) != "optimal":
            print(rstl.solver.termination_condition)

        mh_obj_vals[scen] = round(pyo.value(mh_model.non_decomposed_model.objective))

        print(f"\n MULTI-HORIZON OBJECTIVE VALUE FOR SCENARIO {scen} is: {mh_obj_vals[scen]}")

    ws_val = sum(prob * obj_val for prob, obj_val in zip(list(PROB.values()), list(mh_obj_vals.values())))
    evpi = RP - ws_val

    return evpi, ws_val


if __name__ == "__main__":

    PATH_INPUT = Path(__file__).parents[3] / "examples" / "inputs" / "CASE_BASELINE"
    PATH_OUTPUT = Path(__file__).parents[3] / "examples" / "outputs"

    timestamp = "20241025_142541"  # CHECKME: choose an existing run case

    rslt_file_name = "run_Benders_validation_" + timestamp + ".pickle"
    report_file_name = "run_Benders_validation_" + timestamp + ".json"
    rslt_file_path = PATH_OUTPUT / rslt_file_name
    report_file_path = PATH_OUTPUT / report_file_name

    # Open the PICKLE file
    try:
        with open(rslt_file_path, "rb") as file:
            run_rslt = cloudpickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("The PICKLE file you are trying to read does not exist!")

    # Open the JSON file
    try:
        with open(report_file_path, "r") as file:
            run_settings = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("The JSON file you are trying to read does not exist!")

    RP_OBJ_VAL = run_rslt["mh_rslt"]["obj_val"]

    evpi, ws = calc_evpi(run_settings, RP_OBJ_VAL, PATH_INPUT, PATH_OUTPUT)

    print(f"\n RP = {round(RP_OBJ_VAL)}")
    print(f"\n WS = {round(ws)}")
    print(f"\n EVPI = {round(evpi)}")
