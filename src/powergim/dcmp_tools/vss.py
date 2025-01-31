"""
CALCULATION OF VSS (VALUE OF STOCHASTIC SOLUTION)

1) TAKE THE RESULT OF THE STOCHASTIC SOLUTION (RP)
2) SOLVE THE 'EXPECTED VALUE PROBLEM' (EVP)
3) FIX FIRST STAGE DECISION BASED ON THE EVP
4) CALCULATE EEV ('EXPECTATION OF THE EXPECTED VALUE SOLUTION')
5) CALCULATE VSS

@author: spyridonc
"""
import json
from pathlib import Path

import cloudpickle
import pyomo.environ as pyo

from .decomposition import MultiHorizonPgim
from .utils_spi import ReferenceModelCreator, extract_all_opt_variable_values


def fix_first_stage_vars(model, sol):

    all_variables = list(model.non_decomposed_model.component_data_objects(pyo.Var))
    investment_vars = [
        var
        for var in all_variables
        if (
            var.name.startswith("z_new_lines")
            or var.name.startswith("z_new_capacity")
            or var.name.startswith("z_capacity_total")
        )
    ]
    first_stage_vars = [var for var in investment_vars if var.index()[-1] == 0]

    for i, var in enumerate(first_stage_vars):
        if first_stage_vars[i].name.startswith("z_new_lines"):
            if var.index()[0] in sol["mh_rslt"]["x_new_lines"].keys():
                first_stage_vars[i].fix(round(sol["mh_rslt"]["x_new_lines"][var.index()[0]]))
            else:
                first_stage_vars[i].fix(0)
        elif first_stage_vars[i].name.startswith("z_new_capacity"):
            if var.index()[0] in sol["mh_rslt"]["x_new_capacity"].keys():
                first_stage_vars[i].fix(sol["mh_rslt"]["x_new_capacity"][var.index()[0]])
            else:
                first_stage_vars[i].fix(0)
        elif first_stage_vars[i].name.startswith("z_new_capacity_total"):
            if var.index()[0] in sol["mh_rslt"]["x_new_capacity_total"].keys():
                first_stage_vars[i].fix(sol["mh_rslt"]["x_new_capacity_total"][var.index()[0]])
            else:
                first_stage_vars[i].fix(0)
    return


def calc_vss(case_settings, RP, path_in, path_out, lp_solver="gurobi"):

    GRID_CASE = case_settings["configuration"]["grid_case"]
    BRANCHES_FILE_NAME = case_settings["configuration"]["branches_file_used"]
    N_SAMPLES = case_settings["configuration"]["number_of_samples"]
    PROB = case_settings["configuration"]["scneario_probabilities"]
    MH_USE_BIN_EXPANS = case_settings["configuration"]["using_binary_expansion"]
    MH_USE_FIXED_CAP = case_settings["configuration"]["using_fixed_capacity_lines"]

    # -----------------------------CREATING EVP MODEL-----------------------------
    pgim_evp = ReferenceModelCreator(
        grid_case=GRID_CASE, branches_file_name=BRANCHES_FILE_NAME, s_sample_size=N_SAMPLES, probabilities=PROB
    )

    pgim_evp.create_reference_pgim_model(path_in, path_out, scenario_selection="EVP")

    # -----------------------------CREATING MULTIHORIZON TREE STRUCTURES-----------------------------
    mh_evp = MultiHorizonPgim(pgim_evp, is_stochastic=False)

    mh_evp.create_ancestors_struct()

    # -----------------------------CREATING MULTI-HORIZON PROBLEM FROM PGIM DATA-----------------------------
    mh_evp.get_pgim_params_per_scenario()

    mh_evp.create_multi_horizon_problem(USE_BIN_EXPANS=MH_USE_BIN_EXPANS, USE_FIXED_CAP_LINES=MH_USE_FIXED_CAP)

    solver = pyo.SolverFactory(lp_solver)
    # Set Gurobi solver parameters
    solver.options["TimeLimit"] = 60 * 60 * 2.5  # seconds
    solver.options["MIPGap"] = 0.00001

    results = solver.solve(mh_evp.non_decomposed_model, tee=False, keepfiles=False, symbolic_solver_labels=True)

    if str(results.solver.termination_condition) != "optimal":
        print(results.solver.termination_condition)

    optimal_solution_evp = extract_all_opt_variable_values(mh_evp.non_decomposed_model)

    first_stage_sol = mh_evp.get_first_stage_decision(optimal_solution_evp)

    optimal_solution_evp_dict = {
        "mh_rslt": {
            "x_new_lines": first_stage_sol["new_lines"].to_dict(),
            "x_new_capacity": first_stage_sol["new_capacity"].to_dict(),
        },
    }

    # -----------------------------CREATING EEVP MODEL-----------------------------
    pgim_eevp = ReferenceModelCreator(
        grid_case=GRID_CASE, branches_file_name=BRANCHES_FILE_NAME, s_sample_size=N_SAMPLES, probabilities=PROB
    )

    pgim_eevp.create_reference_pgim_model(path_in, path_out, scenario_selection="default")

    # -----------------------------CREATING MULTIHORIZON TREE STRUCTURES-----------------------------
    mh_eevp = MultiHorizonPgim(pgim_eevp, is_stochastic=True)

    mh_eevp.create_ancestors_struct()

    # -----------------------------CREATING MULTI-HORIZON PROBLEM FROM PGIM DATA-----------------------------
    mh_eevp.get_pgim_params_per_scenario()

    mh_eevp.create_multi_horizon_problem(USE_BIN_EXPANS=MH_USE_BIN_EXPANS, USE_FIXED_CAP_LINES=MH_USE_FIXED_CAP)

    fix_first_stage_vars(mh_eevp, optimal_solution_evp_dict)

    results = solver.solve(mh_eevp.non_decomposed_model, tee=False, keepfiles=False, symbolic_solver_labels=True)

    if str(results.solver.termination_condition) != "optimal":
        print(results.solver.termination_condition)

    eev = round(pyo.value(mh_eevp.non_decomposed_model.objective))
    print(f"MULTI-HORIZON OBJECTIVE VALUE FOR GIVEN FIRST-STAGE INVESTMENTS: {eev}")

    vss = eev - RP

    return vss, eev


if __name__ == "__main__":

    PATH_INPUT = Path(__file__).parents[3] / "examples" / "inputs" / "CASE_BASELINE"
    PATH_OUTPUT = Path(__file__).parents[3] / "examples" / "outputs"

    timestamp = "20241003_145917"  # CHECKME: choose an existing run case

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

    vss, eev = calc_vss(run_settings, RP_OBJ_VAL, PATH_INPUT, PATH_OUTPUT, lp_solver="gurobi")

    print(f"\n RP = {round(RP_OBJ_VAL)}")
    print(f"\n EEV = {round(eev)}")
    print(f"\n VSS = {round(vss)}")
