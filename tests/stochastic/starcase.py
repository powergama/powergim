import logging
import sys
from pathlib import Path

import mpi4py
import mpisppy.opt.lshaped
import mpisppy.opt.ph
import mpisppy.utils.sputils
import pandas as pd
import pyomo.environ as pyo

import powergim as pgim
import powergim.testcases

NUM_SCENARIOS = 4
TMP_PATH = Path()


def my_scenario_creator(scenario_name, grid_data, parameter_data):
    """Create a scenario."""
    print("Scenario {}".format(scenario_name))

    # Adjust data according to scenario name
    num_scenarios = NUM_SCENARIOS
    probabilities = {f"scen{k}": 1 / num_scenarios for k in range(num_scenarios)}
    years = parameter_data["parameters"]["investment_years"]
    cap = f"capacity_{years[1]}"
    if scenario_name == "scen0":
        # Base case, no modification
        pass
    elif scenario_name == "scen1":
        grid_data.generator.loc[0, cap] = 0
    elif scenario_name == "scen2":
        grid_data.generator.loc[0, cap] = 5000
    elif scenario_name == "scen3":
        # More wind, more demand
        grid_data.generator.loc[0, cap] = 5000
        grid_data.generator.loc[0, cap] = 8000
        grid_data.consumer[f"demand_{years[0]}"] = 1.20 * grid_data.consumer[f"demand_{years[0]}"]
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
    years = [0, 10]
    number_nodes = 4
    number_timesteps = 4
    grid_data, parameter_data = powergim.testcases.create_case_star(years, number_nodes, number_timesteps, base_MW=2000)
    scenario_creator_kwargs = {"grid_data": grid_data, "parameter_data": parameter_data}
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
        "iter0_solver_options": {"mipgap": 0.01},  # dict(),
        "iterk_solver_options": {"mipgap": 0.001},  # {"mipgap": 0.005},  # dict(),
    }
    ph = mpisppy.opt.ph.PH(
        options,
        scenario_names,
        scenario_creator=my_scenario_creator,
        scenario_denouement=my_scenario_denouement,  # post-processing and reporting
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # Replace standard solver creation with modified version (to work with CBC)
    # TODO: not working
    # mpisppy.opt.ph.PH._create_solvers = my_create_solvers

    # solve
    conv, obj, tbound = ph.ph_main()
    # conv, obj, tbound = ph_main_modified(ph)

    rank = mpi4py.MPI.COMM_WORLD.Get_rank()

    # Extract results:
    res_ph = []
    variables = ph.gather_var_values_to_rank0()
    df_res = None
    if variables is not None:
        # this is true when rank is zero.
        for scenario_name, variable_name in variables:
            variable_value = variables[scenario_name, variable_name]
            res_ph.append({"scen": scenario_name, "var": variable_name, "value": variable_value})
        df_res = pd.DataFrame(data=res_ph)
        print(f"{rank}: Saving to file...ph_res_rank0.csv")
        df_res.to_csv(TMP_PATH / "ph_res_rank0.csv")
    return ph, df_res


def solve_benders(solver_name):
    years = [0, 10]
    number_nodes = 4
    number_timesteps = 4
    grid_data, parameter_data = powergim.testcases.create_case_star(years, number_nodes, number_timesteps, base_MW=2000)
    scenario_creator_kwargs = {"grid_data": grid_data, "parameter_data": parameter_data}
    scenario_names = [f"scen{k}" for k in range(NUM_SCENARIOS)]
    bounds = {name: -432000 for name in scenario_names}
    options = {
        "root_solver": solver_name,
        "sp_solver": solver_name,
        "sp_solver_options": {"threads": 1},
        "valid_eta_lb": bounds,
        "max_iter": 10,
    }
    ls = mpisppy.opt.lshaped.LShapedMethod(
        options, scenario_names, scenario_creator=my_scenario_creator, scenario_creator_kwargs=scenario_creator_kwargs
    )
    result = ls.lshaped_algorithm()
    variables = ls.gather_var_values_to_rank0()
    for (scen_name, var_name), var_value in variables.items():
        print(scen_name, var_name, var_value)
    return ls, result


def solve_ef(solver_name, solver_io=None):
    years = [0, 10]
    number_nodes = 4
    number_timesteps = 4
    grid_data, parameter_data = powergim.testcases.create_case_star(years, number_nodes, number_timesteps, base_MW=2000)
    scenario_creator_kwargs = {"grid_data": grid_data, "parameter_data": parameter_data}
    scenario_names = [f"scen{k}" for k in range(NUM_SCENARIOS)]

    # Solve estensive form
    main_ef = mpisppy.utils.sputils.create_EF(
        scenario_names,
        scenario_creator=my_scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    solver = pyo.SolverFactory(solver_name, solver_io=solver_io)
    solver.solve(
        main_ef,
        tee=True,
        symbolic_solver_labels=True,
    )

    # Extract results:
    all_var_values = {}
    for scen in mpisppy.utils.sputils.ef_scenarios(main_ef):
        scen_name = scen[0]
        this_scen = scen[1]
        all_var_values[scen_name] = pgim.SipModel.extract_all_variable_values(this_scen)
        all_var_values[scen_name]["OBJ"] = pyo.value(this_scen.OBJ)
        print(f"{scen_name}: OBJ = {pyo.value(this_scen.OBJ)}")
        print(f"{scen_name}: opCost = {all_var_values[scen_name][f'{scen_name}.v_operating_cost'].values}")
    print("EF_Obj", pyo.value(main_ef.EF_Obj))
    return main_ef, all_var_values


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TMP_PATH = Path(sys.argv[1])
        solver_name = sys.argv[2]
    main_ph = solve_ph(solver_name=solver_name)

    # main_ph = solve_ph(solver_name="cbc")
    # main_ef = solve_ef("cbc")
    # ls = solve_benders(solver_name="cbc")

# Trouble using CBC:
# Not able to pass on solver_io="nl" argument to PH algorithm
# CBC solver is created without any options passed to SolverFactory
# See:
# mpisppy/opt/ph.py#25: def ph_main(self, finalize=True):
# mpisppy/opt/ph.py#58: trivial_bound = self.Iter0()
# mpisppy/phbase.py#677: def Iter0(self)
# mpisppy/phbase.py#710: self._create_solvers()
# mpisppy/spopt.py#L838: def _create_solvers(self))
