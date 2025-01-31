import copy
import datetime as dtime
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cloudpickle
import mpisppy.opt.ph
import mpisppy.utils.sputils
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import powergim as pgim

from .decomposition import BendersDecomp, MultiHorizonPgim
from .evpi import calc_evpi
from .results_plots import create_geopandas_figures_results, create_interactive_maps_results
from .utils_spi import CaseMapVisualizer, ReferenceModelCreator, TimeseriesBuilder, extract_all_opt_variable_values
from .vss import calc_vss


def pgim_ref_init(input_options: dict):
    """Initialize a `pgim` reference case.

    Args:
        input_options (dict): User-defined case options.

    Returns:
        pgim_ref (ReferenceModelCreator): Reference `pgim` case.
    """

    # These probabilities were used in the paper.
    if input_options["GRID_CASE"] == "baseline":
        PROB = {"scen0": 0.1, "scen1": 0.45, "scen2": 0.45}
    elif input_options["GRID_CASE"] == "star":
        PROB = {"scen0": 0.8, "scen1": 0.1, "scen2": 0.1}

    input_options["PROB"] = PROB

    # -------------# Sanity checks for the user-defined inputs.
    if not input_options["IS_STOCHASTIC"]:
        input_options["DO_CALC_VSS_EVPI"] = False
    if input_options["BENDERS_SINGLE_SUB"]:
        input_options["BENDERS_SINGLE_CUT"] = True
    if (input_options["GRID_CASE"] == "star") or (input_options["MH_USE_BIN_EXPANS"]):
        input_options["DO_VIZ"] = False

    # ---------------------GETTING/CREATING REQUIRED TIMESERIES CSV FILE-----------------------------
    if input_options["GRID_CASE"] == "baseline":
        TimeseriesBuilder(
            input_options["PATH_INPUT"]
        ).create_csv()  # checking/creating existence of 1 year-long timeseries CSV for pgim.

    # -----------------------------CREATING BASELINE MODEL-----------------------------
    pgim_ref = ReferenceModelCreator(
        grid_case=input_options["GRID_CASE"],
        branches_file_name=input_options["BRANCHES_FILE_NAME"],
        s_sample_size=input_options["N_SAMPLES"],
        probabilities=PROB,
    )
    pgim_ref.create_reference_pgim_model(
        input_options["PATH_INPUT"], input_options["PATH_OUTPUT"], scenario_selection="default"
    )

    return pgim_ref


def do_mh(
    input_options: dict,
    pgim_ref: ReferenceModelCreator,
    # mip_solver: str = "glpk",  # TODO: make automatic checks for admissible values: 'gurobi', 'glpk'
):
    """Formulates the multi-horizon version of the refernce `pgim` model and solves it.

    Args:
        input_options (dict): User-defined case options.
        pgim_ref (ReferenceModelCreator): Reference `pgim` case.

    Returns:
        optimal_solution_multi_hor (dict): Optimal solution of the multi-horzion formulation.
        mh_ref (MultiHorizonPgim): Multi-horizon formulation of the `pgim` reference case.
    """

    mip_solver = input_options["LP_SOLVER"]

    # -----------------------------CREATING MULTIHORIZON TREE STRUCTURES-----------------------------
    mh_ref = MultiHorizonPgim(pgim_ref, is_stochastic=input_options["IS_STOCHASTIC"])
    mh_ref.create_ancestors_struct()

    # -----------------------------VISUALIZING CASE: INPUT DATA-----------------------------
    if input_options["DO_VIZ"] and input_options["GRID_CASE"] == "baseline":
        case_visualizer = CaseMapVisualizer(
            pgim_ref, outFileName="case_map_interactive.html", outPath=input_options["PATH_OUTPUT"]
        )
        case_visualizer.create_html_map()

    # -----------------------------CREATING MULTI-HORIZON PROBLEM FROM PGIM DATA-----------------------------
    mh_ref.get_pgim_params_per_scenario()
    mh_ref.create_multi_horizon_problem(
        USE_BIN_EXPANS=input_options["MH_USE_BIN_EXPANS"], USE_FIXED_CAP_LINES=input_options["MH_USE_FIXED_CAP"]
    )

    # -----------------------------SOLVING MULTI-HORIZON PROBLEM-----------------------------
    solver = pyo.SolverFactory(mip_solver)
    # Set Gurobi solver parameters
    # solver.options['TimeLimit'] = 60*60*3
    # solver.options['MIPGap'] = 0.00001

    results = solver.solve(mh_ref.non_decomposed_model, tee=True, keepfiles=False, symbolic_solver_labels=True)

    if str(results.solver.termination_condition) != "optimal":
        print(results.solver.termination_condition)

    optimal_solution_multi_hor = extract_all_opt_variable_values(mh_ref.non_decomposed_model)

    return optimal_solution_multi_hor, mh_ref


def solve_pgim_ref(
    input_options: dict,
    pgim_ref: ReferenceModelCreator,
    mh_ref: MultiHorizonPgim,
    # mip_solver: str = "glpk",  # TODO: make automatic checks for admissible values: 'gurobi', 'glpk'
):
    """Solves the `pgim` referennce model.

    Args:
        input_options (dict): User-defined case options.
        pgim_ref (ReferenceModelCreator): Reference `pgim` case.
        mh_ref (MultiHorizonPgim): Multi-horizon formulation of the `pgim` reference case.

    Returns:
        optimal_solution_pgim (dict): Optimal solution of the `pgim` reference model.
        my_pgim_stochastic_model_ef : Reference `pgim` model for the stochastic case (extented formulation).
    """

    mip_solver = input_options["LP_SOLVER"]

    # -----------------------------SOLVING PGIM MODEL-----------------------------
    if not input_options["IS_STOCHASTIC"]:

        solver = pyo.SolverFactory(mip_solver)
        # Set Gurobi solver parameters
        if mip_solver == "gurobi":
            solver.options["TimeLimit"] = 60 * 60 * 3
            solver.options["MIPGap"] = 0.00001

        results = solver.solve(pgim_ref.ref_pgim_model, tee=True, keepfiles=False, symbolic_solver_labels=True)

        if str(results.solver.termination_condition) != "optimal":
            print(results.solver.termination_condition)

        optimal_solution_pgim = pgim_ref.ref_pgim_model.extract_all_variable_values()

        my_pgim_stochastic_model_ef = None

    elif input_options["IS_STOCHASTIC"]:
        # Define 3 scenarios (base case and 2 more) --> 7 strategic nodes

        print("\n --------Creating scenarios for DEF")
        # -----------------------Create scenario powerGIM models-----------------------
        # This function edits the inital grid_data and parameters depending on the scenario.
        # Then it creates a model instance for the scenario with the modified parameters.

        if input_options["GRID_CASE"] == "star":

            def my_mpisppy_scenario_creator(scenario_name):
                """Create a scenario."""
                print("\n Scenario {}".format(scenario_name))

                parameter_data = copy.deepcopy(mh_ref.pgim_case.ref_params)
                grid_data = copy.deepcopy(mh_ref.pgim_case.ref_grid_data)

                match scenario_name:
                    case "scen0":
                        pass

                    # --------DEFAULT SCENARIOS

                    case "scen1":  # The "Low wind", "same demand" scenario

                        # Half the wind at n1 (wind farm node).
                        init_wind_capacity = grid_data.generator.loc[grid_data.generator["node"] == "n1"]

                        for iperiod in parameter_data["parameters"]["investment_years"]:
                            grid_data.generator.loc[
                                grid_data.generator["node"] == "n1", ["capacity_" + str(iperiod)]
                            ] = (0.5 * init_wind_capacity.loc[0, "capacity_" + str(iperiod)])

                    case "scen2":  # The "same wind", "high demand" scenario

                        # Double the load at n3 (offshore load node).
                        init_load_capacity = grid_data.consumer.loc[grid_data.consumer["node"] == "n3"]

                        for iperiod in parameter_data["parameters"]["investment_years"]:
                            grid_data.consumer.loc[grid_data.consumer["node"] == "n3", ["demand_" + str(iperiod)]] = (
                                2 * init_load_capacity.loc[1, "demand_" + str(iperiod)]
                            )

                    # --------ALTERNATIVE SCENARIOS

                    # case "scen1": # The "Lower demand at country node" scenario

                    #     # Half the load at n2 (country node).
                    #     init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n2']

                    #     for iperiod in parameter_data['parameters']['investment_years']:
                    #         grid_data.consumer.loc[grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 0.5*init_load_capacity.loc[0,'demand_' + str(iperiod)]

                    # case "scen2": # The "Higher demand at country node" scenario

                    #     # Double the load at n2 (country node).
                    #     init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n2']

                    #     for iperiod in parameter_data['parameters']['investment_years']:
                    #         grid_data.consumer.loc[grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 2*init_load_capacity.loc[0,'demand_' + str(iperiod)]

                    case _:
                        raise ValueError("Invalid scenario name")

                # Create stochastic model:
                pgim_model = pgim.SipModel(
                    grid_data, parameter_data
                )  # A) Initialize a pgim object instane (pgim_model)
                pgim_scen_model = pgim_model.scenario_creator(
                    scenario_name, probability=input_options["PROB"][scenario_name]
                )  # B) Use scenario_creator method to build a scenario instance model

                return pgim_scen_model

        else:

            def my_mpisppy_scenario_creator(scenario_name):
                """Create a scenario."""
                print("\n Scenario {}".format(scenario_name))

                parameter_data = copy.deepcopy(pgim_ref.ref_params)
                grid_data = copy.deepcopy(pgim_ref.ref_grid_data)

                match scenario_name:
                    case "scen0":
                        pass

                    case "scen1":  # oceangrid_A1_lessdemand
                        #  Less NO demand (220 TWh in 2050)

                        demand_scale = 220 / 260
                        # demand_scale = 130/260
                        m_demand = grid_data.consumer["node"].str.startswith("NO_")
                        for year in parameter_data["parameters"]["investment_years"]:

                            # -----Hardcoded scenario #1
                            grid_data.consumer.loc[m_demand, f"demand_{year}"] = (
                                demand_scale * grid_data.consumer.loc[m_demand, f"demand_{year}"]
                            )

                            # -----Hardcoded scenario #2
                            # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 0.5 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]

                            # -----Hardcoded scenario #3
                            # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 2 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                    case "scen2":  # oceangrid_A2_moredemand
                        # More NO demand (340 TWh in 2050)

                        demand_scale = 340 / 260
                        # demand_scale = 520/260
                        m_demand = grid_data.consumer["node"].str.startswith("NO_")
                        for year in parameter_data["parameters"]["investment_years"]:

                            # -----Hardcoded scenario #1
                            grid_data.consumer.loc[m_demand, f"demand_{year}"] = (
                                demand_scale * grid_data.consumer.loc[m_demand, f"demand_{year}"]
                            )

                            # -----Hardcoded scenario #2
                            # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 2 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]

                            # -----Hardcoded scenario #3
                            # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 0.5 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                    case _:
                        raise ValueError("Invalid scenario name")

                # Create stochastic model:
                # A) Initialize a pgim object instane (pgim_model)
                pgim_model = pgim.SipModel(grid_data, parameter_data)
                # B) Use scenario_creator method to build a scenario instance model
                pgim_scen_model = pgim_model.scenario_creator(
                    scenario_name, probability=input_options["PROB"][scenario_name]
                )

                return pgim_scen_model

        # This formulates the extensive form based on the scenarios that are defined (an mpisppy object)
        # Preferred method based on mpispppy: mpisppy.opt.ef.ExtensiveForm --> needs mpi4py
        my_pgim_stochastic_model_ef = mpisppy.utils.sputils.create_EF(
            mh_ref.scenario_names, scenario_creator=my_mpisppy_scenario_creator
        )

        # Solve the EF
        solver = pyo.SolverFactory(mip_solver)
        if mip_solver == "gurobi":
            solver.options["TimeLimit"] = 60 * 60 * 3  # seconds
            solver.options["MIPGap"] = 0.00001

        solver.solve(my_pgim_stochastic_model_ef, tee=True, symbolic_solver_labels=True)

        all_var_all_scen_values = []

        # Extract results:
        for scen in mpisppy.utils.sputils.ef_scenarios(my_pgim_stochastic_model_ef):
            # Iterable has 2 dimensions: (scenario_name, scnenario_model (associated pyomo model variables))
            # scen_name = scen[0]
            this_scen = scen[1]
            all_var_values = pgim.SipModel.extract_all_variable_values(this_scen)
            all_var_all_scen_values.append(all_var_values)

        optimal_solution_pgim = all_var_all_scen_values

    return optimal_solution_pgim, my_pgim_stochastic_model_ef


# -----------------------------RUNNING BENDERS DECOMPOSITION ALGORITHM-----------------------------
def do_bd(input_options: dict, mh_ref: MultiHorizonPgim, timestamp: str):
    """Implements the Bender's algorithm solution approach.

    Args:
        input_options (dict): User-defined case options.
        mh_ref (MultiHorizonPgim): Multi-horizon formulation of the `pgim` reference case.
        timestamp (str): Timestamp of case run.

    Returns:
        _type_: _description_ # TODO
    """

    match input_options["GRID_CASE"]:
        case "star":
            match input_options["IS_STOCHASTIC"]:
                case False:
                    my_BD_setup = BendersDecomp(mh_ref, CONVRG_EPS=0.01, MAX_BD_ITER=1000, INIT_UB=30000)
                case True:
                    my_BD_setup = BendersDecomp(mh_ref, CONVRG_EPS=0.01, MAX_BD_ITER=1000, INIT_UB=30000)
        case "baseline":
            match input_options["IS_STOCHASTIC"]:
                case False:
                    my_BD_setup = BendersDecomp(mh_ref, CONVRG_EPS=0.1, MAX_BD_ITER=1000, INIT_UB=800000)
                case True:
                    my_BD_setup = BendersDecomp(mh_ref, CONVRG_EPS=0.1, MAX_BD_ITER=10000, INIT_UB=3000000)

    gap_Bd = float("Inf")  # Initialize convergence gap

    # Placeholders initialization (plots/pickle)
    x_current_Bd, x_all_Bd, UB_Bd, LB_Bd, iter_Bd_2_plot = (
        [],
        [],
        [],
        [],
        [],
    )  # Keep the current optimal solution of the Master and the Upper/Lower Bounds of all iterations.

    sb_objective_value = 0  # Keep the objective value of the Sub-problem, for the current solution of the Master.

    if input_options["BENDERS_SINGLE_SUB"]:
        sb_objective_value = 0
    else:
        all_sb_objective_values = [0] * mh_ref.s_operational_nodes * mh_ref.s_sample_size

    UB, LB = my_BD_setup.INIT_UB, my_BD_setup.INIT_LB

    # ---------------------Create an initial plot
    if input_options["DO_CNVRG_PLOT"]:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        fig = plt.figure(figsize=(8, 5))

        (line_UB,) = plt.gca().plot([], [], linestyle="-", color="blue", label="Upper Bound")
        (line_LB,) = plt.gca().plot([], [], linestyle="-", color="red", label="Lower Bound")

        plt.gca().legend()

        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        initial_ylim = (0, 1.1 * my_BD_setup.INIT_UB)  # Set initial y-axis limits
        plt.gca().set_ylim(initial_ylim)

        initial_xlim = (0, 1)  # Set initial x-axis limits - Adjust as needed
        plt.gca().set_xlim(initial_xlim)

        # Add labels and title
        plt.xlabel("Iterations")
        plt.ylabel("Objective function value")
        plt.title("Benders convergence plot")
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show(block=False)

    # Initialize Sub-problem(s)
    if input_options["BENDERS_SINGLE_SUB"]:
        sb_Bd_model = None
    else:
        # Create a list of subproblems (instances) per operational node and time sample.
        sb_Bd_models = [None for n_op in range(mh_ref.s_operational_nodes) for t in range(mh_ref.s_sample_size)]
        nodes_from_sb = my_BD_setup.assign_operational_nodes_to_subproblems(
            list(range(1, mh_ref.s_operational_nodes + 1)), mh_ref.s_sample_size
        )

    # ***********************************************************************************
    # ******************************** MAIN BENDERS LOOP ********************************
    # ***********************************************************************************
    start_time_algorithm = time.time()

    for iter_Bd in range(0, my_BD_setup.MAX_BD_ITER + 1):
        print("\n --------Running Benders loop...")
        if iter_Bd == 0:
            # Create the master problem with only the initial LB (alpha).
            if input_options["BENDERS_SINGLE_CUT"]:
                master_Bd_model, master_Bd_solver = my_BD_setup.create_master_problem(solver=input_options["LP_SOLVER"])
            else:
                master_Bd_model, master_Bd_solver = my_BD_setup.create_master_problem(
                    USE_MULTI_CUTS=not input_options["BENDERS_SINGLE_CUT"], solver=input_options["LP_SOLVER"]
                )

        else:
            # Add a cut to the already defined master_Bd_model for the current iteration. Add one extra contraint of type "CUTS".
            if input_options["BENDERS_SINGLE_CUT"]:
                master_Bd_model.CUTS.add(iter_Bd)
            else:
                current_cut_index = (iter_Bd - 1) * len(sb_Bd_models)
                for sb_i, _ in enumerate(sb_Bd_models, 1):
                    master_Bd_model.CUTS.add(current_cut_index + sb_i)

            if input_options["BENDERS_SINGLE_SUB"]:
                for br in master_Bd_model.branch:
                    for node in master_Bd_model.n_strgc:
                        # Set value of Master parameters based on Sub-problem dual values.
                        master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br, node, iter_Bd].set_value(
                            sb_Bd_model.dual[sb_Bd_model.c_rule_fix_master_var_z_capacity_total[br, node]]
                        )

                # Get the current Sub-problem objective value.
                master_Bd_model.sb_current_objective_value[iter_Bd].set_value(sb_objective_value)

                for br in master_Bd_model.branch:
                    for node in master_Bd_model.n_strgc:
                        # Set value of Master parameters based on previous Master solution (fixed x) - ("x_current_Bd" comes from previous iteration)
                        master_Bd_model.x_fixed_z_capacity_total[br, node, iter_Bd].set_value(
                            max(x_current_Bd["z_capacity_total"][br, node], 0)
                        )

                # Create the cut to be added to the Master.
                cut = master_Bd_model.sb_current_objective_value[iter_Bd]
                cut += sum(
                    master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br, node, iter_Bd]
                    * (
                        master_Bd_model.z_capacity_total[br, node]
                        - master_Bd_model.x_fixed_z_capacity_total[br, node, iter_Bd]
                    )
                    for br in master_Bd_model.branch
                    for node in master_Bd_model.n_strgc
                )

                master_Bd_model.Cut_Defn.add(master_Bd_model.a >= cut)  # Add the cut to the Master.

            else:
                # Loop over the list of different subproblems and add the dual-based contraint from each of them.
                for sb_i, sb_Bd_model_i in enumerate(sb_Bd_models, 1):
                    for br in master_Bd_model.branch:
                        # Set value of Master parameters based on Sub-problem dual values.
                        # Make sure to assign the correct dual to the corresponding master parameter indexed by subproblem (i.e., scenario for L-shaped method).
                        if input_options["BENDERS_SINGLE_CUT"]:
                            master_Bd_model.sb_current_master_solution_dual_z_capacity_total[
                                br, sb_i - 1, iter_Bd
                            ].set_value(sb_Bd_model_i.dual[sb_Bd_model_i.c_rule_fix_master_var_z_capacity_total[br]])
                        else:
                            master_Bd_model.sb_current_master_solution_dual_z_capacity_total[
                                br, sb_i - 1, current_cut_index + sb_i
                            ].set_value(sb_Bd_model_i.dual[sb_Bd_model_i.c_rule_fix_master_var_z_capacity_total[br]])

                # Get the current Sub-problem(s) objective value(s).
                if input_options["BENDERS_SINGLE_CUT"]:
                    master_Bd_model.sb_current_objective_value[iter_Bd].set_value(sum(all_sb_objective_values))

                    for br in master_Bd_model.branch:
                        for node in master_Bd_model.n_strgc:
                            # Set value of Master parameters based on previous Master solution (fixed x) - ("x_current_Bd" comes from previous iteration)
                            master_Bd_model.x_fixed_z_capacity_total[br, node, iter_Bd].set_value(
                                max(x_current_Bd["z_capacity_total"][br, node], 0)
                            )

                    # Create the cut to be added to the Master.
                    cut = master_Bd_model.sb_current_objective_value[iter_Bd]
                    cut += sum(
                        master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br, sb_i, iter_Bd]
                        * (
                            master_Bd_model.z_capacity_total[br, nodes_from_sb[sb_i]]
                            - master_Bd_model.x_fixed_z_capacity_total[br, nodes_from_sb[sb_i], iter_Bd]
                        )
                        for br in master_Bd_model.branch
                        for sb_i in master_Bd_model.numSubProb
                    )

                    master_Bd_model.Cut_Defn.add(master_Bd_model.a >= cut)  # Add the cut to the Master.

                else:
                    # CHECKME: join the for loops (?)
                    for sb_i, _ in enumerate(sb_Bd_models, 1):
                        master_Bd_model.sb_current_objective_value[sb_i - 1, current_cut_index + sb_i].set_value(
                            all_sb_objective_values[sb_i - 1]
                        )

                    for sb_i, _ in enumerate(sb_Bd_models, 1):
                        for br in master_Bd_model.branch:
                            for node in master_Bd_model.n_strgc:
                                # Set value of Master parameters based on previous Master solution (fixed x) - ("x_current_Bd" comes from previous iteration)
                                master_Bd_model.x_fixed_z_capacity_total[br, node, current_cut_index + sb_i].set_value(
                                    max(x_current_Bd["z_capacity_total"][br, node], 0)
                                )

                    for sb_i, _ in enumerate(sb_Bd_models, 1):

                        cut = master_Bd_model.sb_current_objective_value[sb_i - 1, current_cut_index + sb_i]
                        cut += sum(
                            master_Bd_model.sb_current_master_solution_dual_z_capacity_total[
                                br, sb_i - 1, current_cut_index + sb_i
                            ]
                            * (
                                master_Bd_model.z_capacity_total[
                                    br,
                                    mh_ref.dict_map_operational_to_strategic_node[
                                        "n_op_" + str(nodes_from_sb[sb_i - 1])
                                    ],
                                ]
                                - master_Bd_model.x_fixed_z_capacity_total[
                                    br,
                                    mh_ref.dict_map_operational_to_strategic_node[
                                        "n_op_" + str(nodes_from_sb[sb_i - 1])
                                    ],
                                    current_cut_index + sb_i,
                                ]
                            )
                            for br in master_Bd_model.branch
                        )

                        master_Bd_model.Cut_Defn.add(master_Bd_model.a[sb_i - 1] >= cut)  # Add the cut to the Master.

        # ---------------------Solve the Master.
        master_Bd_solver.options["TimeLimit"] = 60 * 5
        master_Bd_solver.solve(master_Bd_model)

        x_current_Bd = extract_all_opt_variable_values(master_Bd_model)  # Store the current Master solution.
        x_all_Bd.append(x_current_Bd)
        LB = master_Bd_model.obj()  # Update Lower Bound (approximated Sub-problem cost)
        LB_Bd.append(LB)

        # ---------------------SUB-PROBLEMS(S)
        if input_options["BENDERS_SINGLE_SUB"]:
            if iter_Bd == 0:
                # Create the Sub-problem (only at the first iteration).
                sb_Bd_model = my_BD_setup.create_single_sub_problem()

                # Create a solver for the sub-problem
                sb_solver = pyo.SolverFactory(input_options["LP_SOLVER"])

            # Fix the x "replica" variables in the Sub-problem with the current (updated) Master solution.
            for br in master_Bd_model.branch:
                for node in master_Bd_model.n_strgc:
                    sb_Bd_model.z_capacity_total_fixed[br, node].set_value(
                        max(x_current_Bd["z_capacity_total"][br, node], 0)
                    )

            # ---------------------Solve the Sub-problem.
            sb_solver_rslt = sb_solver.solve(sb_Bd_model)

            # Check if the sub-problem solver was successful
            if sb_solver_rslt.solver.termination_condition == TerminationCondition.optimal:
                # Get current solution and objective of Sub-problem.
                # y_current_Bd = extract_all_opt_variable_values(sb_Bd_model)
                sb_objective_value = pyo.value(sb_Bd_model.obj)
            else:
                print(f"Sub-problem failed at iteration: {iter_Bd}.")
                break

            # Update the Upper Bound.
            UB = min(UB, master_Bd_model.obj() - master_Bd_model.a.value + sb_Bd_model.obj())

        else:
            # CASE FOR MULTIPLE SUB-PROBLEMS (single/multi-cut)
            # Create the Sub-problem(s) (only at the first iteration).
            if iter_Bd == 0:
                sb_Bd_models = [
                    my_BD_setup.create_sub_problem(n_op, t)
                    for n_op in range(1, mh_ref.s_operational_nodes + 1)
                    for t in range(mh_ref.s_sample_size)
                ]

            # Fix the x "replica" variables in the Sub-problem with the current (updated) Master solution.
            for sb_i, sb_model_i in enumerate(sb_Bd_models):
                for br in master_Bd_model.branch:
                    sb_model_i.z_capacity_total_fixed[br].set_value(
                        max(
                            x_current_Bd["z_capacity_total"][
                                br, mh_ref.dict_map_operational_to_strategic_node["n_op_" + str(nodes_from_sb[sb_i])]
                            ],
                            0,
                        )
                    )

            sb_solver = pyo.SolverFactory(input_options["LP_SOLVER"])

            for sb_i, sb_i_model in enumerate(sb_Bd_models):

                sb_i_solver_rslt = sb_solver.solve(sb_i_model)

                if sb_i_solver_rslt.solver.termination_condition == TerminationCondition.optimal:

                    print(f"Sub-problem {sb_i} solved at iteration: {iter_Bd}.")

                    # Get current solution and objective of Sub-problem.
                    # y_current_Bd = extract_all_opt_variable_values(sb_i_model)
                    all_sb_objective_values[sb_i] = pyo.value(sb_i_model.obj)
                else:
                    print(f"Sub-problem {sb_i} failed at iteration: {iter_Bd}.")
                    break

            # Update the Upper Bound.
            if input_options["BENDERS_SINGLE_CUT"]:
                UB = min(UB, master_Bd_model.obj() - master_Bd_model.a.value + sum(all_sb_objective_values))
            else:
                temp_a_sum = 0
                for i in master_Bd_model.a:
                    temp_a_sum += master_Bd_model.a[i].value
                UB = min(UB, master_Bd_model.obj() - temp_a_sum + sum(all_sb_objective_values))

        UB_Bd.append(UB)
        iter_Bd_2_plot.append(iter_Bd)  # Update the plot.

        if input_options["DO_CNVRG_PLOT"]:

            line_UB.set_data(iter_Bd_2_plot, UB_Bd)
            line_LB.set_data(iter_Bd_2_plot, LB_Bd)

            # CHECKME: Comment-out if i want static y-axis limits
            new_ylim = (0, min(UB_Bd) + 10000)
            plt.gca().set_ylim(new_ylim)

            # Dynamically adjust x-axis limits
            new_xlim = (0, max(iter_Bd_2_plot) + 1)
            plt.gca().set_xlim(new_xlim)

            plt.draw()
            plt.pause(0.1)

        newgap_Bd = abs(UB - LB) / UB * 100
        print(f"Benders gap at iteration {iter_Bd} is {round((UB-LB))} ({round((UB-LB)/UB*100,1)}%).")

        if newgap_Bd > my_BD_setup.CONVRG_EPS:
            gap_Bd = min(gap_Bd, newgap_Bd)
        else:
            print("-------------Benders converged!!!\n")
            break

    else:
        print(f"Max iterations ({iter_Bd}) exceeded.\n")

    end_time_algorithm = time.time()

    elapsed_time_algorithm = end_time_algorithm - start_time_algorithm
    elapsed_time_delta = dtime.timedelta(seconds=elapsed_time_algorithm)  # convert elapsed time to a timedelta object.

    print(f"\n\n----------------------Bender's algorithm took ---> {str(elapsed_time_delta)} (HH:MM:SS).")

    # -----------------------------EXTRACTING RESULTS FROM MASTER PROBLEM-----------------------------
    solution_Master_Bd = extract_all_opt_variable_values(master_Bd_model)
    # solution_Subproblem_Bd = extract_all_opt_variable_values(sb_Bd_model)

    if input_options["DO_CNVRG_PLOT"]:
        plt.tight_layout()
        # ---------------SAVING CONVERGENCE FIGURE
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig2save = (
            input_options["PATH_OUTPUT"]
            / f"benders_convergence_plot_case_{input_options['GRID_CASE']}_stochastic_{input_options['IS_STOCHASTIC']}_samples_{input_options['N_SAMPLES']}_multiCut_{not input_options['BENDERS_SINGLE_CUT']}_fixed_cap_{input_options['MH_USE_FIXED_CAP']}_{timestamp}.pdf"
        )
        fig.savefig(fig2save, format="pdf")
    else:
        print(
            f"\n\ninput_options['DO_CNVRG_PLOT'] = {input_options['DO_CNVRG_PLOT']}) ---> Convergence plot was NOT saved.\n\n"
        )

    return (
        solution_Master_Bd,
        master_Bd_model,
        my_BD_setup,
        elapsed_time_delta,
        temp_a_sum,
        UB_Bd,
        LB_Bd,
        iter_Bd,
        iter_Bd_2_plot,
        x_current_Bd,
    )


def results_log(
    input_options: dict,
    pgim_ref: ReferenceModelCreator,
    optimal_solution_pgim: dict,
    my_pgim_stochastic_model_ef,
    mh_ref: MultiHorizonPgim,
    master_Bd_model,
    temp_a_sum: float,
):
    """Logging results from the different methods.

    Args:
        input_options (dict): User-defined case options.
        pgim_ref (ReferenceModelCreator): Reference `pgim` case.
        optimal_solution_pgim (dict): Optimal solution of the `pgim` reference model.
        my_pgim_stochastic_model_ef (_type_): Reference `pgim` model for the stochastic case (extented formulation).
        mh_ref (MultiHorizonPgim): Multi-horizon formulation of the `pgim` reference case.
        master_Bd_model (_type_): Master problem of the Benders algorithm.
        temp_a_sum (float): Value of the sub-problems(s) value function.

    Returns:
        pgim_obj_val (int): Objective value of the reference `pgim` model.
        pgim_opex (int): OPEX value of the reference `pgim` model.
        mh_obj_val (int): Objective value of the multi-horizon formulation of the reference `pgim` model.
        bd_obj_val (int): Objective value of the Bender's soltuion approach.
        bd_opex (int): OPEX value of the Bender's soltuion approach.
        get_first_stage_decision_pgim (Callable): Function to get the first-stage values from the reference `pgim` model.
    """
    # -----------------------------VALIDATING RESULTS-----------------------------
    if not mh_ref.is_stochastic:

        pgim_obj_val = round(pyo.value(pgim_ref.ref_pgim_model.OBJ))
        mh_obj_val = round(pyo.value(mh_ref.non_decomposed_model.objective))
        bd_obj_val = round(pyo.value(master_Bd_model.obj))

        # OBJECTIVE VALUE
        print(f"PGIM OBJECTIVE VALUE: {pgim_obj_val}")
        print(f"MULTI-HORIZON OBJECTIVE VALUE: {mh_obj_val}")
        print(f"BENDERS OBJECTIVE VALUE: {bd_obj_val}")
        print("\n------------------------------------------------------\n")

        # OPEX
        pgim_opex = round(optimal_solution_pgim["v_operating_cost"].sum())
        print(f"PGIM OPEX: {pgim_opex}")
        if input_options["BENDERS_SINGLE_CUT"]:
            bd_opex = round(master_Bd_model.a.value)
        else:
            bd_opex = round(temp_a_sum)
        print(f"BENDERS OPEX: {bd_opex}")

        def get_first_stage_decision_pgim(opt_sol_pgim):
            x_cbl = (
                opt_sol_pgim["v_branch_new_cables"]
                .xs(2035, level=1)
                .loc[opt_sol_pgim["v_branch_new_cables"].xs(2035, level=1).values > 0.01]
            )
            x_cpt = (
                opt_sol_pgim["v_branch_new_capacity"]
                .xs(2035, level=1)
                .loc[opt_sol_pgim["v_branch_new_capacity"].xs(2035, level=1).values > 0.01]
            )
            x = {"new_lines": x_cbl, "new_capacity": x_cpt}
            return x

    elif mh_ref.is_stochastic:

        pgim_obj_val = round(pyo.value(my_pgim_stochastic_model_ef.EF_Obj))
        mh_obj_val = round(pyo.value(mh_ref.non_decomposed_model.objective))
        bd_obj_val = round(pyo.value(master_Bd_model.obj))

        # OBJECTIVE VALUE
        print(f"PGIM DEF OBJECTIVE VALUE: {pgim_obj_val}")
        print(f"MULTI-HORIZON OBJECTIVE VALUE: {mh_obj_val}")
        print(f"BENDERS OBJECTIVE VALUE: {bd_obj_val}")
        print("\n------------------------------------------------------\n")

        # EXPECTED OPEX
        pgim_opex = round(
            input_options["PROB"]["scen0"] * optimal_solution_pgim[0]["scen0.v_operating_cost"].sum()
            + input_options["PROB"]["scen1"] * optimal_solution_pgim[1]["scen1.v_operating_cost"].sum()
            + input_options["PROB"]["scen2"] * optimal_solution_pgim[2]["scen2.v_operating_cost"].sum()
        )
        print(f"DEF EXPECTED OPEX: {pgim_opex}")
        if input_options["BENDERS_SINGLE_CUT"]:
            bd_opex = round(master_Bd_model.a.value)
        else:
            bd_opex = round(temp_a_sum)
        print(f"BENDERS EXPECTED OPEX: {bd_opex}")

        def get_first_stage_decision_pgim(opt_sol_pgim):
            scen = 0
            x_cbl = (
                opt_sol_pgim[scen][f"scen{scen}.v_branch_new_cables"]
                .xs(2035, level=1)
                .loc[opt_sol_pgim[scen][f"scen{scen}.v_branch_new_cables"].xs(2035, level=1).values > 0.01]
            )
            x_cpt = (
                opt_sol_pgim[scen][f"scen{scen}.v_branch_new_capacity"]
                .xs(2035, level=1)
                .loc[opt_sol_pgim[scen][f"scen{scen}.v_branch_new_capacity"].xs(2035, level=1).values > 0.01]
            )
            x = {"new_lines": x_cbl, "new_capacity": x_cpt}
            return x

    return pgim_obj_val, pgim_opex, mh_obj_val, bd_obj_val, bd_opex, get_first_stage_decision_pgim


def get_first_stage_values(
    get_first_stage_decision_pgim: Callable,
    optimal_solution_pgim: dict,
    mh_ref: MultiHorizonPgim,
    optimal_solution_multi_hor: dict,
    x_current_Bd: dict,
):

    x_pgim = get_first_stage_decision_pgim(optimal_solution_pgim)
    x_mh = mh_ref.get_first_stage_decision(optimal_solution_multi_hor)
    x_bd = mh_ref.get_first_stage_decision(x_current_Bd)

    return x_pgim, x_mh, x_bd


def save_result_files(
    input_options: dict,
    timestamp: str,
    formatted_now_time: str,
    elapsed_time_delta: timedelta,
    my_BD_setup: BendersDecomp,
    pgim_obj_val: int,
    pgim_opex: int,
    optimal_solution_pgim: dict,
    mh_obj_val: int,
    optimal_solution_multi_hor: dict,
    bd_obj_val: int,
    bd_opex: int,
    x_current_Bd: dict,
    UB_Bd: list,
    LB_Bd: list,
    iter_Bd: list,
    iter_Bd_2_plot: list,
    x_pgim: dict,
    x_mh: dict,
    x_bd: dict,
):
    """Saves case results.

    Args:
        input_options (dict): _description_
        timestamp (str): _description_
        formatted_now_time (str): _description_
        elapsed_time_delta (timedelta): _description_
        my_BD_setup (BendersDecomp): _description_
        pgim_obj_val (int): _description_
        pgim_opex (int): _description_
        optimal_solution_pgim (dict): _description_
        mh_obj_val (int): _description_
        optimal_solution_multi_hor (dict): _description_
        bd_obj_val (int): _description_
        bd_opex (int): _description_
        x_current_Bd (dict): _description_
        UB_Bd (list): _description_
        LB_Bd (list): _description_
        iter_Bd (list): _description_
        iter_Bd_2_plot (list): _description_
        x_pgim (dict): _description_
        x_mh (dict): _description_
        x_bd (dict): _description_

    Returns: # TODO
        _type_: _description_
    """

    # ---------------SAVING RESULTS REPORT
    report_to_save = {
        "run_datetime": formatted_now_time,
        "Benders_algorithm": {
            "duration (HH:MM:SS)": str(elapsed_time_delta),
            "LP solver": input_options["LP_SOLVER"],
            "convergence_gap_relative [%] (UB-LB)/UB": my_BD_setup.CONVRG_EPS,
            "convergence_gap_relative_to_OBJ_val [%]": round(
                (my_BD_setup.CONVRG_EPS / 100 * UB_Bd[-1]) / bd_obj_val * 100, 3
            ),
            "termination_iteration": f"{iter_Bd}/{my_BD_setup.MAX_BD_ITER}",
        },
        "configuration": {
            "grid_case": input_options["GRID_CASE"],
            "branches_file_used": input_options["BRANCHES_FILE_NAME"],
            "is_stochastic": input_options["IS_STOCHASTIC"],
            "scneario_probabilities": input_options["PROB"],
            "number_of_samples": input_options["N_SAMPLES"],
            "using_single_cut": input_options["BENDERS_SINGLE_CUT"],
            "using_binary_expansion": input_options["MH_USE_BIN_EXPANS"],
            "using_fixed_capacity_lines": input_options["MH_USE_FIXED_CAP"],
        },
        "solution_and results": {
            "pgim_method": {
                "objective_function_value_(CAPEX+OPEX)": pgim_obj_val,
                "OPEX": pgim_opex,
                "first_stage_decision": {
                    "new_lines": {
                        "index_names": x_pgim["new_lines"].index.names,
                        "index_tuples": [idx for idx in x_pgim["new_lines"].index],
                        "data": x_pgim["new_lines"].to_dict(),
                    },
                    "new_capacity": {
                        "index_names": x_pgim["new_capacity"].index.names,
                        "index_tuples": [idx for idx in x_pgim["new_capacity"].index],
                        "data": x_pgim["new_capacity"].to_dict(),
                    },
                },
            },
            "multi-horizon_method": {
                "objective_function_value_(CAPEX+OPEX)": mh_obj_val,
                "first_stage_decision": {
                    "new_lines": {
                        "index_names": x_mh["new_lines"].index.names,
                        "index_tuples": [idx for idx in x_mh["new_lines"].index],
                        "data": x_mh["new_lines"].to_dict(),
                    },
                    "new_capacity": {
                        "index_names": x_mh["new_capacity"].index.names,
                        "index_tuples": [idx for idx in x_mh["new_capacity"].index],
                        "data": x_mh["new_capacity"].to_dict(),
                    },
                },
            },
            "benders_decomposition_method": {
                "objective_function_value_(CAPEX+OPEX)": bd_obj_val,
                "OPEX": bd_opex,
                "first_stage_decision": {
                    "new_lines": {
                        "index_names": x_bd["new_lines"].index.names,
                        "index_tuples": [idx for idx in x_bd["new_lines"].index],
                        "data": x_bd["new_lines"].to_dict(),
                    },
                    "new_capacity": {
                        "index_names": x_bd["new_capacity"].index.names,
                        "index_tuples": [idx for idx in x_bd["new_capacity"].index],
                        "data": x_bd["new_capacity"].to_dict(),
                    },
                },
            },
        },
    }

    # Write JSON data to a file
    os.makedirs(os.path.dirname(input_options["PATH_OUTPUT"]), exist_ok=True)
    report_file_out_path = input_options["PATH_OUTPUT"] / f"run_Benders_validation_{timestamp}.json"

    with open(report_file_out_path, "w") as file:
        json.dump(report_to_save, file, indent=4)

    print("\n---------------Results file has been written successfully.")

    # ---------------SAVING RESULTS OBJECTS
    results_file_out_path = (
        input_options["PATH_OUTPUT"] / f"run_Benders_validation_{timestamp}.pickle"
    )  # write a PICKLE file for results.

    print(f"{dtime.datetime.now()}: Dumping results to PICKLE file...")
    rslt_to_save = {
        "pgim_rslt": {
            "obj_val": pgim_obj_val,
            "opex": pgim_opex,
            "x_new_lines": x_pgim["new_lines"].to_dict(),
            "x_new_capacity": x_pgim["new_capacity"].to_dict(),
            "optimal_solution": optimal_solution_pgim,
        },
        "mh_rslt": {
            "obj_val": mh_obj_val,
            "x_new_lines": x_mh["new_lines"].to_dict(),
            "x_new_capacity": x_mh["new_capacity"].to_dict(),
            "optimal_solution": optimal_solution_multi_hor,
        },
        "bd_rslt": {
            "obj_val": bd_obj_val,
            "opex": bd_opex,
            "upper_bound": UB_Bd,
            "lower_bound": LB_Bd,
            "iter-to_plot": iter_Bd_2_plot,
            "master_problem_solution": x_current_Bd,
            "x_new_lines": x_bd["new_lines"].to_dict(),
            "x_new_capacity": x_bd["new_capacity"].to_dict(),
        },
    }

    with open(results_file_out_path, "wb") as file:
        cloudpickle.dump(rslt_to_save, file)

    print("\n---------------Results have been pickled successfully.")

    return report_to_save, report_file_out_path


def calc_stochastic_metrics(input_options: dict, timestamp: str, report_to_save: dict, report_file_out_path: str):
    """Calculates common stochastic optimiztion metrics.

    Args:
        input_options (dict): User-defined case options.
        timestamp (str): Timestamp of case run.
        report_to_save (dict): Case result report (`.json`)
        report_file_out_path (str): Report output path.
    """
    # ---------------CALCUATE VSS AND EVPI (from given results file)
    if input_options["DO_CALC_VSS_EVPI"]:

        rslt_file_name = "run_Benders_validation_" + timestamp + ".pickle"
        report_file_name = "run_Benders_validation_" + timestamp + ".json"

        rslt_file_path = input_options["PATH_OUTPUT"] / rslt_file_name
        report_file_path = input_options["PATH_OUTPUT"] / report_file_name

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
        print(f"\n RP = {round(RP_OBJ_VAL)}")

        current_vss, current_eev = calc_vss(
            run_settings,
            RP_OBJ_VAL,
            input_options["PATH_INPUT"],
            input_options["PATH_OUTPUT"],
            lp_solver=input_options["LP_SOLVER"],
        )

        print(f"\n EEV = {round(current_eev)}")
        print(f"\n VSS = {round(current_vss)}")

        current_evpi, current_ws = calc_evpi(
            run_settings,
            RP_OBJ_VAL,
            input_options["PATH_INPUT"],
            input_options["PATH_OUTPUT"],
            lp_solver=input_options["LP_SOLVER"],
        )

        print(f"\n WS = {round(current_ws)}")
        print(f"\n EVPI = {round(current_evpi)}")

        report_to_save["solution_and results"]["multi-horizon_method"]["stochastic_metrics"] = {
            "RP": RP_OBJ_VAL,
            "EEV": current_eev,
            "VSS": current_vss,
            "WS": current_ws,
            "EVPI": current_evpi,
        }
        with open(report_file_out_path, "w") as file:
            json.dump(report_to_save, file, indent=4)

        print("\n---------------Results file has been modified successfully.")


def visualize_results(input_options: dict, timestamp: str):
    """Visualization of the results.

    Args:
        input_options (dict): User-defined case options.
        timestamp (str): Timestamp of case run.
    """
    rslt_file_name = "run_Benders_validation_" + timestamp + ".pickle"
    report_file_name = "run_Benders_validation_" + timestamp + ".json"

    rslt_file_path = input_options["PATH_OUTPUT"] / rslt_file_name
    report_file_path = input_options["PATH_OUTPUT"] / report_file_name

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

    if input_options["DO_VIZ"]:
        if input_options["GRID_CASE"] == "baseline":
            create_interactive_maps_results(
                run_settings, run_rslt, input_options["PATH_INPUT"], input_options["PATH_OUTPUT"]
            )
            create_geopandas_figures_results(
                timestamp, run_settings, run_rslt, input_options["PATH_INPUT"], input_options["PATH_OUTPUT"]
            )

        print("\n---------------Graphics have been geenrated successfully.")


def run_case_bd(
    now_time: datetime,
    input_options: dict = {
        "PATH_INPUT": Path(__file__).parents[3] / "examples" / "inputs" / "CASE_BASELINE",
        "PATH_OUTPUT": Path(__file__).parents[3] / "examples" / "outputs",
        "BRANCHES_FILE_NAME": "branches_reduced.csv",
        "GRID_CASE": "star",
        "IS_STOCHASTIC": False,
        "N_SAMPLES": 2,
        "BENDERS_SINGLE_SUB": False,
        "BENDERS_SINGLE_CUT": False,
        "MH_USE_BIN_EXPANS": False,
        "MH_USE_FIXED_CAP": False,
        "LP_SOLVER": "gurobi",
        "DO_CALC_VSS_EVPI": False,
        "DO_CNVRG_PLOT": False,
        "DO_VIZ": False,
    },
):
    """Main user-interface to apply Benders solution approach for a given set of options.

    Args:
        now_time (datetime): Date and time of case run.
        input_options (_type_, optional): User-defined case options..
    """

    formatted_now_time = now_time.strftime("%Y-%B-%A %H:%M:%S")
    timestamp = now_time.strftime("%Y%m%d_%H%M%S")

    print(f"\n\nRunning case...timestamp: {timestamp}")

    pgim_ref = pgim_ref_init(input_options)

    optimal_solution_multi_hor, mh_ref = do_mh(input_options, pgim_ref)

    optimal_solution_pgim, my_pgim_stochastic_model_ef = solve_pgim_ref(input_options, pgim_ref, mh_ref)

    (
        solution_Master_Bd,
        master_Bd_model,
        my_BD_setup,
        elapsed_time_delta,
        temp_a_sum,
        UB_Bd,
        LB_Bd,
        iter_Bd,
        iter_Bd_2_plot,
        x_current_Bd,
    ) = do_bd(input_options, mh_ref, timestamp)

    pgim_obj_val, pgim_opex, mh_obj_val, bd_obj_val, bd_opex, get_first_stage_decision_pgim = results_log(
        input_options, pgim_ref, optimal_solution_pgim, my_pgim_stochastic_model_ef, mh_ref, master_Bd_model, temp_a_sum
    )

    x_pgim, x_mh, x_bd = get_first_stage_values(
        get_first_stage_decision_pgim, optimal_solution_pgim, mh_ref, optimal_solution_multi_hor, x_current_Bd
    )

    report_to_save, report_file_out_path = save_result_files(
        input_options,
        timestamp,
        formatted_now_time,
        elapsed_time_delta,
        my_BD_setup,
        pgim_obj_val,
        pgim_opex,
        optimal_solution_pgim,
        mh_obj_val,
        optimal_solution_multi_hor,
        bd_obj_val,
        bd_opex,
        x_current_Bd,
        UB_Bd,
        LB_Bd,
        iter_Bd,
        iter_Bd_2_plot,
        x_pgim,
        x_mh,
        x_bd,
    )

    calc_stochastic_metrics(input_options, timestamp, report_to_save, report_file_out_path)

    visualize_results(input_options, timestamp)

    return pgim_obj_val, mh_obj_val, bd_obj_val, pgim_opex, bd_opex
