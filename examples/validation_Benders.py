#%% -*- coding: utf-8 -*-
"""
Example script on using the `powergim.dcmp_tools` subpackage.
---
This creates a case and solves it with all 3 methods: 
- pgim (default), 
- mh (multi-horizon reformulation)
- bd (using the developed Bender's decompostion).

The methods are compared against each other to check and validate that the decomposition worked as expected.

This could be a test file.

@author: spyridonc
"""
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import powergim as pgim
import powergim.dcmp_tools
import mpisppy.opt.ph
import mpisppy.utils.sputils
from pathlib import Path
import copy
import math
import datetime as dtime
from datetime import datetime
import time
import json
import os
import cloudpickle

PATH_INPUT  = Path(__file__).parents[0] / "inputs"/"CASE_BASELINE"
PATH_OUTPUT = Path(__file__).parents[0] / "outputs"

#%%
#-----------------------------INITIALIZING INPUTS-----------------------------
BRANCHES_FILE_NAME = "branches_reduced.csv" # NOTE: Define the branch CSV input file to use.
GRID_CASE          = 'baseline'             # NOTE: Define case to run: 'star'/'baseline'.

IS_STOCHASTIC      = True    # NOTE: If 'True', solves the 'stochastic' problem and if 'False' solves the 'deterministic' one.
N_SAMPLES          = 2        # NOTE: For demonstration, use 2 samples. For paper results use 300 ('baseline' case).
BENDERS_SINGLE_SUB = False    # NOTE: If True, Benders uses the non-decomposed (single) sub-problem. If False Benders uses the decomposed subproblems.
BENDERS_SINGLE_CUT = False    # NOTE: If True, Benders uses a single-cut algorithm. If False Benders uses the multi-cut version.
MH_USE_BIN_EXPANS  = False    # NOTE: If True, integers are modelled using their binary expansion instead (recommend to keep 'False').
MH_USE_FIXED_CAP   = False    # NOTE: If True, branches are of fixed, specified capacity. If True, branch capacity is a variable (keep 'False' for paper results).
LP_SOLVER          = 'gurobi' # NOTE: Solver for LP(s): 'gurobi' or 'appsi_highs'.
DO_CALC_VSS_EVPI   = True    # NOTE: Activates the calculation of VSS and EVPI for the stochastic cases.
DO_VIZ             = True     # NOTE: Activates the visualization of results (interactive html maps and paper figures).

# NOTE: These probabilities were used in the paper.
if GRID_CASE == 'baseline':
    PROB = {"scen0": 0.1, "scen1": 0.45, "scen2": 0.45}
elif GRID_CASE == 'star':
    PROB = {"scen0": 0.8, "scen1": 0.1, "scen2": 0.1}

#-------------# NOTE: Sanity checks for the user-defined inputs.
if not IS_STOCHASTIC: DO_CALC_VSS_EVPI = False
if BENDERS_SINGLE_SUB: BENDERS_SINGLE_CUT=True
if (GRID_CASE == 'star') or (MH_USE_BIN_EXPANS): DO_VIZ = False # NOTE: Results vizualization is currently NOT supported for the 'star' case nor if "binary expansion" is used.

#%%
#---------------------GETTING/CREATING REQUIRED TIMESERIES CSV FILE-----------------------------
if GRID_CASE == 'baseline':
    powergim.dcmp_tools.TimeseriesBuilder(PATH_INPUT).create_csv() # checking/creating existence of 1 year-long timeseries CSV for pgim.
#%% 
#-----------------------------CREATING BASELINE MODEL-----------------------------
pgim_ref = powergim.dcmp_tools.ReferenceModelCreator(grid_case=GRID_CASE,
                                                     branches_file_name=BRANCHES_FILE_NAME,
                                                     s_sample_size=N_SAMPLES,
                                                     probabilities=PROB
                                                     )
pgim_ref.create_reference_pgim_model(PATH_INPUT,PATH_OUTPUT,scenario_selection='default')
#%%
#-----------------------------CREATING MULTIHORIZON TREE STRUCTURES-----------------------------
mh_ref = powergim.dcmp_tools.MultiHorizonPgim(pgim_ref, is_stochastic=IS_STOCHASTIC)
mh_ref.create_ancestors_struct()
#%% 
#-----------------------------VISUALIZING CASE: INPUT DATA-----------------------------
if DO_VIZ and GRID_CASE == 'baseline':
    case_visualizer = powergim.dcmp_tools.CaseMapVisualizer(pgim_ref,outFileName='case_map_interactive.html',outPath=PATH_OUTPUT)
    case_visualizer.create_html_map()
#%%
#-----------------------------CREATING MULTI-HORIZON PROBLEM FROM PGIM DATA-----------------------------
mh_ref.get_pgim_params_per_scenario()
mh_ref.create_multi_horizon_problem(USE_BIN_EXPANS=MH_USE_BIN_EXPANS,USE_FIXED_CAP_LINES=MH_USE_FIXED_CAP)
#%%
#-----------------------------SOLVING MULTI-HORIZON PROBLEM-----------------------------
solver = pyo.SolverFactory('gurobi')
# Set Gurobi solver parameters
solver.options['TimeLimit'] = 60*60*3   # NOTE: TimeLimit = 60*60*3 seconds for paper results.
solver.options['MIPGap'] = 0.00001      # NOTE: MIPGap = 0.00001 for paper results.

results = solver.solve(mh_ref.non_decomposed_model,tee=True,keepfiles=False,symbolic_solver_labels=True)
 
if str(results.solver.termination_condition) != "optimal": print(results.solver.termination_condition)
     
optimal_solution_multi_hor = powergim.dcmp_tools.extract_all_opt_variable_values(mh_ref.non_decomposed_model) # NOTE: pgim method extract_all_variable_values() should be more generic (not only for pgim objects) - using the 'generalized' function extract_all_opt_variable_values() instead.
#%%
#-----------------------------SOLVING PGIM MODEL (FOR VALIDATION)-----------------------------
if not IS_STOCHASTIC:
    
    results = solver.solve(pgim_ref.ref_pgim_model,tee=True,keepfiles=False,symbolic_solver_labels=True)
    
    if str(results.solver.termination_condition) != "optimal": print(results.solver.termination_condition)
    
    optimal_solution_pgim = pgim_ref.ref_pgim_model.extract_all_variable_values()

elif IS_STOCHASTIC:
    # Define 3 scenarios (base case and 2 more) --> 7 strategic nodes

    print("\n --------Creating scenarios for DEF")
    # -----------------------Create scenario powerGIM models-----------------------
    # This function edits the inital grid_data and parameters depending on the scenario.
    # Then it creates a model instance for the scenario with the modified parameters.

    # CHECKME: Comment/Uncomment the corresponding lines to use the hardcoded DEFAULT/ALTERNATIVE SCENARIOS.
    if mh_ref.grid_case == 'star':

        def my_mpisppy_scenario_creator(scenario_name):
            """Create a scenario."""
            print("\n Scenario {}".format(scenario_name))

            parameter_data = copy.deepcopy(mh_ref.pgim_case.ref_params)
            grid_data = copy.deepcopy(mh_ref.pgim_case.ref_grid_data)

            match scenario_name:
                case "scen0":
                    pass

                # --------DEFAULT SCENARIOS

                case "scen1": # The "Low wind", "same demand" scenario

                    # Half the wind at n1 (wind farm node).
                    init_wind_capacity = grid_data.generator.loc[grid_data.generator['node'] == 'n1']
                    
                    for iperiod in parameter_data['parameters']['investment_years']:
                        grid_data.generator.loc[grid_data.generator['node'] == 'n1', ['capacity_' + str(iperiod)]] = 0.5*init_wind_capacity.loc[0,'capacity_' + str(iperiod)]
                

                case "scen2": # The "same wind", "high demand" scenario

                    # Double the load at n3 (offshore load node).
                    init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n3']
                    
                    for iperiod in parameter_data['parameters']['investment_years']:
                        grid_data.consumer.loc[grid_data.consumer['node'] == 'n3', ['demand_' + str(iperiod)]] = 2*init_load_capacity.loc[1,'demand_' + str(iperiod)]


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
            pgim_model = pgim.SipModel(grid_data, parameter_data) # A) Initialize a pgim object instane (pgim_model)
            pgim_scen_model = pgim_model.scenario_creator(scenario_name, probability=PROB[scenario_name]) # B) Use scenario_creator method to build a scenario instance model
            
            return pgim_scen_model
    
    # CHECKME: Comment/Uncomment the corresponding lines to use the hardcoded scenarios #1, #2 or #3, as examples.
    else:
        def my_mpisppy_scenario_creator(scenario_name):
            """Create a scenario."""
            print("\n Scenario {}".format(scenario_name))

            parameter_data = copy.deepcopy(pgim_ref.ref_params)
            grid_data = copy.deepcopy(pgim_ref.ref_grid_data)

            match scenario_name:
                case "scen0":
                    pass
                
                case "scen1": # oceangrid_A1_lessdemand
                    #  Less NO demand (220 TWh in 2050)

                    demand_scale = 220/260
                    # demand_scale = 130/260
                    m_demand = grid_data.consumer["node"].str.startswith("NO_")
                    for year in parameter_data['parameters']['investment_years']:

                        # -----Hardcoded scenario #1
                        grid_data.consumer.loc[m_demand, f"demand_{year}"] = demand_scale * grid_data.consumer.loc[m_demand, f"demand_{year}"]
                        
                        # -----Hardcoded scenario #2
                        # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 0.5 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]
                        
                        # -----Hardcoded scenario #3
                        # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 2 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                case "scen2": # oceangrid_A2_moredemand
                    # More NO demand (340 TWh in 2050)

                    demand_scale = 340/260
                    # demand_scale = 520/260
                    m_demand = grid_data.consumer["node"].str.startswith("NO_")
                    for year in parameter_data['parameters']['investment_years']:
                        
                        # -----Hardcoded scenario #1
                        grid_data.consumer.loc[m_demand, f"demand_{year}"] = demand_scale * grid_data.consumer.loc[m_demand, f"demand_{year}"]
                        
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
            pgim_scen_model = pgim_model.scenario_creator(scenario_name, probability=PROB[scenario_name])
            
            return pgim_scen_model

    # This formulates the extensive form based on the scenarios that are defined (an mpisppy object)
    # Preferred method based on mpispppy: mpisppy.opt.ef.ExtensiveForm --> needs mpi4py
    my_pgim_stochastic_model_ef = mpisppy.utils.sputils.create_EF(mh_ref.scenario_names,scenario_creator=my_mpisppy_scenario_creator)

    # Solve the EF
    solver = pyo.SolverFactory("gurobi")
    solver.options['TimeLimit'] = 60*60*3 # seconds
    solver.options['MIPGap'] = 0.00001

    solver.solve(my_pgim_stochastic_model_ef,tee=True,symbolic_solver_labels=True)

    all_var_all_scen_values = []

    # Extract results:
    for scen in mpisppy.utils.sputils.ef_scenarios(my_pgim_stochastic_model_ef):
        # Iterable has 2 dimensions: (scenario_name, scnenario_model (associated pyomo model variables))
        scen_name = scen[0]
        this_scen = scen[1]
        all_var_values = pgim.SipModel.extract_all_variable_values(this_scen)
        all_var_all_scen_values.append(all_var_values)
        
    optimal_solution_pgim = all_var_all_scen_values

# NOTE: The scenario function above is kept in this script intetnionally because it should not be part or a functionality of the multi-horizon formulation.
# It should be a stand-alone because scenarios are always user-defined.
#%%
#-----------------------------RUNNING BENDERS DECOMPOSITION ALGORITHM-----------------------------
# NOTE: Option to deactivate running the Benders algorithm section: MAX_BD_ITER=1
# NOTE: Make sure that INIT_UB is 'big enough' for Benders not to diverge.
# NOTE: CONVRG_EPS is the desired relative convergence gap [%].
# NOTE: Which LP_SOLVER i choose matters for the convergence plot.

match GRID_CASE:
        case 'star':
            match IS_STOCHASTIC:
                case False:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.01,MAX_BD_ITER=1000,INIT_UB=30000)
                case True:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.01,MAX_BD_ITER=1000,INIT_UB=30000)
        case 'baseline':
            match IS_STOCHASTIC:
                case False:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.1,MAX_BD_ITER=1000,INIT_UB=800000) 
                case True:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.1,MAX_BD_ITER=10000,INIT_UB=3000000) # NOTE: For paper results: CONVRG_EPS=0.1, INIT_UB=3000000

gap_Bd = float('Inf') # Initialize convergence gap 

# Placeholders initialization (plots/pickle)
x_current_Bd, x_all_Bd, UB_Bd, LB_Bd, iter_Bd_2_plot = [], [], [], [], []  # Keep the current optimal solution of the Master and the Upper/Lower Bounds of all iterations.

sb_objective_value = 0 # Keep the objective value of the Sub-problem, for the current solution of the Master.

if BENDERS_SINGLE_SUB:
    sb_objective_value = 0
else:
    all_sb_objective_values = [0] * mh_ref.s_operational_nodes * mh_ref.s_sample_size   

UB, LB = my_BD_setup.INIT_UB, my_BD_setup.INIT_LB

# ---------------------Create an initial plot
fig = plt.figure(figsize=(8,5))

line_UB, = plt.gca().plot([], [], linestyle='-', color='blue', label='Upper Bound')
line_LB, = plt.gca().plot([], [], linestyle='-', color='red', label='Lower Bound')

plt.gca().legend()

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

initial_ylim = (0, 1.1*my_BD_setup.INIT_UB) # Set initial y-axis limits
plt.gca().set_ylim(initial_ylim)

initial_xlim = (0, 1)  # Set initial x-axis limits - Adjust as needed
plt.gca().set_xlim(initial_xlim)

# Add labels and title
plt.xlabel('Iterations')
plt.ylabel('Objective function value')
plt.title('Benders convergence plot')
plt.legend()

# Show the plot
plt.grid(True)
plt.show(block=False)  


# Initialize Sub-problem(s)
if BENDERS_SINGLE_SUB:
    sb_Bd_model = None
else:
    # Create a list of subproblems (instances) per operational node and time sample.
    sb_Bd_models = [None for n_op in range(mh_ref.s_operational_nodes) for t in range(mh_ref.s_sample_size)]
    nodes_from_sb = my_BD_setup.assign_operational_nodes_to_subproblems(list(range(1,mh_ref.s_operational_nodes+1)), mh_ref.s_sample_size)


#***********************************************************************************
#******************************** MAIN BENDERS LOOP ********************************
#***********************************************************************************
start_time_algorithm = time.time()


for iter_Bd in range(0, my_BD_setup.MAX_BD_ITER + 1):
    print("\n --------Running Benders loop...")
    if iter_Bd == 0:
        # Create the master problem with only the initial LB (alpha).
        if BENDERS_SINGLE_CUT:
            master_Bd_model, master_Bd_solver = my_BD_setup.create_master_problem()
        else:
            master_Bd_model, master_Bd_solver = my_BD_setup.create_master_problem(USE_MULTI_CUTS=not BENDERS_SINGLE_CUT)

    else:      
        # Add a cut to the already defined master_Bd_model for the current iteration. Add one extra contraint of type "CUTS".
        if BENDERS_SINGLE_CUT:
            master_Bd_model.CUTS.add(iter_Bd)
        else:
            current_cut_index = (iter_Bd-1)*len(sb_Bd_models)
            for sb_i, _ in enumerate(sb_Bd_models,1):
                master_Bd_model.CUTS.add(current_cut_index+sb_i)


        if BENDERS_SINGLE_SUB:
            for br in master_Bd_model.branch:
                for node in master_Bd_model.n_strgc:
                    # Set value of Master parameters based on Sub-problem dual values.
                    master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,node,iter_Bd].set_value(sb_Bd_model.dual[sb_Bd_model.c_rule_fix_master_var_z_capacity_total[br,node]])

            # Get the current Sub-problem objective value.
            master_Bd_model.sb_current_objective_value[iter_Bd].set_value(sb_objective_value)

            for br in master_Bd_model.branch:
                for node in master_Bd_model.n_strgc:
                    # Set value of Master parameters based on previous Master solution (fixed x) - ("x_current_Bd" comes from previous iteration)
                    master_Bd_model.x_fixed_z_capacity_total[br,node,iter_Bd].set_value(max(x_current_Bd['z_capacity_total'][br,node],0))
            
            # Create the cut to be added to the Master.       
            cut =  master_Bd_model.sb_current_objective_value[iter_Bd]        
            cut += sum(master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,node,iter_Bd] * 
                    (master_Bd_model.z_capacity_total[br,node] - master_Bd_model.x_fixed_z_capacity_total[br,node,iter_Bd]) 
                    for br in master_Bd_model.branch for node in master_Bd_model.n_strgc)
            
            master_Bd_model.Cut_Defn.add(master_Bd_model.a >= cut) # Add the cut to the Master.

        else:
            # Loop over the list of different subproblems and add the dual-based contraint from each of them.               
            for sb_i, sb_Bd_model_i in enumerate(sb_Bd_models,1): 
                for br in master_Bd_model.branch:
                    # Set value of Master parameters based on Sub-problem dual values.
                    # Make sure to assign the correct dual to the corresponding master parameter indexed by subproblem (i.e., scenario for L-shaped method).
                    if BENDERS_SINGLE_CUT:
                        master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,sb_i-1,iter_Bd].set_value(sb_Bd_model_i.dual[sb_Bd_model_i.c_rule_fix_master_var_z_capacity_total[br]])
                    else:
                        master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,sb_i-1,current_cut_index+sb_i].set_value(sb_Bd_model_i.dual[sb_Bd_model_i.c_rule_fix_master_var_z_capacity_total[br]])

            # Get the current Sub-problem(s) objective value(s).
            if BENDERS_SINGLE_CUT:
                master_Bd_model.sb_current_objective_value[iter_Bd].set_value(sum(all_sb_objective_values))

                for br in master_Bd_model.branch:
                    for node in master_Bd_model.n_strgc:
                        # Set value of Master parameters based on previous Master solution (fixed x) - ("x_current_Bd" comes from previous iteration)
                        master_Bd_model.x_fixed_z_capacity_total[br,node,iter_Bd].set_value(max(x_current_Bd['z_capacity_total'][br,node],0))

                # Create the cut to be added to the Master.       
                cut =  master_Bd_model.sb_current_objective_value[iter_Bd]        
                cut += sum(master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,sb_i,iter_Bd] * 
                        (master_Bd_model.z_capacity_total[br,nodes_from_sb[sb_i]] - master_Bd_model.x_fixed_z_capacity_total[br,nodes_from_sb[sb_i],iter_Bd]) 
                        for br in master_Bd_model.branch for sb_i in master_Bd_model.numSubProb)
                
                master_Bd_model.Cut_Defn.add(master_Bd_model.a >= cut) # Add the cut to the Master.

            else:
                # CHECKME: join the for loops (?)
                for sb_i, _ in enumerate(sb_Bd_models,1): 
                    master_Bd_model.sb_current_objective_value[sb_i-1,current_cut_index+sb_i].set_value(all_sb_objective_values[sb_i-1])

                for sb_i, _ in enumerate(sb_Bd_models,1): 
                    for br in master_Bd_model.branch:
                        for node in master_Bd_model.n_strgc:
                            # Set value of Master parameters based on previous Master solution (fixed x) - ("x_current_Bd" comes from previous iteration)
                            master_Bd_model.x_fixed_z_capacity_total[br,node,current_cut_index+sb_i].set_value(max(x_current_Bd['z_capacity_total'][br,node],0))

                for sb_i, _ in enumerate(sb_Bd_models,1): 
            
                    cut =  master_Bd_model.sb_current_objective_value[sb_i-1,current_cut_index+sb_i]        
                    cut += sum(master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,sb_i-1,current_cut_index+sb_i] * 
                                (master_Bd_model.z_capacity_total[br,mh_ref.dict_map_operational_to_strategic_node['n_op_' + str(nodes_from_sb[sb_i-1])]] - \
                                master_Bd_model.x_fixed_z_capacity_total[br,mh_ref.dict_map_operational_to_strategic_node['n_op_' + str(nodes_from_sb[sb_i-1])],current_cut_index+sb_i]) 
                            for br in master_Bd_model.branch)
                    
                    master_Bd_model.Cut_Defn.add(master_Bd_model.a[sb_i-1] >= cut) # Add the cut to the Master.
    
    #---------------------Solve the Master.
    master_Bd_solver.options['TimeLimit'] = 60*5 # NOTE: 'TimeLimit' = 60*5 seconds for paper results.
    master_Bd_solver_rslt = master_Bd_solver.solve(master_Bd_model)

    x_current_Bd = powergim.dcmp_tools.extract_all_opt_variable_values(master_Bd_model) # Store the current Master solution.
    x_all_Bd.append(x_current_Bd)
    LB = master_Bd_model.obj() # Update Lower Bound (approximated Sub-problem cost)
    LB_Bd.append(LB)

    # ---------------------SUB-PROBLEMS(S)
    if BENDERS_SINGLE_SUB:
        if iter_Bd == 0:
        # Create the Sub-problem (only at the first iteration).
            sb_Bd_model = my_BD_setup.create_single_sub_problem()

            # Create a solver for the sub-problem
            sb_solver = pyo.SolverFactory(LP_SOLVER)
        
        # Fix the x "replica" variables in the Sub-problem with the current (updated) Master solution.   
        for br in master_Bd_model.branch:
            for node in master_Bd_model.n_strgc:
                sb_Bd_model.z_capacity_total_fixed[br,node].set_value(max(x_current_Bd['z_capacity_total'][br,node],0))
        
        # ---------------------Solve the Sub-problem.
        sb_solver_rslt = sb_solver.solve(sb_Bd_model)     

        # Check if the sub-problem solver was successful
        if sb_solver_rslt.solver.termination_condition == TerminationCondition.optimal:
            # Get current solution and objective of Sub-problem.
            y_current_Bd       = powergim.dcmp_tools.extract_all_opt_variable_values(sb_Bd_model)
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
            sb_Bd_models = [my_BD_setup.create_sub_problem(n_op,t) for n_op in range(1,mh_ref.s_operational_nodes+1) for t in range(mh_ref.s_sample_size)]
    
        # Fix the x "replica" variables in the Sub-problem with the current (updated) Master solution.   
        for sb_i, sb_model_i in enumerate(sb_Bd_models):
            for br in master_Bd_model.branch:
                sb_model_i.z_capacity_total_fixed[br].set_value(max(x_current_Bd['z_capacity_total'][br,mh_ref.dict_map_operational_to_strategic_node['n_op_' + str(nodes_from_sb[sb_i])]],0))

        sb_solver = pyo.SolverFactory(LP_SOLVER)
        
        for sb_i, sb_i_model in enumerate(sb_Bd_models):
            
            sb_i_solver_rslt = sb_solver.solve(sb_i_model)

            if sb_i_solver_rslt.solver.termination_condition == TerminationCondition.optimal:

                print(f"Sub-problem {sb_i} solved at iteration: {iter_Bd}.")
                
                # Get current solution and objective of Sub-problem.
                y_current_Bd = powergim.dcmp_tools.extract_all_opt_variable_values(sb_i_model)
                all_sb_objective_values[sb_i] = pyo.value(sb_i_model.obj)
            else:
                print(f"Sub-problem {sb_i} failed at iteration: {iter_Bd}.")
                break

        # Update the Upper Bound.
        if BENDERS_SINGLE_CUT:
            UB = min(UB, master_Bd_model.obj() - master_Bd_model.a.value + sum(all_sb_objective_values))
        else:
            temp_a_sum = 0
            for i in master_Bd_model.a: temp_a_sum += master_Bd_model.a[i].value
            UB = min(UB, master_Bd_model.obj() - temp_a_sum + sum(all_sb_objective_values))
        

    UB_Bd.append(UB)
    
    iter_Bd_2_plot.append(iter_Bd) # Update the plot.

    line_UB.set_data(iter_Bd_2_plot, UB_Bd)
    line_LB.set_data(iter_Bd_2_plot, LB_Bd)
    
    # CHECKME: Comment-out if i want static y-axis limits
    new_ylim = (0, min(UB_Bd)+10000) 
    plt.gca().set_ylim(new_ylim)
    
    # Dynamically adjust x-axis limits
    new_xlim = (0, max(iter_Bd_2_plot)+1)
    plt.gca().set_xlim(new_xlim)

    plt.draw()
    plt.pause(0.1) 
    
    newgap_Bd = abs(UB-LB)/UB*100
    print(f"Benders gap at iteration {iter_Bd} is {round((UB-LB))} ({round((UB-LB)/UB*100,1)}%).")
    
    if (newgap_Bd > my_BD_setup.CONVRG_EPS):
        gap_Bd = min(gap_Bd, newgap_Bd)
    else:
        print("-------------Benders converged!!!\n")
        break
    
else:
    print(f"Max iterations ({iter_Bd}) exceeded.\n")

end_time_algorithm = time.time()

elapsed_time_algorithm = end_time_algorithm-start_time_algorithm
elapsed_time_delta = dtime.timedelta(seconds=elapsed_time_algorithm) # convert elapsed time to a timedelta object.

print(f"\n\n----------------------Bender's algorithm took ---> {str(elapsed_time_delta)} (HH:MM:SS).")

#%%
#-----------------------------EXTRACTING RESULTS FROM MASTER PROBLEM-----------------------------
solution_Master_Bd = powergim.dcmp_tools.extract_all_opt_variable_values(master_Bd_model)
# solution_Subproblem_Bd = extract_all_opt_variable_values(sb_Bd_model) # CHECKME: Uncomment if i want to get the solutions from all subproblems.
#%%
#-----------------------------VALIDATING RESULTS-----------------------------  
if not mh_ref.is_stochastic:

    pgim_obj_val = round(pyo.value(pgim_ref.ref_pgim_model.OBJ))
    mh_obj_val   = round(pyo.value(mh_ref.non_decomposed_model.objective))
    bd_obj_val   = round(pyo.value(master_Bd_model.obj))

    # OBJECTIVE VALUE
    print(f'PGIM OBJECTIVE VALUE: {pgim_obj_val}')
    print(f'MULTI-HORIZON OBJECTIVE VALUE: {mh_obj_val}')
    print(f'BENDERS OBJECTIVE VALUE: {bd_obj_val}')
    print('\n------------------------------------------------------\n')

    # OPEX
    pgim_opex = round(optimal_solution_pgim["v_operating_cost"].sum())
    print(f'PGIM OPEX: {pgim_opex}')
    if BENDERS_SINGLE_CUT:
        bd_opex = round(master_Bd_model.a.value)
    else:
        bd_opex = round(temp_a_sum)
    print(f'BENDERS OPEX: {bd_opex}')

    def get_first_stage_decision_pgim(opt_sol_pgim):
        x_cbl = opt_sol_pgim['v_branch_new_cables'].xs(2035, level=1).loc[opt_sol_pgim['v_branch_new_cables'].xs(2035, level=1).values>0.01]
        x_cpt = opt_sol_pgim['v_branch_new_capacity'].xs(2035, level=1).loc[opt_sol_pgim['v_branch_new_capacity'].xs(2035, level=1).values>0.01]
        x = {'new_lines':x_cbl, 'new_capacity':x_cpt}
        return x

elif mh_ref.is_stochastic:   

    pgim_obj_val = round(pyo.value(my_pgim_stochastic_model_ef.EF_Obj))
    mh_obj_val   = round(pyo.value(mh_ref.non_decomposed_model.objective))
    bd_obj_val   = round(pyo.value(master_Bd_model.obj))

    # OBJECTIVE VALUE
    print(f'PGIM DEF OBJECTIVE VALUE: {pgim_obj_val}')
    print(f'MULTI-HORIZON OBJECTIVE VALUE: {mh_obj_val}')
    print(f'BENDERS OBJECTIVE VALUE: {bd_obj_val}')
    print('\n------------------------------------------------------\n')

    # EXPECTED OPEX
    pgim_opex = round(  PROB['scen0']*all_var_all_scen_values[0]["scen0.v_operating_cost"].sum() + \
                        PROB['scen1']*all_var_all_scen_values[1]["scen1.v_operating_cost"].sum()+ \
                        PROB['scen2']*all_var_all_scen_values[2]["scen2.v_operating_cost"].sum()
                        )
    print(f'DEF EXPECTED OPEX: {pgim_opex}')
    if BENDERS_SINGLE_CUT:
        bd_opex = round(master_Bd_model.a.value)
    else:
        bd_opex = round(temp_a_sum)
    print(f'BENDERS EXPECTED OPEX: {bd_opex}')

    # CHECKME: Shouldn't this be a pgim method?
    def get_first_stage_decision_pgim(opt_sol_pgim):
        scen = 0
        x_cbl = opt_sol_pgim[scen][f'scen{scen}.v_branch_new_cables'].xs(2035, level=1).loc[opt_sol_pgim[scen][f'scen{scen}.v_branch_new_cables'].xs(2035, level=1).values>0.01]
        x_cpt = opt_sol_pgim[scen][f'scen{scen}.v_branch_new_capacity'].xs(2035, level=1).loc[opt_sol_pgim[scen][f'scen{scen}.v_branch_new_capacity'].xs(2035, level=1).values>0.01]
        x = {'new_lines':x_cbl, 'new_capacity':x_cpt}
        return x

x_pgim = get_first_stage_decision_pgim(optimal_solution_pgim)
x_mh   = mh_ref.get_first_stage_decision(optimal_solution_multi_hor)
x_bd   = mh_ref.get_first_stage_decision(x_current_Bd)

plt.tight_layout()
#%%
# ---------------SAVING CONVERGENCE FIGURE
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    
fig2save = PATH_OUTPUT / f"benders_convergence_plot_case_{GRID_CASE}_stochastic_{IS_STOCHASTIC}_samples_{N_SAMPLES}_multiCut_{not BENDERS_SINGLE_CUT}_fixed_cap_{MH_USE_FIXED_CAP}_{timestamp}.pdf"
fig.savefig(fig2save, format='pdf')

now_time = now = datetime.now()
formatted_now_time = now.strftime("%Y-%B-%A %H:%M:%S")
#%% 
# ---------------SAVING RESULTS REPORT
report_to_save = {"run_datetime":formatted_now_time,
                "Benders_algorithm": 
                    {
                        "duration (HH:MM:SS)": str(elapsed_time_delta),
                        "LP solver": LP_SOLVER,
                        "convergence_gap_relative [%] (UB-LB)/UB": my_BD_setup.CONVRG_EPS,
                        "convergence_gap_relative_to_OBJ_val [%]": round((my_BD_setup.CONVRG_EPS/100*UB_Bd[-1])/bd_obj_val*100,3),
                        "termination_iteration": f"{iter_Bd}/{my_BD_setup.MAX_BD_ITER}",
                        },           
                "configuration": 
                    {
                        "grid_case": GRID_CASE,
                        "branches_file_used": BRANCHES_FILE_NAME,
                        "is_stochastic": IS_STOCHASTIC,
                        "scneario_probabilities": PROB,
                        "number_of_samples": N_SAMPLES,
                        "using_single_cut": BENDERS_SINGLE_CUT,
                        "using_binary_expansion": MH_USE_BIN_EXPANS,
                        "using_fixed_capacity_lines": MH_USE_FIXED_CAP,
                        },
                "solution_and results":
                    {
                        "pgim_method": 
                        {
                            "objective_function_value_(CAPEX+OPEX)": pgim_obj_val,
                            "OPEX": pgim_opex,
                            "first_stage_decision": {
                                "new_lines": {
                                                'index_names': x_pgim['new_lines'].index.names,
                                                'index_tuples': [idx for idx in x_pgim['new_lines'].index],
                                                'data': x_pgim['new_lines'].to_dict()  
                                                },
                                "new_capacity": {
                                                'index_names': x_pgim['new_capacity'].index.names,
                                                'index_tuples': [idx for idx in x_pgim['new_capacity'].index],
                                                'data': x_pgim['new_capacity'].to_dict()  
                                                }
                                                    }
                        },

                        "multi-horizon_method": 
                        {
                            "objective_function_value_(CAPEX+OPEX)": mh_obj_val,
                            "first_stage_decision": {
                                "new_lines": {
                                                'index_names': x_mh['new_lines'].index.names,
                                                'index_tuples': [idx for idx in x_mh['new_lines'].index],
                                                'data': x_mh['new_lines'].to_dict()  
                                                },
                                "new_capacity": {
                                                'index_names': x_mh['new_capacity'].index.names,
                                                'index_tuples': [idx for idx in x_mh['new_capacity'].index],
                                                'data': x_mh['new_capacity'].to_dict()  
                                                },
                                                    }
                        },
                        "benders_decomposition_method": 
                        {
                            "objective_function_value_(CAPEX+OPEX)": bd_obj_val,
                            "OPEX": bd_opex,
                            "first_stage_decision": {
                                "new_lines": {
                                                'index_names': x_bd['new_lines'].index.names,
                                                'index_tuples': [idx for idx in x_bd['new_lines'].index],
                                                'data': x_bd['new_lines'].to_dict()  
                                                },
                                "new_capacity": {
                                                'index_names': x_bd['new_capacity'].index.names,
                                                'index_tuples': [idx for idx in x_bd['new_capacity'].index],
                                                'data': x_bd['new_capacity'].to_dict()  
                                                }
                                                    }
                        }
                    }
                }

# Write JSON data to a file
os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True)
report_file_out_path = PATH_OUTPUT/f"run_Benders_validation_{timestamp}.json"

with open(report_file_out_path, 'w') as file:
    json.dump(report_to_save, file, indent=4)

print("\n---------------Results file has been written successfully.")
#%% 
# ---------------SAVING RESULTS OBJECTS
results_file_out_path = PATH_OUTPUT/f"run_Benders_validation_{timestamp}.pickle" # write a PICKLE file for results.

print(f"{dtime.datetime.now()}: Dumping results to PICKLE file...")
rslt_to_save = {"pgim_rslt": 
                {
                    "obj_val": pgim_obj_val,
                    "opex": pgim_opex,
                    "x_new_lines": x_pgim['new_lines'].to_dict(),
                    "x_new_capacity": x_pgim['new_capacity'].to_dict(),
                    'optimal_solution':optimal_solution_pgim                         
                },

                "mh_rslt": 
                {
                    "obj_val": mh_obj_val,
                    "x_new_lines": x_mh['new_lines'].to_dict(),
                    "x_new_capacity": x_mh['new_capacity'].to_dict(),
                    'optimal_solution': optimal_solution_multi_hor                
                },

                "bd_rslt": 
                {
                    "obj_val": bd_obj_val,
                    "opex": bd_opex,
                    "upper_bound": UB_Bd,
                    "lower_bound": LB_Bd,
                    "iter-to_plot": iter_Bd_2_plot,
                    'master_problem_solution': x_current_Bd,
                    "x_new_lines": x_bd['new_lines'].to_dict(),
                    "x_new_capacity": x_bd['new_capacity'].to_dict(),
                }
            }

with open(results_file_out_path, "wb") as file:
    cloudpickle.dump(rslt_to_save, file)

print("\n---------------Results have been pickled successfully.")
#%%
# # ---------------CALCUATE VSS AND EVPI (from given results file)
if DO_CALC_VSS_EVPI or DO_VIZ:
    
    rslt_file_name = "run_Benders_validation_" + timestamp + ".pickle"
    report_file_name = "run_Benders_validation_" + timestamp + ".json"

    rslt_file_path = PATH_OUTPUT/rslt_file_name
    report_file_path = PATH_OUTPUT/report_file_name

    # Open the PICKLE file
    try:
        with open(rslt_file_path, 'rb') as file:
            run_rslt = cloudpickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("The PICKLE file you are trying to read does not exist!")

    # Open the JSON file
    try:
        with open(report_file_path, 'r') as file:
            run_settings = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("The JSON file you are trying to read does not exist!")
    
    
    if DO_CALC_VSS_EVPI:
        RP_OBJ_VAL = run_rslt['mh_rslt']['obj_val']
        print(f'\n RP = {round(RP_OBJ_VAL)}')

        current_vss, current_eev = powergim.dcmp_tools.calc_vss(run_settings,RP_OBJ_VAL,PATH_INPUT,PATH_OUTPUT)

        print(f'\n EEV = {round(current_eev)}')
        print(f'\n VSS = {round(current_vss)}')

        current_evpi, current_ws = powergim.dcmp_tools.calc_evpi(run_settings,RP_OBJ_VAL,PATH_INPUT,PATH_OUTPUT)

        print(f'\n WS = {round(current_ws)}')
        print(f'\n EVPI = {round(current_evpi)}')

        report_to_save['solution_and results']['multi-horizon_method']['stochastic_metrics']={  'RP':RP_OBJ_VAL,
                                                                                                'EEV':current_eev,
                                                                                                'VSS':current_vss, 
                                                                                                'WS': current_ws,
                                                                                                'EVPI': current_evpi
                                                                                                }
        with open(report_file_out_path, 'w') as file:
            json.dump(report_to_save, file, indent=4)

        print("\n---------------Results file has been modified successfully.")
    

    if DO_VIZ:
        if GRID_CASE == 'baseline':
            powergim.dcmp_tools.create_interactive_maps_results(run_settings,run_rslt,PATH_INPUT,PATH_OUTPUT)
            powergim.dcmp_tools.create_geopandas_figures_results(timestamp,run_settings,run_rslt,PATH_INPUT,PATH_OUTPUT)

        print("\n---------------Graphics have been geenrated successfully.")

#%% -----------------ASSERTIONS
# CHECKME: Uncomment for automatic validation of mh/bd methods against pgim. 
assert math.isclose(pgim_obj_val,mh_obj_val, rel_tol=0.01), "PGIM and MH objective values do not match"
assert math.isclose(pgim_obj_val,bd_obj_val, rel_tol=0.01), "PGIM and BD objective values do not match"  
assert math.isclose(pgim_opex,bd_opex, rel_tol=0.01), "PGIM and BD operational costs do not match" 

print('\n------------VALIDATION: checked')