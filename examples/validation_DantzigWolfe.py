# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------------
    IMPORTANT: THIS IS VALID ONLY FOR THE 'DETERMINISTIC' CASE
----------------------------------------------------------------------------------

Example script on using the `powergim.dcmp_tools` subpackage.
---
This creates a case and solves it with all 3 methods: 
- pgim (default), 
- mh (multi-horizon reformulation9
- dw (using the developed Dantzig-Wolfe decompostion).

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
import datetime as dtime
from datetime import datetime
import time
import json
import os
import math
import cloudpickle

PATH_INPUT  = Path(__file__).parents[0] / "inputs"/"CASE_BASELINE"
PATH_OUTPUT = Path(__file__).parents[0] / "outputs"

#%%
#-----------------------------INITIALIZING INPUTS-----------------------------
GRID_CASE          = 'star' # CHECKME: for demonstration choose `star`. Option `baseline` takes too much time to converge and is highly sensitive to intial conditions (columns).
BRANCHES_FILE_NAME = "branches_reduced.csv"
IS_STOCHASTIC      = False
N_SAMPLES          = 2
MH_USE_BIN_EXPANS  = False
MH_USE_FIXED_CAP   = False
LP_SOLVER          = 'gurobi' # 'appsi_highs' / 'gurobi'
DO_VIZ             = False 

if GRID_CASE == 'baseline':
    PROB = {"scen0": 0.1, "scen1": 0.45, "scen2": 0.45}
elif GRID_CASE == 'star':
    PROB = {"scen0": 0.8, "scen1": 0.1, "scen2": 0.1}

if IS_STOCHASTIC:
    raise Exception("Sorry, current DW-CG implmentation supports only 'deterministic' pgim problem --> Set `IS_STOCHASTIC=False`.")
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
mh_ref = powergim.dcmp_tools.MultiHorizonPgim(pgim_ref, is_stochastic=IS_STOCHASTIC, ancestors_include_current=True)

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
solver.options['TimeLimit'] = 60*10 # seconds
solver.options['MIPGap'] = 0.001
 
results = solver.solve(mh_ref.non_decomposed_model,tee=True,keepfiles=False,symbolic_solver_labels=True)
 
if str(results.solver.termination_condition) != "optimal": print(results.solver.termination_condition)
     
optimal_solution_multi_hor = powergim.dcmp_tools.extract_all_opt_variable_values(mh_ref.non_decomposed_model)
#%%
#-----------------------------SOLVING PGIM MODEL (FOR VALIDATION)-----------------------------
if not mh_ref.is_stochastic:
    
    results = solver.solve(pgim_ref.ref_pgim_model,tee=True,keepfiles=False,symbolic_solver_labels=True)
    
    if str(results.solver.termination_condition) != "optimal": print(results.solver.termination_condition)
    
    optimal_solution_pgim = pgim_ref.ref_pgim_model.extract_all_variable_values()

elif mh_ref.is_stochastic:
    # Define 3 scenarios (base case and 2 more)  --> 7 strategic nodes
    """
    # NOTE: The Dantzig-Wolfe method was designed for a multi-horizon structure where all operational nodes are common in the first strategic node (first period).
    # This is not inline with pgim structure.
    # Therefore, the method is not valid for the 'STOCHASTIC' case, but can be used for the 'DETERMINISTIC' one.
    """

    print("\n --------Creating scenarios for DEF")
    # -----------------------Create scenario powerGIM models-----------------------
    # This function edits the inital grid_data and parameters depending on the scenario
    # Then it creates a model instance for the scenario with the modified parameters
    if mh_ref.grid_case == 'star':

        def my_mpisppy_scenario_creator(scenario_name):
            """Create a scenario."""
            print("\n Scenario {}".format(scenario_name))

            parameter_data = copy.deepcopy(mh_ref.pgim_case.ref_params)
            grid_data = copy.deepcopy(mh_ref.pgim_case.ref_grid_data)

            match scenario_name:
                case "scen0":
                    pass

                # --------MAIN SCENARIOS

                # case "scen1": # The "Low wind", "same demand" scenario

                #     # Half the wind at n1 (wind farm node)
                #     init_wind_capacity = grid_data.generator.loc[grid_data.generator['node'] == 'n1']
                    
                #     for iperiod in parameter_data['parameters']['investment_years']:
                #         grid_data.generator.loc[grid_data.generator['node'] == 'n1', ['capacity_' + str(iperiod)]] = 0.5*init_wind_capacity.loc[0,'capacity_' + str(iperiod)]
                

                # case "scen2": # The "same wind", "high demand" scenario

                #     # Double the load at n3 (offshore load node)
                #     init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n3']
                    
                #     for iperiod in parameter_data['parameters']['investment_years']:
                #         grid_data.consumer.loc[grid_data.consumer['node'] == 'n3', ['demand_' + str(iperiod)]] = 2*init_load_capacity.loc[1,'demand_' + str(iperiod)]


                # --------ALTERNATIVE SCENARIOS

                case "scen1": # The "Lower demand at country node" scenario

                    # Half the load at n2 (country node)
                    init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n2']
                    
                    for iperiod in parameter_data['parameters']['investment_years']:
                        grid_data.consumer.loc[grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 0.5*init_load_capacity.loc[0,'demand_' + str(iperiod)]


                case "scen2": # The "Higher demand at country node" scenario

                    # Double the load at n2 (country node)
                    init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n2']
                    
                    for iperiod in parameter_data['parameters']['investment_years']:
                        grid_data.consumer.loc[grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 2*init_load_capacity.loc[0,'demand_' + str(iperiod)]

                case _:
                    raise ValueError("Invalid scenario name")

            # Create stochastic model:
        
            # A) Initialize a pgim object instane (pgim_model)
            pgim_model = pgim.SipModel(grid_data, parameter_data)
            
            # B) Use scenario_creator method to build a scenario instance model
            pgim_scen_model = pgim_model.scenario_creator(scenario_name, probability=PROB[scenario_name])
            
            return pgim_scen_model
    else:
        # raise Exception('The scenario creator has not been defind yet for the baseline grid case.')
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
                        grid_data.consumer.loc[m_demand, f"demand_{year}"] = demand_scale * grid_data.consumer.loc[m_demand, f"demand_{year}"]
                        # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 0.5 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]
                        # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 2 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]


                case "scen2": # oceangrid_A2_moredemand
                    # More NO demand (340 TWh in 2050)

                    demand_scale = 340/260
                    # demand_scale = 520/260
                    m_demand = grid_data.consumer["node"].str.startswith("NO_")
                    for year in parameter_data['parameters']['investment_years']:
                        grid_data.consumer.loc[m_demand, f"demand_{year}"] = demand_scale * grid_data.consumer.loc[m_demand, f"demand_{year}"]
                        # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 2 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]
                        # grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 0.5 * grid_data.consumer.loc[grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]


                case _:
                    raise ValueError("Invalid scenario name")

            # Create stochastic model:
            pgim_model = pgim.SipModel(grid_data, parameter_data) # A) Initialize a pgim object instane (pgim_model)
            pgim_scen_model = pgim_model.scenario_creator(scenario_name, probability=PROB[scenario_name]) # B) Use scenario_creator method to build a scenario instance model
            
            return pgim_scen_model

    # This formulates the extensive form based on the scenarios that are defined (an mpisppy object)
    # Preferred method based on mpispppy: mpisppy.opt.ef.ExtensiveForm --> needs mpi4py
    my_pgim_stochastic_model_ef = mpisppy.utils.sputils.create_EF(mh_ref.scenario_names,scenario_creator=my_mpisppy_scenario_creator)

    # Solve the EF
    solver = pyo.SolverFactory("gurobi")
    solver.options['TimeLimit'] = 60*10 # seconds

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
#%%
#-----------------------------RUNNING DANTZIG-WOLFE DECOMPOSITION ALGORITHM-----------------------------
match GRID_CASE:
        case 'star':
            match IS_STOCHASTIC:
                case False:
                    my_DW_setup = powergim.dcmp_tools.DantzigWolfeDecomp(mh_ref,CONVRG_EPS=1,MAX_DW_ITER=2000,INIT_UB=35000)
                case True:
                    my_DW_setup = powergim.dcmp_tools.DantzigWolfeDecomp(mh_ref,CONVRG_EPS=1,MAX_DW_ITER=2000,INIT_UB=250000)
        case 'baseline':
            match IS_STOCHASTIC:
                case False:
                    my_DW_setup = powergim.dcmp_tools.DantzigWolfeDecomp(mh_ref,CONVRG_EPS=1,MAX_DW_ITER=100000,INIT_UB=730000)
                case True:
                    my_DW_setup = powergim.dcmp_tools.DantzigWolfeDecomp(mh_ref,CONVRG_EPS=1,MAX_DW_ITER=100000,INIT_UB=760000)


# Initialize placeholders
UB_DW, LB_DW, iter_DW_2_plot = [], [], [] # Keep the Upper Bounds of all iterations.

y_hat_init              = [0] * mh_ref.s_strategic_nodes
all_sb_objective_values = [0] * mh_ref.s_strategic_nodes 
sb_n_sol                = [0] * mh_ref.s_strategic_nodes

sb_n_j_sol = [[0 for i in range(mh_ref.s_strategic_nodes)] for j in range(my_DW_setup.MAX_DW_ITER)]

convergence_check = False

# ----------Create an initial plot
fig = plt.figure(figsize=(8,5))
line_UB, = plt.gca().plot([], [], linestyle='-', color='blue', label='Upper Bound')
line_LB, = plt.gca().plot([], [], linestyle='-', color='red', label='Lower Bound')

plt.gca().legend()

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

initial_ylim = (0, my_DW_setup.INIT_UB) # Set initial y-axis limits
plt.gca().set_ylim(initial_ylim)

initial_xlim = (0, 1)  # Set initial x-axis limits - Adjust as needed
plt.gca().set_xlim(initial_xlim)

# Add labels and title
plt.xlabel('Iterations')
plt.ylabel('Objective function value')
plt.title('Dantzig-Wolfe convergence plot')
plt.legend()

# Show the plot
plt.grid(True)
plt.show(block=False)  


#***********************************************************************************
#***************************** MAIN DANTZIG-WOLFE LOOP *****************************
#***********************************************************************************
start_time_algorithm = time.time()

# Create the initial RRMP
my_DW_setup.create_initial_RRMP()
# Just to update the model
RRMP_solver = pyo.SolverFactory(LP_SOLVER)

RRMP_solver_rslt = RRMP_solver.solve(my_DW_setup.RRMP_model)

for iter_DW in range(0, my_DW_setup.MAX_DW_ITER):
    
    if iter_DW == 0:
        # Assign sub-problem parameters based on a first-guess FEP (trivial flat zero solution)
        for n in range(mh_ref.s_strategic_nodes): 

            sb_n_model = my_DW_setup.assign_RRMP_duals_to_sub_problems(n, iter_DW, first_FEP=True)           
            sb_n_solver = pyo.SolverFactory(LP_SOLVER)           
            sb_n_solver_rslt = sb_n_solver.solve(sb_n_model)

            if sb_n_solver_rslt.solver.termination_condition == TerminationCondition.optimal:
                print(f"Sub-problem {n} solved at iteration: {iter_DW} (initial FEP).")

                # Get current solution and objective of Sub-problem.
                y_hat_init[n] = powergim.dcmp_tools.extract_all_opt_variable_values(sb_n_model)
                sb_n_j_sol[iter_DW][n] = y_hat_init[n]
                all_sb_objective_values[n] = pyo.value(sb_n_model.obj)
            else:
                print(f"\n\n-----------------Initial Sub-problem failed at iteration: {iter_DW}.")
                break
        
        # Add a column (update the columns set)
        my_DW_setup.RRMP_model.J_columns.add(iter_DW)

        my_DW_setup.assign_initial_y_hat_to_RRMP(iter_DW, y_hat_init)

        my_DW_setup.update_RRMP(iter_DW)
        RRMP_solver = pyo.SolverFactory(LP_SOLVER)
        RRMP_solver_rslt = RRMP_solver.solve(my_DW_setup.RRMP_model)
        
        if RRMP_solver_rslt.solver.termination_condition == TerminationCondition.optimal:
            print(f"RRMP solved at iteration: {iter_DW} (initial FEP).")
        else:
            print(f"Initial RRMP failed at iteration: {iter_DW}.")
            break

    else:
        # Assign SP parameters based on a previously found FEP
        for n in my_DW_setup.RRMP_model.n_strgc:
            
            sb_n_model = my_DW_setup.assign_RRMP_duals_to_sub_problems(n, iter_DW-1, first_FEP=False)           
            sb_n_solver = pyo.SolverFactory(LP_SOLVER)
                        
            sb_n_solver_rslt = sb_n_solver.solve(sb_n_model)
            
            print(f"Sub-problem {n} solved at iteration: {iter_DW}.")
            # Get current solution and objective of Sub-problem.
            sb_n_sol[n]                = powergim.dcmp_tools.extract_all_opt_variable_values(sb_n_model)
            sb_n_j_sol[iter_DW][n]     = sb_n_sol[n]
            all_sb_objective_values[n] = pyo.value(sb_n_model.obj)
            
        # Check reduced costs of subproblems (convergence)
        if iter_DW > 2:
            if min(all_sb_objective_values)>=0:
                print(f"No negative reduced cost found at iteration: {iter_DW}.")
                convergence_check = True
            
        # Add a column (update the columns set)
        my_DW_setup.RRMP_model.J_columns.add(iter_DW)
        my_DW_setup.assign_columns_data_to_RMP(iter_DW, sb_n_sol)
        my_DW_setup.update_RRMP(iter_DW)

        # Solve the problem
        RRMP_solver = pyo.SolverFactory(LP_SOLVER)
        RRMP_solver.options['Method'] = 2
        RRMP_solver.options['Crossover'] = -1
        RRMP_solver_rslt = RRMP_solver.solve(my_DW_setup.RRMP_model)

        # my_DW_setup.RRMP_model.pprint()

        if RRMP_solver_rslt.solver.termination_condition == TerminationCondition.optimal:
            print(f"RRMP solved at iteration: {iter_DW}.")
        else:
            print(f"RRMP failed at iteration: {iter_DW}.")
            break
                
    # CONVERGENCE CHECK TO BREAK, OTHERWISE CONTINUE
    UB = pyo.value(my_DW_setup.RRMP_model.obj)
    LB = pyo.value(my_DW_setup.RRMP_model.obj) + sum(all_sb_objective_values)
    
    UB_DW.append(UB)
    LB_DW.append(LB)
    iter_DW_2_plot.append(iter_DW+1)
    
    # update_plot
    line_UB.set_data(iter_DW_2_plot, UB_DW)
    line_LB.set_data(iter_DW_2_plot, LB_DW)
        
    # Dynamically adjust y-axis limits
    # new_ylim = (0, min(UB_DW)+1000)
    if IS_STOCHASTIC:
        if GRID_CASE == 'star':
            new_ylim = (220000, 300000)  # Toy - stochastic
        else:
            new_ylim = (250000, 600000)
    else:
        new_ylim = (0, UB +1000) 

    plt.gca().set_ylim(new_ylim)
        
    # Dynamically adjust x-axis limits
    new_xlim = (0, max(iter_DW_2_plot)+1)
    plt.gca().set_xlim(new_xlim)

    plt.draw()
    plt.pause(0.1)  # Adjust the pause time as needed
        
    print(f"Column generation relative gap at iteration {iter_DW} is {round(abs((UB-LB)/(UB+1)*100),2)} %.")
        
    if abs((UB-LB)/(UB+1)*100) <= my_DW_setup.CONVRG_EPS:
        print("-------------RRMP column generation converged!!!\n")
        convergence_check = True
    
    # If final step is reached or the RRMP has converged, solve the IRPM with the most updated column information
    if iter_DW >= 1:
        if (iter_DW == my_DW_setup.MAX_DW_ITER - 1) or (convergence_check == True):

            # Define IRMP
            my_DW_setup.create_IRMP(columns_indexes = list(range(iter_DW+1)))

            my_DW_setup.assign_columns_data_to_RMP_OLD_implementation(iter_DW, sb_n_j_sol, sb_n_sol, FINAL_ITER_FLAG=True)

            # Solve the problem
            IRMP_solver = pyo.SolverFactory(LP_SOLVER)
            # Set Gurobi solver parameters
            # IRMP_solver.options['TimeLimit'] = 60*1 # seconds
            # IRMP_solver.options['MIPGap'] = 0.005
                        
            IRMP_solver_rslt = IRMP_solver.solve(my_DW_setup.IRMP_model)

            mip_gap = abs((pyo.value(my_DW_setup.IRMP_model.obj)-LB)/(UB+1)*100)
            print(f"-------------IRMP column generation  gap: {round(mip_gap,2)} % \n")
            break
       
else:
   print(f"Max iterations ({iter_DW}) exceeded.\n")

end_time_algorithm = time.time()

elapsed_time_algorithm = end_time_algorithm-start_time_algorithm
# Convert elapsed time to a timedelta object
elapsed_time_delta = dtime.timedelta(seconds=elapsed_time_algorithm)

print(f"\n\n----------------------DW-CG algorithm took ---> {str(elapsed_time_delta)} (HH:MM:SS).")

#---------------------------END OF DW-CG ALGORTIHM-----------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#%% ---------------------------------VALIDATION
solution_RRMP = powergim.dcmp_tools.extract_all_opt_variable_values(my_DW_setup.RRMP_model)
solution_IRMP = powergim.dcmp_tools.extract_all_opt_variable_values(my_DW_setup.IRMP_model)

IC_per_node_RRMP, OC_per_node_RRMP = my_DW_setup.calculate_costs(my_DW_setup.RRMP_model,solution_RRMP)
IC_per_node_IRMP, OC_per_node_IRMP = my_DW_setup.calculate_costs(my_DW_setup.IRMP_model,solution_IRMP)

mh_obj_val   = round(pyo.value(mh_ref.non_decomposed_model.objective))
dw_RRMP_obj_val = round(pyo.value(my_DW_setup.RRMP_model.obj))
dw_IRMP_obj_val = round(pyo.value(my_DW_setup.IRMP_model.obj))

dw_RRMP_capex = round(sum(IC_per_node_RRMP))
dw_IRMP_capex = round(sum(IC_per_node_IRMP))

dw_RRMP_opex = round(sum(OC_per_node_RRMP))
dw_IRMP_opex = round(sum(OC_per_node_IRMP))


if not mh_ref.is_stochastic:
    # OBJECTIVE VALUE
    pgim_obj_val = round(pyo.value(pgim_ref.ref_pgim_model.OBJ))
   
    print(f'PGIM OBJECTIVE VALUE: \t\t {pgim_obj_val}')
    print(f'MULTI-HORIZON OBJECTIVE VALUE: \t {mh_obj_val}')
    print(f'DW-RRMP OBJECTIVE VALUE: \t {dw_RRMP_obj_val}')
    print(f'DW-IRMP OBJECTIVE VALUE: \t {dw_IRMP_obj_val}')
    print('------------------------------------------------------')

    # CAPEX
    pgim_capex = round(optimal_solution_pgim["v_investment_cost"].sum())
   
    print(f'PGIM CAPEX: \t {pgim_capex}')
    print(f'DW-RRMP CAPEX: \t {dw_RRMP_capex}')
    print(f'DW-IRMP CAPEX: \t {dw_IRMP_capex}')
    print('------------------------------------------------------')

    # OPEX
    pgim_opex = round(optimal_solution_pgim["v_operating_cost"].sum())
   

    print(f'PGIM OPEX: \t {pgim_opex}')
    print(f'DW-RRMP OPEX: \t {dw_RRMP_opex}')
    print(f'DW-IRMP OPEX: \t {dw_IRMP_opex}')

    def get_first_stage_decision_pgim(opt_sol_pgim):
        x_cbl = opt_sol_pgim['v_branch_new_cables'].xs(2035, level=1).loc[opt_sol_pgim['v_branch_new_cables'].xs(2035, level=1).values>0.01]
        x_cpt = opt_sol_pgim['v_branch_new_capacity'].xs(2035, level=1).loc[opt_sol_pgim['v_branch_new_capacity'].xs(2035, level=1).values>0.01]
        x = {'new_lines':x_cbl, 'new_capacity':x_cpt}
        return x

    
elif mh_ref.is_stochastic:   
    # OBJECTIVE VALUE
    pgim_obj_val = round(pyo.value(my_pgim_stochastic_model_ef.EF_Obj))
    
    print(f'PGIM DEF OBJECTIVE VALUE: \t {pgim_obj_val}')
    print(f'MULTI-HORIZON OBJECTIVE VALUE: \t {mh_obj_val}')
    print(f'DW-RRMP OBJECTIVE VALUE: \t {dw_RRMP_obj_val}')
    print(f'DW-IRMP OBJECTIVE VALUE: \t {dw_IRMP_obj_val}')
    print('------------------------------------------------------')

    # EXPECTED OPEX
    # print(f'DEF EXPECTED OPEX: \t {round(all_var_all_scen_values[0]["scen0.v_operating_cost"].sum()*0.5 + 0.25*(all_var_all_scen_values[1]["scen1.v_operating_cost"].sum()+all_var_all_scen_values[2]["scen2.v_operating_cost"].sum()))}')
    pgim_opex = round(  PROB['scen0']*all_var_all_scen_values[0]["scen0.v_operating_cost"].sum() + \
                        PROB['scen1']*all_var_all_scen_values[1]["scen1.v_operating_cost"].sum()+ \
                        PROB['scen2']*all_var_all_scen_values[2]["scen2.v_operating_cost"].sum()
                        )
    print(f'DEF EXPECTED OPEX: {pgim_opex}')
    print(f'DW-RRMP EXPECTED OPEX: \t {dw_RRMP_opex}')
    print(f'DW-IRMP EXPECTED OPEX: \t {dw_IRMP_opex}')

    def get_first_stage_decision_pgim(opt_sol_pgim):
        scen = 0
        x_cbl = opt_sol_pgim[scen][f'scen{scen}.v_branch_new_cables'].xs(2035, level=1).loc[opt_sol_pgim[scen][f'scen{scen}.v_branch_new_cables'].xs(2035, level=1).values>0.01]
        x_cpt = opt_sol_pgim[scen][f'scen{scen}.v_branch_new_capacity'].xs(2035, level=1).loc[opt_sol_pgim[scen][f'scen{scen}.v_branch_new_capacity'].xs(2035, level=1).values>0.01]
        x = {'new_lines':x_cbl, 'new_capacity':x_cpt}
        return x

print('------------------------------------------------------')
print(f'DW-RRMP Investment costs: {round(sum(IC_per_node_RRMP))}')
print(f'DW-IRMP Investment costs: {round(sum(IC_per_node_IRMP))}')
print(f'DW-IRMP Operational costs: {round(sum(OC_per_node_IRMP))}')
print('\n------------------------------------------------------\n')
print(f'DW-IRMP Total costs: {round(sum(IC_per_node_IRMP) + sum(OC_per_node_IRMP))}')


x_pgim = get_first_stage_decision_pgim(optimal_solution_pgim)
x_mh   = mh_ref.get_first_stage_decision(optimal_solution_multi_hor)
x_dw   = mh_ref.get_first_stage_decision(solution_IRMP)

plt.tight_layout()
#%%
# ---------------SAVING FIGURE AND RESULTS
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    
fig2save = PATH_OUTPUT / f"dw_convergence_plot_case_{GRID_CASE}_stochastic_{IS_STOCHASTIC}_samples_{N_SAMPLES}_fixed_cap_{MH_USE_FIXED_CAP}_{timestamp}.pdf"
fig.savefig(fig2save, format='pdf')

now_time = now = datetime.now()
formatted_now_time = now.strftime("%Y-%B-%A %H:%M:%S")
#%%
data_to_save = {"run_time":formatted_now_time,
                "DW_algorithm": 
                    {
                        "duration (HH:MM:SS)": str(elapsed_time_delta),
                        "LP solver": LP_SOLVER,
                        "convergence_gap_relative_to_UB_val [%]": round(abs((UB-LB)/(UB+1)*100),2),
                        "termination_iteration": f"{iter_DW}/{my_DW_setup.MAX_DW_ITER}",
                        },   
                "configuration": 
                    {
                        "grid_case": GRID_CASE,
                        "branches_file_used": BRANCHES_FILE_NAME,
                        "is_stochastic": IS_STOCHASTIC,
                        "number_of_samples": N_SAMPLES,
                        "DW_implementation": 'NEW',
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
                        "DW_decomposition_method": 
                        {
                            "objective_function_value_RRMP(CAPEX+OPEX)": dw_RRMP_obj_val,
                            "objective_function_value_IRMP(CAPEX+OPEX)": dw_IRMP_obj_val,
                            "RRMP_CAPEX": dw_RRMP_capex,
                            "IRMP_CAPEX": dw_IRMP_capex,
                            "RRMP_OPEX": dw_RRMP_opex,
                            "IRMP_OPEX": dw_IRMP_opex,
                            "first_stage_decision": {
                                "new_lines": {
                                                'index_names': x_dw['new_lines'].index.names,
                                                'index_tuples': [idx for idx in x_dw['new_lines'].index],
                                                'data': x_dw['new_lines'].to_dict()  
                                                },
                                "new_capacity": {
                                                'index_names': x_dw['new_capacity'].index.names,
                                                'index_tuples': [idx for idx in x_dw['new_capacity'].index],
                                                'data': x_dw['new_capacity'].to_dict()  
                                                }
                                                    }
                        }
                    }
                }
# Write JSON data to a file
os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True)
file_out_path = PATH_OUTPUT/f"run_DW_validation_{timestamp}.json"

with open(file_out_path, 'w') as file:
    json.dump(data_to_save, file, indent=4)

print("\n---------------Results file has been written successfully.")
#%%
# Write a PICKLE file for results
results_file_out_path = PATH_OUTPUT/f"run_DW_validation_{timestamp}.pickle"


print(f"{dtime.datetime.now()}: Dumping results to PICKLE file...")
dump_dict = {"pgim_rslt": 
                {
                    "obj_val": pgim_obj_val,
                    "opex": pgim_opex,
                    "x_new_lines": x_pgim['new_lines'].to_dict(),
                    "x_new_capacity": x_pgim['new_capacity'].to_dict()
                                            
                },

                "mh_rslt": 
                {
                    "obj_val": mh_obj_val,
                    "x_new_lines": x_mh['new_lines'].to_dict(),
                    "x_new_capacity": x_mh['new_capacity'].to_dict()                
                },

                "dw_rslt": 
                {
                    "obj_val": dw_IRMP_obj_val,
                    "opex": dw_IRMP_opex,
                    "IRMP_solution": solution_IRMP,
                    "x_new_lines": x_dw['new_lines'].to_dict(),
                    "x_new_capacity": x_dw['new_capacity'].to_dict()
                }
            }

with open(results_file_out_path, "wb") as file:
    cloudpickle.dump(dump_dict, file)


print("\n---------------Results have been pickled successfully.")
#%%
# -----------------ASSERTIONS
assert abs((round(sum(IC_per_node_RRMP) + sum(OC_per_node_RRMP))) - (round(pyo.value(my_DW_setup.RRMP_model.obj)))) <= 10, 'Discrepancy in objective value and costs in RRMP'
assert abs((round(sum(IC_per_node_IRMP) + sum(OC_per_node_IRMP))) - (round(pyo.value(my_DW_setup.IRMP_model.obj)))) <= 10, 'Discrepancy in objective value and costs in IRMP'

assert math.isclose(pgim_obj_val,mh_obj_val, rel_tol=0.02), "PGIM and MH objective values do not match"
assert math.isclose(pgim_obj_val,dw_IRMP_obj_val, rel_tol=0.02), "PGIM and DW objective values do not match"  
assert math.isclose(pgim_opex,dw_IRMP_opex, rel_tol=0.02), "PGIM and DW operational costs do not match" 

print('\n------------VALIDATION: checked')