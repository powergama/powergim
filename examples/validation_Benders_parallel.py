"""
Example script on using the parallel Benders capability of the `powergim.dcmp_tools` subpackage.
---

Open a PowerShell, cd current/script/dir and run: 

    `mpiexec -n num_processors python validation_Benders_parallel.py`
    
where, 'num_processors' the number of processors to use.

Example: `mpiexec -n 4 python validation_Benders_parallel.py`

@author: spyridonc
"""
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import powergim as pgim
import powergim.dcmp_tools
from mpi4py import MPI
import dill
MPI.pickle.__init__(dill.dumps, dill.loads) # NOTE: this is needed to properly seriealize SipModels which are distributed in processors through MultiHorizonPgim classes.
import mpisppy.opt.ph
import mpisppy.utils.sputils
from pathlib import Path
import copy
import datetime as dtime
from datetime import datetime
import time
import json
import os
import cloudpickle

#%%
processesToUse = MPI.COMM_WORLD.Get_size()
worker = MPI.COMM_WORLD.Get_rank()
BD_MAX_ITER = 1000
ITER_TO_STOP = BD_MAX_ITER-1

CONVERGENCE_FLAG = False

#-----------------------------PARALLEL FUNCTION DEFINITION-----------------------------
def solve_current_worker_subproblems(incoming_subproblems_list):
    # This will contain the objectives of all subproblems for current BD iteration and worker.
    current_BD_iter_and_worker_data = []
    worker_sb_i_solver = pyo.SolverFactory('appsi_highs') # NOTE LP_SOLVER: 'appsi_highs' (not 'gurobi') for parallel implementation.
    
    for sb_i in incoming_subproblems_list:
        worker_sb_i_model = sb_i[1]  # select the pyomo optimization subproblem
        worker_sb_i_duals = []       # initialize placeholder for subproblem duals

        worker_sb_i_solver_rslt = worker_sb_i_solver.solve(worker_sb_i_model,tee=False)

        if worker_sb_i_solver_rslt.solver.termination_condition == TerminationCondition.optimal:

            worker_sb_i_obj_val = pyo.value(worker_sb_i_model.obj)

            for br in worker_sb_i_model.branch:
                worker_sb_i_duals.append(worker_sb_i_model.dual[worker_sb_i_model.c_rule_fix_master_var_z_capacity_total[br]])
        else:
            raise ValueError(f"Sub-problem {sb_i[0]} at process {worker} failed.")

        current_BD_iter_and_worker_data.append([worker_sb_i_obj_val,worker_sb_i_duals])

    return current_BD_iter_and_worker_data

#%%
if worker == 0:
    #-----------------------------INITIALIZING INPUTS-----------------------------
    PATH_INPUT  = Path(__file__).parents[0] / "inputs"/"CASE_BASELINE"
    PATH_OUTPUT = Path(__file__).parents[0] / "outputs"

    GRID_CASE          = 'baseline'
    BRANCHES_FILE_NAME = "branches_reduced.csv"
    IS_STOCHASTIC      = False
    N_SAMPLES          = 2
    BENDERS_SINGLE_CUT = False
    MH_USE_BIN_EXPANS  = False
    MH_USE_FIXED_CAP   = False
    # NOTE: These probabilities were used in the paper.
    if GRID_CASE == 'baseline':
        PROB = {"scen0": 0.1, "scen1": 0.45, "scen2": 0.45} # What the probabilities should be? - this should be parameter to vary
    elif GRID_CASE == 'star':
        PROB = {"scen0": 0.8, "scen1": 0.1, "scen2": 0.1} # What the probabilities should be? - this should be parameter to vary


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
    #-----------------------------CREATING MULTI-HORIZON PROBLEM FROM PGIM DATA-----------------------------
    mh_ref.get_pgim_params_per_scenario()
    mh_ref.create_multi_horizon_problem(USE_BIN_EXPANS=MH_USE_BIN_EXPANS,USE_FIXED_CAP_LINES=MH_USE_FIXED_CAP)
    #%%
    #-----------------------------SOLVING MULTI-HORIZON PROBLEM-----------------------------
    solver = pyo.SolverFactory('gurobi')
    solver.options['TimeLimit'] = 60*15 # seconds
    
    results = solver.solve(mh_ref.non_decomposed_model,tee=True,keepfiles=False,symbolic_solver_labels=True)
    
    if str(results.solver.termination_condition) != "optimal": print(results.solver.termination_condition)
        
    optimal_solution_multi_hor = powergim.dcmp_tools.extract_all_opt_variable_values(mh_ref.non_decomposed_model)
    #%%
    #-----------------------------SOLVING PGIM MODEL (FOR VALIDATION)-----------------------------
    if not mh_ref.is_stochastic:
    
        results = solver.solve(pgim_ref.ref_pgim_model, tee=True,keepfiles=False,symbolic_solver_labels=True)
        
        if str(results.solver.termination_condition) != "optimal": print(results.solver.termination_condition)
        
        optimal_solution_pgim = pgim_ref.ref_pgim_model.extract_all_variable_values()

    elif mh_ref.is_stochastic:
        # Define 3 scenarios (base case and 2 more)  --> 7 strategic nodes

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

                    case "scen1": # The "Low wind", "same demand" scenario

                        # Half the wind at n1 (wind farm node)
                        init_wind_capacity = grid_data.generator.loc[grid_data.generator['node'] == 'n1']
                        
                        for iperiod in parameter_data['parameters']['investment_years']:
                            grid_data.generator.loc[grid_data.generator['node'] == 'n1', ['capacity_' + str(iperiod)]] = 0.5*init_wind_capacity.loc[0,'capacity_' + str(iperiod)]
                    

                    case "scen2": # The "same wind", "high demand" scenario

                        # Double the load at n3 (offshore load node)
                        init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n3']
                        
                        for iperiod in parameter_data['parameters']['investment_years']:
                            grid_data.consumer.loc[grid_data.consumer['node'] == 'n3', ['demand_' + str(iperiod)]] = 2*init_load_capacity.loc[1,'demand_' + str(iperiod)]


                    # --------ALTERNATIVE SCENARIOS

                    # case "scen1": # The "Lower demand at country node" scenario

                    #     # Half the load at n2 (country node)
                    #     init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n2']
                        
                    #     for iperiod in parameter_data['parameters']['investment_years']:
                    #         grid_data.consumer.loc[grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 0.5*init_load_capacity.loc[0,'demand_' + str(iperiod)]


                    # case "scen2": # The "Higher demand at country node" scenario

                    #     # Double the load at n2 (country node)
                    #     init_load_capacity = grid_data.consumer.loc[grid_data.consumer['node'] == 'n2']
                        
                    #     for iperiod in parameter_data['parameters']['investment_years']:
                    #         grid_data.consumer.loc[grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 2*init_load_capacity.loc[0,'demand_' + str(iperiod)]

                    case _:
                        raise ValueError("Invalid scenario name")

                # Create stochastic model:
                pgim_model = pgim.SipModel(grid_data, parameter_data) # A) Initialize a pgim object instane (pgim_model)
                pgim_scen_model = pgim_model.scenario_creator(scenario_name, probability=PROB[scenario_name]) # B) Use scenario_creator method to build a scenario instance model
                
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
        solver.options['TimeLimit'] = 60*15 # seconds

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
    match GRID_CASE:
        case 'star':
            match IS_STOCHASTIC:
                case False:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.01,MAX_BD_ITER=BD_MAX_ITER,INIT_UB=35000)
                case True:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.01,MAX_BD_ITER=BD_MAX_ITER,INIT_UB=800000)
        case 'baseline':
            match IS_STOCHASTIC:
                case False:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.1,MAX_BD_ITER=BD_MAX_ITER,INIT_UB=800000)
                case True:
                    my_BD_setup = powergim.dcmp_tools.BendersDecomp(mh_ref,CONVRG_EPS=0.1,MAX_BD_ITER=BD_MAX_ITER,INIT_UB=3000000)

    gap_Bd = float('Inf') # Initial convergence gap 

    # Placeholders initialization (plots/pickle)
    x_current_Bd, x_all_Bd, UB_Bd, LB_Bd, iter_Bd_2_plot = [], [], [], [], []  # Keep the current optimal solution of the Master and the Upper/Lower Bounds of all iterations

    all_sb_objective_values = [0] * mh_ref.s_operational_nodes * mh_ref.s_sample_size
    all_dual_values = [[0]*mh_ref.s_branches] * mh_ref.s_operational_nodes * mh_ref.s_sample_size

    UB, LB = my_BD_setup.INIT_UB, my_BD_setup.INIT_LB

    # ---------------------Create an initial plot
    fig = plt.figure(figsize=(8,5))

    line_UB, = plt.gca().plot([], [], linestyle='-', color='blue', label='Upper Bound')
    line_LB, = plt.gca().plot([], [], linestyle='-', color='red', label='Lower Bound')

    plt.gca().legend()

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    initial_ylim = (0, my_BD_setup.INIT_UB) # Set initial y-axis limits
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


    # Initialize Sub-problems
    # Create a list of subproblems (instances) per operational node and per time sample.
    sb_Bd_models = [None for n_op in range(mh_ref.s_operational_nodes) for t in range(mh_ref.s_sample_size)]
    nodes_from_sb = my_BD_setup.assign_operational_nodes_to_subproblems(list(range(1,mh_ref.s_operational_nodes+1)), mh_ref.s_sample_size)


    #***********************************************************************************
    #******************************** MAIN BENDERS LOOP ********************************
    #***********************************************************************************
    start_time_BD_algorithm = time.time()

    for iter_Bd in range(0, my_BD_setup.MAX_BD_ITER + 1):

        if (CONVERGENCE_FLAG == True) or iter_Bd == ITER_TO_STOP+1:
            # ITER_TO_STOP_WORKERS = iter_Bd
            print(f"\n\n---------Premature ending activated at iter {iter_Bd}...")
            # Broadcast termination signal to all workers
            for i_worker in range(1, processesToUse):
                MPI.COMM_WORLD.send('terminate', dest=i_worker, tag=11)
            break

    
        print("\n ---------------------------------Running Benders loop...")
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

            # Loop over the list of different subproblems and add the dual-based contraint from each of them.               
            for sb_i, sb_Bd_model_i in enumerate(sb_Bd_models,1): 
                for i_br, br in enumerate(master_Bd_model.branch):
                    # Set value of Master parameters based on Sub-problem dual values.
                    # Make sure to assign the correct dual to the corresponding master parameter indexed by subproblem (i.e., scenario for L-shaped method).
                    if BENDERS_SINGLE_CUT:
                        master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,sb_i-1,iter_Bd].set_value(all_dual_values[sb_i-1][i_br])

                    else:
                        master_Bd_model.sb_current_master_solution_dual_z_capacity_total[br,sb_i-1,current_cut_index+sb_i].set_value(all_dual_values[sb_i-1][i_br])



            # Get the current Sub-problems objective value(s).
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
                
                master_Bd_model.Cut_Defn.add(master_Bd_model.a >= cut)

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
                    # Add the cut to the Master.
                    master_Bd_model.Cut_Defn.add(master_Bd_model.a[sb_i-1] >= cut)
        
        #---------------------Solve the Master.
        master_Bd_solver.options['TimeLimit'] = 60*3 # seconds
        master_Bd_solver_rslt = master_Bd_solver.solve(master_Bd_model)

        # Store the current Master solution.        
        x_current_Bd = powergim.dcmp_tools.extract_all_opt_variable_values(master_Bd_model)
        x_all_Bd.append(x_current_Bd)
        # Update Lower Bound (approximated Sub-problem cost)
        LB = master_Bd_model.obj()
        LB_Bd.append(LB)

        # ---------------------SUB-PROBLEMS(S)
        # Create the Sub-problem(s) (only at the first iteration).
        if iter_Bd == 0:
            sb_Bd_models = [my_BD_setup.create_sub_problem(n_op,t) for n_op in range(1,mh_ref.s_operational_nodes+1) for t in range(mh_ref.s_sample_size)]
   

        # Fix the x "replica" variables in the Sub-problem with the current (updated) Master solution.   
        for sb_i, sb_model_i in enumerate(sb_Bd_models):
            for br in master_Bd_model.branch:
                sb_model_i.z_capacity_total_fixed[br].set_value(max(x_current_Bd['z_capacity_total'][br,mh_ref.dict_map_operational_to_strategic_node['n_op_' + str(nodes_from_sb[sb_i])]],0))

        
        # ---------------------Solve the Sub-problems.
        print(f"------------Solving sub-problems in parallel in Bender's iteration {iter_Bd}...")

        #===============================MASTER WORKER PREPARES PARALLEL SOLUTION OF SUB-PROBLEMS (start)=====================================

        # Distribute subproblems to workers (available processes)
        minSubproblemsPerWorker = len(sb_Bd_models) // len(range(1, processesToUse))  # Divide the tasks as evenly as possible between the processes. Some processes will do n tasks (solve n subproblems),
        maxSubproblemsPerWorker = minSubproblemsPerWorker + 1   # and some other processes will do n + 1 tasks
        plusOneWorkers = len(sb_Bd_models) - (minSubproblemsPerWorker * len(range(1, processesToUse)))
        stdWorkers = len(range(1, processesToUse)) - plusOneWorkers

        # Sending subproblems to workers
        for i_worker in range(1,processesToUse):
            if i_worker <= stdWorkers:
                sbFrom = (i_worker-1) * minSubproblemsPerWorker
                sbTo   = sbFrom + minSubproblemsPerWorker
            else:
                sbFrom = stdWorkers * minSubproblemsPerWorker + (i_worker - stdWorkers -1) * maxSubproblemsPerWorker
                sbTo = sbFrom + maxSubproblemsPerWorker

            i_workerSBs = range(sbFrom, sbTo) 
            data_to_send_per_worker = [(sb_index, sb_Bd_models[sb_index]) for sb_index in i_workerSBs]
            MPI.COMM_WORLD.send(data_to_send_per_worker, dest=i_worker, tag=11)
            # print(f"\n Sending taks {data_to_send_per_worker} to worker {i_worker}")


        # Receiving subproblems data from all workers
        for i_worker in range(1,processesToUse):
            if i_worker <= stdWorkers:
                sbFrom = (i_worker-1) * minSubproblemsPerWorker
                sbTo   = sbFrom + minSubproblemsPerWorker
            else:
                sbFrom = stdWorkers * minSubproblemsPerWorker + (i_worker - stdWorkers -1) * maxSubproblemsPerWorker
                sbTo = sbFrom + maxSubproblemsPerWorker

            i_workerSBs = range(sbFrom, sbTo) 

            data_received_from_worker = MPI.COMM_WORLD.recv(source=i_worker, tag=12)
            # print(f"\n Data received from worker {i_worker}: \n {data_received_from_worker}")

            local_worker_index = 0

            for sb_index in i_workerSBs:
                all_sb_objective_values[sb_index] = data_received_from_worker[local_worker_index][0]
                all_dual_values[sb_index] = data_received_from_worker[local_worker_index][1]
                
                local_worker_index += 1

                # print(f"\n Received obj. value {all_sb_objective_values[sb_index]} for sub-problem {sb_index} and from worker {i_worker}.")
                # print(f"\n Received dual value {all_dual_values[sb_index]} for sub-problem {sb_index} and from worker {i_worker}.")
        
            # Terminating workers employed to solve subproblems.
            MPI.COMM_WORLD.send(None, dest=i_worker, tag=11)
            print(f"Terminating worker {i_worker}.")

        print(f"\n -------All sub-problems workers executed and terminated at iteration {iter_Bd}.")
        # print(f"\n Complete rslts list: {all_sb_objective_values}.\n")

        #===============================MASTER WORKER PREPARES PARALLEL SOLUTION OF SUB-PROBLEMS (END)=====================================



                
        # Update the Upper Bound.
        if BENDERS_SINGLE_CUT:
            UB = min(UB, master_Bd_model.obj() - master_Bd_model.a.value + sum(all_sb_objective_values))
        else:
            temp_a_sum = 0
            for i in master_Bd_model.a: temp_a_sum += master_Bd_model.a[i].value
            UB = min(UB, master_Bd_model.obj() - temp_a_sum + sum(all_sb_objective_values))
        
        UB_Bd.append(UB)
        
        # Update the plot
        iter_Bd_2_plot.append(iter_Bd)

        line_UB.set_data(iter_Bd_2_plot, UB_Bd)
        line_LB.set_data(iter_Bd_2_plot, LB_Bd)
        
        # Dynamically adjust y-axis limits
        new_ylim = (0, min(UB_Bd)+1000)  # Adjust as needed
        plt.gca().set_ylim(new_ylim)
        
        # Dynamically adjust x-axis limits
        new_xlim = (0, max(iter_Bd_2_plot)+1)  # Adjust as needed
        plt.gca().set_xlim(new_xlim)

        plt.draw()
        plt.pause(0.1)  # Adjust the pause time as needed
        
        newgap_Bd = abs(UB-LB)/UB*100
        print(f"Benders gap at iteration {iter_Bd} is {round((UB-LB))} ({round((UB-LB)/UB*100,1)}%).")
        
        if newgap_Bd > my_BD_setup.CONVRG_EPS:
            gap_Bd = min(gap_Bd, newgap_Bd)
        else:
            print("-------------Benders converged!!!\n")
            CONVERGENCE_FLAG = True
    else:
        print(f"Max iterations ({iter_Bd}) exceeded.\n")
    
    end_time_BD_algorithm = time.time()

    elapsed_time_BD_algorithm = end_time_BD_algorithm-start_time_BD_algorithm
    # Convert elapsed time to a timedelta object
    elapsed_time_delta = dtime.timedelta(seconds=elapsed_time_BD_algorithm)

    print(f"\n\n----------------------Bender's algorithm took ---> {str(elapsed_time_delta)} (HH:MM:SS).")
    
else:
    #===============================EACH SLAVE WORKER SOLVES ITS OWN SUB-PROBLEMS (START)=====================================
    for iter_Bd in range(0, BD_MAX_ITER + 1):
        # Worker processes
        while True:
            # Receive task data from the main process
            current_worker_data = MPI.COMM_WORLD.recv(source=0, tag=11)

            if current_worker_data is None or (current_worker_data == 'terminate'):
                # current_worker_active = False
                termination_flag = True
                break

            # Execute the tasks in parallel
            current_worker_res = solve_current_worker_subproblems(current_worker_data)
            # print(f"Process {worker} solved sub-problems {current_worker_data}.")

            # Send result back to the main process
            MPI.COMM_WORLD.send(current_worker_res, dest=0, tag=12)

        if current_worker_data == 'terminate':
            break

    #===============================EACH SLAVE WORKER SOLVES ITS OWN SUB-PROBLEMS (END)=====================================

    #%%
#===============================EXTRACTING RESULTS FROM MASTER WORKER=====================================
if worker == 0:
    #-----------------------------EXTRACTING RESULTS FROM MASTER PROBLEM-----------------------------
    solution_Master_Bd = powergim.dcmp_tools.extract_all_opt_variable_values(master_Bd_model)

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
        
    # -----------------ASSERTIONS
    # CHECKME: Uncomment for automatic validation of mh/bd methods against pgim. 
    # assert math.isclose(pgim_obj_val,mh_obj_val, rel_tol=0.01), "PGIM and MH objective values do not match"
    # assert math.isclose(pgim_obj_val,bd_obj_val, rel_tol=0.01), "PGIM and BD objective values do not match"  
    # assert math.isclose(pgim_opex,bd_opex, rel_tol=0.01), "PGIM and BD operational costs do not match" 

    x_pgim = get_first_stage_decision_pgim(optimal_solution_pgim)
    x_mh   = mh_ref.get_first_stage_decision(optimal_solution_multi_hor)
    x_bd   = mh_ref.get_first_stage_decision(x_current_Bd)


    plt.tight_layout()
    # plt.show()

    # ---------------SAVING FIGURE AND RESULTS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    
    fig2save = PATH_OUTPUT / f"benders_convergence_plot_PARALLEL_case_{GRID_CASE}_stochastic_{IS_STOCHASTIC}_samples_{N_SAMPLES}_multiCut_{not BENDERS_SINGLE_CUT}_fixed_cap_{MH_USE_FIXED_CAP}_{timestamp}.pdf"
    fig.savefig(fig2save, format='pdf')

    now_time = now = datetime.now()
    formatted_now_time = now.strftime("%Y-%B-%A %H:%M:%S")

    data_to_save = {"run_time":formatted_now_time,
                    "Benders_algorithm": 
                    {
                        "duration (HH:MM:SS)": str(elapsed_time_delta),
                        "convergence_gap_relative [%] (UB-LB)/UB": my_BD_setup.CONVRG_EPS,
                        "convergence_gap_relative_to_OBJ_val [%]": round((my_BD_setup.CONVRG_EPS/100*UB_Bd[-1])/bd_obj_val*100,3),
                        "termination_iteration": f"{iter_Bd}/{my_BD_setup.MAX_BD_ITER}",
                        },     
                    "configuration": 
                        {
                            "available_processors": processesToUse,
                            "max_num_iterations": BD_MAX_ITER,
                            "iter_to_stop_prematurely": ITER_TO_STOP,
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
    report_file_out_path = PATH_OUTPUT/f"run_Benders_validation_parallel_{timestamp}.json"

    with open(report_file_out_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)

    print("\n---------------Results file has been written successfully.")


    # Write a PICKLE file for results
    results_file_out_path = PATH_OUTPUT/f"run_Benders_validation_parallel_{timestamp}.pickle"


    print(f"{dtime.datetime.now()}: Dumping results to PICKLE file...")
    dump_dict = {"pgim_rslt": 
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
                        "x_new_capacity": x_bd['new_capacity'].to_dict()
                    }
                }

    with open(results_file_out_path, "wb") as file:
        cloudpickle.dump(dump_dict, file)


    print("\n---------------Results have been pickled successfully.")