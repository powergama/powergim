"""
VIZUALIZE RESULTS FOR THE DIFFERENT MODELS (PGIM, MH, BD)
1) CREATE INTERACTIVE MAPS (HTML)
2) CREATE GEOPANDAS FIGURES

@author: spyridonc
"""
#%%
import powergim
# from powergim.plots_map2 import plot_map2
from pathlib import Path
from .utils_spi import  CaseMapVisualizer, ReferenceModelCreator
import json
import cloudpickle
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import pandas as pd
from .decomposition import MultiHorizonPgim

#%%
def grid_data_result_mh(ref_grid_data, years, all_var_values):
        """Create GridData pgim object including optimisation results.

        Extends the pgim method `grid_data_result()` for mulit-horizon and 
        decomposition based alternatives."""

        strategic_nodes = all_var_values['z_new_lines'].index.get_level_values('n_strgc').unique().to_list()

        dict_map_n_strgc_to_years = {2035:strategic_nodes[0], 
                                     2040:strategic_nodes[1], 
                                     2050:strategic_nodes[2]}

        nodes = ref_grid_data.node.copy()
        branches = ref_grid_data.branch.copy()
        generators = ref_grid_data.generator.copy()
        consumers = ref_grid_data.consumer.copy()

        is_expanded = all_var_values["z_new_lines"].clip(upper=1).unstack("n_strgc")
        new_branch_cap = is_expanded * all_var_values["z_new_capacity"].unstack("n_strgc")

        for y in years:
            # NOTE: This is to represent generators as blue dots in the graph (to be consistent with grid developements coloring). 
            for inode,node_row_data in nodes.iterrows():
                nodes.at[inode,f"capacity_{y}"] = generators.loc[generators['node']==inode][f"capacity_{y}"].sum()
            
            branches[f"capacity_{y}"] = branches[f"capacity_{y}"] + new_branch_cap[dict_map_n_strgc_to_years[y]]
            
            branches[f"cum_expand_{y}"] = branches[[f"expand_{p}" for p in years]].sum(axis=1).clip(upper=1)

        grid_res = powergim.grid_data.GridData(years, nodes, branches, generators, consumers)

        return grid_res

def get_mh_optimal_solution_per_pgim_scen(scen,mh_sol,mh_ref):
    index_level = 'n_strgc'
    index_values_to_keep = [eval(key[-1]) for key, value in mh_ref.dict_map_strategic_node_to_scenario.items() if value == f'scen{scen}'][-2:] # keep only the two last elements
    #FIXME: this is too hard-coded - i need to find a better way to perform the mapping and always results in P-element lists, where P is the number of periods
    index_values_to_keep.insert(0,0) # adds manually a 0 at the position 0 of the list
    
    filtered_sol = {}
    for key, value in mh_sol.items():
        if isinstance(value, pd.Series) and isinstance(value.index, pd.MultiIndex) and (index_level in value.index.names):
            filtered_sol[key] = value.loc[value.index.get_level_values(index_level).isin(index_values_to_keep)] # Filter the Series to keep only the rows where the index at the specified level matches the desired index value.
        else:
            filtered_sol[key] = value # Retain non-Series items as they are
    
    return filtered_sol



def rename_optimal_solution_vars_per_scen(scen, opt_sol_pgim):
    prefix_to_remove = f'scen{scen}.'

    # Create a new dictionary with renamed keys
    opt_sol_pgim_scen = {
                        (key[len(prefix_to_remove):] if key.startswith(prefix_to_remove) else key): value
                        for key, value in opt_sol_pgim[scen].items()
                        }
    
    # CHECKME: can i avoid selecting the variable sets manually and automate?
    for key, value in opt_sol_pgim_scen.items():
        if isinstance(value, pd.Series) and isinstance(value.index, pd.MultiIndex):
            if f'scen{scen}.s_node' in value.index.names:
                value.index = value.index.set_names('s_node', level=f'scen{scen}.s_node')
            if f'scen{scen}.s_period' in value.index.names:
                value.index = value.index.set_names('s_period', level=f'scen{scen}.s_period')
            if f'scen{scen}.s_gen' in value.index.names:
                value.index = value.index.set_names('s_gen', level=f'scen{scen}.s_gen')
            if f'scen{scen}.s_branch' in value.index.names:
                value.index = value.index.set_names('s_branch', level=f'scen{scen}.s_branch')
            if f'scen{scen}.s_load' in value.index.names:
                value.index = value.index.set_names('s_load', level=f'scen{scen}.s_load')
            if f'scen{scen}.s_time' in value.index.names:
                value.index = value.index.set_names('s_time', level=f'scen{scen}.s_time')
    
    return opt_sol_pgim_scen



def create_interactive_maps_results(case_settings,case_results,path_in,path_out):

    GRID_CASE          = case_settings['configuration']['grid_case']
    BRANCHES_FILE_NAME = case_settings['configuration']['branches_file_used']
    N_SAMPLES          = case_settings['configuration']['number_of_samples']
    PROB               = case_settings['configuration']['scneario_probabilities']

    opt_sol_mh = case_results['mh_rslt']['optimal_solution']
    opt_sol_bd = case_results['bd_rslt']['master_problem_solution']

    #-----------------------------CREATING REF PGIM MODEL-----------------------------
    # The reference is common to all solution approaches (pgim, mh, bd) and it encapsulates the common input data.
    pgim_ref = ReferenceModelCreator(grid_case=GRID_CASE,
                                     branches_file_name=BRANCHES_FILE_NAME,
                                     s_sample_size=N_SAMPLES,
                                     probabilities=PROB
                                     )

    pgim_ref.create_reference_pgim_model(path_in,path_out,scenario_selection='default')
    
    if pgim_ref.grid_case == 'baseline':
        #-----------------------------VISUALIZING PGIM-RESULT-----------------------------
        # NOTE: Interactive map is currently not supported for pgim result. Vali only for mh and bd methods.
        #-----------------------------VISUALIZING MH-RESULT-----------------------------
        case_visualizer = CaseMapVisualizer(pgim_ref,opt_sol_mh,outFileName='case_map_interactive_rslt_mh.html',outPath=path_out,visualize_rslt=True)
        case_visualizer.create_html_map()
        #-----------------------------VISUALIZING BD-RESULT-----------------------------
        case_visualizer = CaseMapVisualizer(pgim_ref,opt_sol_bd,outFileName='case_map_interactive_rslt_bd.html',outPath=path_out,visualize_rslt=True)
        case_visualizer.create_html_map()

        print('-------------Interactive maps of the case results have been created.\n')
    
    return



def create_geopandas_figures_results(timestamp,case_settings,case_results,path_in,path_out):

    GRID_CASE          = case_settings['configuration']['grid_case']
    BRANCHES_FILE_NAME = case_settings['configuration']['branches_file_used']
    N_SAMPLES          = case_settings['configuration']['number_of_samples']
    PROB               = case_settings['configuration']['scneario_probabilities']
    IS_STOCHASTIC      = case_settings['configuration']['is_stochastic']

    opt_sol_pgim = case_results['pgim_rslt']['optimal_solution']
    opt_sol_mh   = case_results['mh_rslt']['optimal_solution']
    opt_sol_bd   = case_results['bd_rslt']['master_problem_solution']

    #-----------------------------CREATING REF PGIM MODEL-----------------------------
    # The reference is common to all solution approaches (pgim, mh, bd) and it encapsulates the common input data.
    pgim_ref = ReferenceModelCreator(grid_case=GRID_CASE,
                                     branches_file_name=BRANCHES_FILE_NAME,
                                     s_sample_size=N_SAMPLES,
                                     probabilities=PROB
                                     )

    pgim_ref.create_reference_pgim_model(path_in,path_out,scenario_selection='default')

    mh_ref = MultiHorizonPgim(pgim_ref, is_stochastic=IS_STOCHASTIC)

    mh_ref.create_ancestors_struct()

    my_cmap = mcolors.ListedColormap(["lightseagreen","darkorange"])

    if not IS_STOCHASTIC:        
        #%=======================PLOT: PGIM=======================
        rslt_grid_data = pgim_ref.ref_pgim_model.grid_data_result(opt_sol_pgim)
        fig,axs = plt.subplots(1,3,figsize=(12,6))
        for i in [0,1,2]:
            years_include = [[_period for _period in pgim_ref.investment_periods][x] for x in range(i+1)]
            print(years_include)
            rslt_grid_data.branch["capacity"] = sum(rslt_grid_data.branch[f"capacity_{year}"] for year in years_include)

            fig=plot_map2(ax=axs[i],
                          grid_data=rslt_grid_data,
                          years=years_include,
                          include_zero_capacity=False,
                          shapefile_path=path_in/'GIS/shapefiles',
                          column = f"expand_{years_include[-1]}",
                          cmap=my_cmap,
                          node_options=dict(markersize=10)
                          )
            
            axs[i].set_title(years_include[-1])
        
        plt.tight_layout()
        plt.show(block=False)  

        # ---------------SAVING FIGURE
        figure_save_path = path_out / f"investments_sequence_plot_pgim_run_{timestamp}.pdf" 
        plt.savefig(figure_save_path, format='pdf')



        #%========================PLOT: MH=======================
        rslt_grid_data = grid_data_result_mh(pgim_ref.ref_grid_data,pgim_ref.investment_periods,opt_sol_mh)

        fig,axs = plt.subplots(1,3,figsize=(12,6))
        for i in [0,1,2]:
            years_include = [[_period for _period in pgim_ref.investment_periods][x] for x in range(i+1)]
            print(years_include)
            rslt_grid_data.branch["capacity"] = sum(rslt_grid_data.branch[f"capacity_{year}"] for year in years_include)
            
            fig=plot_map2(ax=axs[i],
                          grid_data=rslt_grid_data,
                          years=years_include,
                          include_zero_capacity=False,
                          shapefile_path=path_in/'GIS/shapefiles',
                          column = f"expand_{years_include[-1]}",
                          cmap=my_cmap,
                          node_options=dict(markersize=10)
                          )
            
            axs[i].set_title(years_include[-1])
        
        plt.tight_layout()
        plt.show(block=False)  

        # ---------------SAVING FIGURE
        figure_save_path = path_out / f"investments_sequence_plot_mh_run_{timestamp}.pdf" 
        plt.savefig(figure_save_path, format='pdf')





        #%=======================PLOT: BD=======================
        rslt_grid_data = grid_data_result_mh(pgim_ref.ref_grid_data,pgim_ref.investment_periods,opt_sol_bd)

        fig,axs = plt.subplots(1,3,figsize=(12,6))
        for i in [0,1,2]:
            years_include = [[_period for _period in pgim_ref.investment_periods][x] for x in range(i+1)]
            print(years_include)
            rslt_grid_data.branch["capacity"] = sum(rslt_grid_data.branch[f"capacity_{year}"] for year in years_include)
            
            fig=plot_map2(ax=axs[i],
                          grid_data=rslt_grid_data,
                          years=years_include,
                          include_zero_capacity=False,
                          shapefile_path=path_in/'GIS/shapefiles',
                          column = f"expand_{years_include[-1]}",
                          cmap=my_cmap,
                          node_options=dict(markersize=10)
                          )
            
            axs[i].set_title(years_include[-1])
        
        plt.tight_layout()
        plt.show(block=False)  

        # ---------------SAVING FIGURE
        figure_save_path = path_out / f"investments_sequence_plot_bd_run_{timestamp}.pdf" 
        plt.savefig(figure_save_path, format='pdf')


        print('-------------Investments sequence plots for the case results have been created.\n')

    else:

        for scen in range(len(opt_sol_pgim)):

            opt_sol_pgim_scen = rename_optimal_solution_vars_per_scen(scen,opt_sol_pgim)       
        
            #%=======================PLOT: PGIM=======================
            rslt_grid_data = pgim_ref.ref_pgim_model.grid_data_result(opt_sol_pgim_scen)
            fig,axs = plt.subplots(1,3,figsize=(12,6))
            for i in [0,1,2]:
                years_include = [[_period for _period in pgim_ref.investment_periods][x] for x in range(i+1)]
                print(years_include)
                rslt_grid_data.branch["capacity"] = sum(rslt_grid_data.branch[f"capacity_{year}"] for year in years_include)
                
                fig=plot_map2(ax=axs[i],
                              grid_data=rslt_grid_data,
                              years=years_include,
                              include_zero_capacity=False,
                              shapefile_path=path_in/'GIS/shapefiles',
                              column = f"expand_{years_include[-1]}",
                              cmap=my_cmap,
                              node_options=dict(markersize=10)
                              )
                
                axs[i].set_title(years_include[-1])
            
            plt.tight_layout()
            plt.show(block=False)  

            # ---------------SAVING FIGURE
            figure_save_path = path_out / f"investments_sequence_plot_pgim_scen_{scen}_run_{timestamp}.pdf" 
            plt.savefig(figure_save_path, format='pdf')



            #%========================PLOT: MH=======================
            filtered_opt_sol_mh_scen = get_mh_optimal_solution_per_pgim_scen(scen,opt_sol_mh,mh_ref)

            rslt_grid_data = grid_data_result_mh(pgim_ref.ref_grid_data,pgim_ref.investment_periods,filtered_opt_sol_mh_scen)

            fig,axs = plt.subplots(1,3,figsize=(12,6))
            for i in [0,1,2]:
                years_include = [[_period for _period in pgim_ref.investment_periods][x] for x in range(i+1)]
                print(years_include)
                rslt_grid_data.branch["capacity"] = sum(rslt_grid_data.branch[f"capacity_{year}"] for year in years_include)
                
                fig=plot_map2(ax=axs[i],
                              grid_data=rslt_grid_data,
                              years=years_include,
                              include_zero_capacity=False,
                              shapefile_path=path_in/'GIS/shapefiles',
                              column = f"cum_expand_{years_include[-1]}",
                              cmap=my_cmap,
                              node_options=dict(markersize=10)
                              )
                
                axs[i].set_title(years_include[-1])
            
            plt.tight_layout()
            plt.show(block=False)  

            # ---------------SAVING FIGURE
            figure_save_path = path_out / f"investments_sequence_plot_mh_scen_{scen}_run_{timestamp}.pdf" 
            plt.savefig(figure_save_path, format='pdf')





            #%=======================PLOT: BD=======================
            filtered_opt_sol_bd_scen = get_mh_optimal_solution_per_pgim_scen(scen,opt_sol_bd,mh_ref)

            rslt_grid_data = grid_data_result_mh(pgim_ref.ref_grid_data,pgim_ref.investment_periods,filtered_opt_sol_bd_scen)

            fig,axs = plt.subplots(1,3,figsize=(12,6))
            for i in [0,1,2]:
                years_include = [[_period for _period in pgim_ref.investment_periods][x] for x in range(i+1)]
                print(years_include)
                rslt_grid_data.branch["capacity"] = sum(rslt_grid_data.branch[f"capacity_{year}"] for year in years_include)
                
                fig=plot_map2(ax=axs[i],
                              grid_data=rslt_grid_data,
                              years=years_include,
                              include_zero_capacity=False,
                              shapefile_path=path_in/'GIS/shapefiles',
                              column = f"cum_expand_{years_include[-1]}",
                              cmap=my_cmap,
                              node_options=dict(markersize=10)
                              )
                
                axs[i].set_title(years_include[-1])
            
            plt.tight_layout()
            plt.show(block=False)  

            # ---------------SAVING FIGURE
            figure_save_path = path_out / f"investments_sequence_plot_bd_scen_{scen}_run_{timestamp}.pdf" 
            plt.savefig(figure_save_path, format='pdf')


            print('-------------Investments sequence plots for the case results have been created.\n')

    return





if __name__ == "__main__":

    PATH_INPUT  = Path(__file__).parents[3] / "examples"/ "inputs"/"CASE_BASELINE"
    PATH_OUTPUT = Path(__file__).parents[3] / "examples"/ "outputs"

    timestamp = '20250117_090747' # CHECKME: choose an existing run case

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
    
    if run_settings['configuration']['grid_case'] == 'baseline':
        create_interactive_maps_results(run_settings,run_rslt,PATH_INPUT,PATH_OUTPUT)
        create_geopandas_figures_results(timestamp,run_settings,run_rslt,PATH_INPUT,PATH_OUTPUT)