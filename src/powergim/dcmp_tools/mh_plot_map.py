import powergim

# CHECKME: should this be a MultiHorizonPgim method instead?
def grid_data_result_mh(ref_grid_data, years, all_var_values):
        """Create grid data representing optimisation results"""

        strategic_nodes = all_var_values['z_new_lines'].index.get_level_values('n_strgc').unique().to_list()
        # operational_nodes = all_var_values['z_generation'].index.get_level_values('n_op').unique().to_list()

        dict_map_n_strgc_to_years = {2035:strategic_nodes[0], 
                                     2040:strategic_nodes[1], 
                                     2050:strategic_nodes[2]}
        
        # dict_map_n_op_to_years = {  2035:operational_nodes[0], 
        #                             2040:operational_nodes[1], 
        #                              2050:operational_nodes[2]}

        nodes = ref_grid_data.node.copy()
        branches = ref_grid_data.branch.copy()
        generators = ref_grid_data.generator.copy()
        consumers = ref_grid_data.consumer.copy()

        is_expanded = all_var_values["z_new_lines"].clip(upper=1).unstack("n_strgc")
        new_branch_cap = is_expanded * all_var_values["z_new_capacity"].unstack("n_strgc")

        for y in years:
            # NOTE: This is to represent generators as blue dots in the graph (to be consistent with grid developements). 
            for inode,node_row_data in nodes.iterrows():
                nodes.at[inode,f"capacity_{y}"] = generators.loc[generators['node']==inode][f"capacity_{y}"].sum()
            
            # Alternatively i can represent all nodes with blue dots by setting a big value.
            # nodes.at[inode,f"capacity_{y}"] = 100000

            branches[f"capacity_{y}"] = branches[f"capacity_{y}"] + new_branch_cap[dict_map_n_strgc_to_years[y]]
            
            branches[f"cum_expand_{y}"] = branches[[f"expand_{p}" for p in years]].sum(axis=1).clip(upper=1)

            #NOTE: Keep lines below commented so i can use the function for both mh and bd methods 
            # The bd method master solution does not contain operation variables: create a different function? 

            # # mean absolute flow:
            # branches[f"flow_{y}"] = (
            #     (
            #         all_var_values["z_flow_12"].unstack("n_op")
            #         + all_var_values["z_flow_21"].unstack("n_op")
            #     )[dict_map_n_op_to_years[y]]
            #     .unstack("t")
            #     .mean(axis=1)
            # )

            # # mean directional flow:
            # branches[f"flow12_{y}"] = (
            #     all_var_values["z_flow_12"].unstack("n_op")[dict_map_n_op_to_years[y]].unstack("t").mean(axis=1)
            # )
            # branches[f"flow21_{y}"] = (
            #     all_var_values["z_flow_21"].unstack("n_op")[dict_map_n_op_to_years[y]].unstack("t").mean(axis=1)
            # )

            # generators[f"output_{y}"] = (
            #     all_var_values["z_generation"].unstack("n_op")[dict_map_n_op_to_years[y]].unstack("t").mean(axis=1)
            # )

        grid_res = powergim.grid_data.GridData(years, nodes, branches, generators, consumers)

        return grid_res