"""
Decomposition classes definitions:
- `MultiHorizonPgim`
- `BendersDecomp`
- `DantzigWolfeDecomp`
---
@author: spyridonc
"""
import copy
import math

import pandas as pd
import pyomo.environ as pyo

import powergim as pgim

from .utils_spi import ParamsExtractor


class MultiHorizonPgim:
    """
     Re-formulate `pgim` as a multi-horizon (MH) problem.

     Mapping of operational to strategic nodes for MH formulation:

     -----Deterministic:
    {'n_op_1' : 'n_strgc_0', 'n_op_2' : 'n_strgc_1', 'n_op_3' : 'n_strgc_2'}

     -----Stochastic:
     {
     'n_op_1' : 'n_strgc_0', 'n_op_2' : 'n_strgc_0', 'n_op_3' : 'n_strgc_0',

     'n_op_4' : 'n_strgc_1', 'n_op_5' : 'n_strgc_2', 'n_op_6' : 'n_strgc_3',

     'n_op_7' : 'n_strgc_4', 'n_op_8' : 'n_strgc_5', 'n_op_9' : 'n_strgc_6'
     }

     TODO: Create the MH tree for arbitary scenarios number, automatically.

    """

    def __init__(self, pgim_case, is_stochastic=False, ancestors_include_current=False):

        self.pgim_case = pgim_case
        self.branch_input_file_name = pgim_case.branch_input_file_name
        self.is_stochastic = is_stochastic
        self.s_sample_size = pgim_case.s_sample_size
        self.sample_random_state = pgim_case.sample_random_state
        self.investment_periods = pgim_case.investment_periods
        self.grid_case = pgim_case.grid_case
        self.prob = pgim_case.probabilities
        self.ancestors_include_current = ancestors_include_current

        if self.grid_case == "star":
            self.s_nodes = 4
            self.randomize_profiles = False

    # TODO: find automatic way to create the ancestors dict (scenario tree) based on number of scenarios.
    def create_ancestors_struct(self):
        """Define the MH tree structure."""
        # DETERMINISTIC
        if not self.is_stochastic:
            self.s_strategic_nodes = 3
            self.s_operational_nodes = self.s_strategic_nodes

            self.dict_strategic_nodes_periods = {
                "n_strgc_0": self.investment_periods[0],
                "n_strgc_1": self.investment_periods[1],
                "n_strgc_2": self.investment_periods[2],
            }
            self.dict_operational_nodes_periods = {
                "n_op_1": self.investment_periods[0],
                "n_op_2": self.investment_periods[1],
                "n_op_3": self.investment_periods[2],
            }

            # Set of the strategic nodes that have ancestors (multi-Horizon formulation)
            if self.ancestors_include_current:
                # This is used with Dantzig–Wolfe decomposition methods
                self.dict_ancestors_set = {"n_strgc_0": [0], "n_strgc_1": [0, 1], "n_strgc_2": [0, 1, 2]}
            else:
                # This is used with Benders decomposition methods
                self.dict_ancestors_set = {"n_strgc_1": [0], "n_strgc_2": [0, 1]}

            # They remain hard-coded for deterministic problem
            self.dict_strategic_nodes_probabilities = {"n_strgc_0": 1, "n_strgc_1": 1, "n_strgc_2": 1}
            self.dict_operational_nodes_probabilities = {"n_op_1": 1, "n_op_2": 1, "n_op_3": 1}
            self.scenario_names = [list(self.prob.keys())[0]]
            self.dict_map_strategic_node_to_scenario = {
                "n_strgc_0": self.scenario_names[0],
                "n_strgc_1": self.scenario_names[0],
                "n_strgc_2": self.scenario_names[0],
            }
            self.dict_map_operational_node_to_scenario = {
                "n_op_1": self.scenario_names[0],
                "n_op_2": self.scenario_names[0],
                "n_op_3": self.scenario_names[0],
            }

            self.dict_map_operational_to_strategic_node = {"n_op_1": 0, "n_op_2": 1, "n_op_3": 2}

        # STOCHASTIC:
        elif self.is_stochastic:
            self.s_strategic_nodes = 7
            self.s_operational_nodes = self.s_strategic_nodes + 2

            self.dict_strategic_nodes_periods = {
                "n_strgc_0": self.investment_periods[0],
                "n_strgc_1": self.investment_periods[1],
                "n_strgc_2": self.investment_periods[1],
                "n_strgc_3": self.investment_periods[1],
                "n_strgc_4": self.investment_periods[2],
                "n_strgc_5": self.investment_periods[2],
                "n_strgc_6": self.investment_periods[2],
            }

            self.dict_operational_nodes_periods = {
                "n_op_1": self.investment_periods[0],
                "n_op_2": self.investment_periods[0],
                "n_op_3": self.investment_periods[0],
                "n_op_4": self.investment_periods[1],
                "n_op_5": self.investment_periods[1],
                "n_op_6": self.investment_periods[1],
                "n_op_7": self.investment_periods[2],
                "n_op_8": self.investment_periods[2],
                "n_op_9": self.investment_periods[2],
            }

            # Set of the strategic nodes that have ancestors (multi-Horizon formulation)
            if self.ancestors_include_current:
                # This is used with Dantzig–Wolfe decomposition methods
                self.dict_ancestors_set = {
                    "n_strgc_0": [0],
                    "n_strgc_1": [0, 1],
                    "n_strgc_2": [0, 2],
                    "n_strgc_3": [0, 3],
                    "n_strgc_4": [0, 1, 4],
                    "n_strgc_5": [0, 2, 5],
                    "n_strgc_6": [0, 3, 6],
                }
            else:
                # This is used with Benders decomposition methods
                self.dict_ancestors_set = {
                    "n_strgc_1": [0],
                    "n_strgc_2": [0],
                    "n_strgc_3": [0],
                    "n_strgc_4": [0, 1],
                    "n_strgc_5": [0, 2],
                    "n_strgc_6": [0, 3],
                }

            self.dict_strategic_nodes_probabilities = {
                "n_strgc_0": 1,
                "n_strgc_1": self.prob["scen1"],
                "n_strgc_4": self.prob["scen1"],
                "n_strgc_2": self.prob["scen0"],
                "n_strgc_5": self.prob["scen0"],
                "n_strgc_3": self.prob["scen2"],
                "n_strgc_6": self.prob["scen2"],
            }

            self.dict_operational_nodes_probabilities = {
                "n_op_1": self.prob["scen1"],
                "n_op_4": self.prob["scen1"],
                "n_op_7": self.prob["scen1"],
                "n_op_2": self.prob["scen0"],
                "n_op_5": self.prob["scen0"],
                "n_op_8": self.prob["scen0"],
                "n_op_3": self.prob["scen2"],
                "n_op_6": self.prob["scen2"],
                "n_op_9": self.prob["scen2"],
            }

            self.scenario_names = list(self.prob.keys())

            self.dict_map_strategic_node_to_scenario = {
                "n_strgc_0": self.scenario_names[0],
                "n_strgc_1": self.scenario_names[1],
                "n_strgc_4": self.scenario_names[1],
                "n_strgc_2": self.scenario_names[0],
                "n_strgc_5": self.scenario_names[0],
                "n_strgc_3": self.scenario_names[2],
                "n_strgc_6": self.scenario_names[2],
            }

            self.dict_map_operational_node_to_scenario = {
                "n_op_1": self.scenario_names[1],
                "n_op_4": self.scenario_names[1],
                "n_op_7": self.scenario_names[1],
                "n_op_2": self.scenario_names[0],
                "n_op_5": self.scenario_names[0],
                "n_op_8": self.scenario_names[0],
                "n_op_3": self.scenario_names[2],
                "n_op_6": self.scenario_names[2],
                "n_op_9": self.scenario_names[2],
            }

            self.dict_map_operational_to_strategic_node = {
                "n_op_1": 0,
                "n_op_2": 0,
                "n_op_3": 0,
                "n_op_4": 1,
                "n_op_5": 2,
                "n_op_6": 3,
                "n_op_7": 4,
                "n_op_8": 5,
                "n_op_9": 6,
            }

    def get_pgim_params_per_scenario(self):
        """Extract scenario parameters from reference `pgim` model."""

        self.params_per_scenario = dict.fromkeys(self.scenario_names, None)
        _my_pgim_model_per_scen = dict.fromkeys(self.scenario_names, None)

        for iscen in self.scenario_names:

            # Re-initialize placeholder for scenario params.
            _my_params = dict.fromkeys(
                [
                    "max_number_cables",
                    "max_new_branch_capacity",
                    "max_gen_power",
                    "max_load",
                    "IC_new_cables_coefs",
                    "IC_new_capacity_coefs",
                    "OC_gen_coefs",
                    "OC_load_shed_coefs",
                    "branch_losses",
                ],
                None,
            )

            # Re-initialize local pgim instance from the "reference" pgim model.
            # This is used to create variants for the different scenarios.

            _parameter_data = copy.deepcopy(self.pgim_case.ref_params)
            _grid_data = copy.deepcopy(self.pgim_case.ref_grid_data)

            match iscen:

                case "scen0":  # Scenario 0 (nominal)
                    ...
                case "scen1":  # Scenario 1

                    match self.grid_case:

                        case "star":
                            # "Half wind, same demand"

                            # Half the wind at n1 (wind farm node).
                            _init_wind_capacity = _grid_data.generator.loc[_grid_data.generator["node"] == "n1"]

                            for iperiod in _parameter_data["parameters"]["investment_years"]:
                                _grid_data.generator.loc[
                                    _grid_data.generator["node"] == "n1", ["capacity_" + str(iperiod)]
                                ] = (0.5 * _init_wind_capacity.loc[0, "capacity_" + str(iperiod)])

                            # ------ALTERNATIVE SCENARIO
                            # Half the load at n2 (country node).
                            # _init_load_capacity = _grid_data.consumer.loc[_grid_data.consumer['node'] == 'n2']

                            # for iperiod in _parameter_data['parameters']['investment_years']:
                            #     _grid_data.consumer.loc[_grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 0.5*_init_load_capacity.loc[0,'demand_' + str(iperiod)]

                        case "baseline":
                            #  Less NO demand (220 TWh in 2050)

                            demand_scale = 220 / 260
                            # demand_scale = 130/260
                            m_demand = _grid_data.consumer["node"].str.startswith("NO_")
                            for year in _parameter_data["parameters"]["investment_years"]:
                                _grid_data.consumer.loc[m_demand, f"demand_{year}"] = (
                                    demand_scale * _grid_data.consumer.loc[m_demand, f"demand_{year}"]
                                )
                                # _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 0.5 * _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]
                                # _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 2 * _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                case "scen2":  # Scenario 2

                    match self.grid_case:
                        case "star":
                            # "Same wind, double demand"

                            # Double the load at n3 (offshore load node).
                            _init_load_capacity = _grid_data.consumer.loc[_grid_data.consumer["node"] == "n3"]

                            for iperiod in _parameter_data["parameters"]["investment_years"]:
                                _grid_data.consumer.loc[
                                    _grid_data.consumer["node"] == "n3", ["demand_" + str(iperiod)]
                                ] = (2 * _init_load_capacity.loc[1, "demand_" + str(iperiod)])

                            # Alternative example: change the fuel price for this scenario, at 2050.
                            # for iperiod in [my_star_params['parameters']['investment_years'][-1]]:
                            #     _grid_data.generator.loc[_grid_data.generator['type'] == 'gentype1', ['fuelcost_' + str(iperiod)]] = 120

                            # ------ALTERNATIVE SCENARIO
                            # Double the load at n2 (country node).
                            # _init_load_capacity = _grid_data.consumer.loc[_grid_data.consumer['node'] == 'n2']

                            # for iperiod in _parameter_data['parameters']['investment_years']:
                            #     _grid_data.consumer.loc[_grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 2*_init_load_capacity.loc[0,'demand_' + str(iperiod)]

                        case "baseline":
                            #  More NO demand (340 TWh in 2050)

                            demand_scale = 340 / 260
                            # demand_scale = 520/260
                            m_demand = _grid_data.consumer["node"].str.startswith("NO_")
                            for year in _parameter_data["parameters"]["investment_years"]:
                                _grid_data.consumer.loc[m_demand, f"demand_{year}"] = (
                                    demand_scale * _grid_data.consumer.loc[m_demand, f"demand_{year}"]
                                )
                                # _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 2 * _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]
                                # _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 0.5 * _grid_data.consumer.loc[_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                case _:
                    raise ValueError("Invalid scenario name.")

            _my_pgim_model_per_scen[iscen] = pgim.SipModel(_grid_data, _parameter_data)
            _extractor = ParamsExtractor(_my_pgim_model_per_scen[iscen])

            _my_params["max_number_cables"] = _extractor.get_max_number_cables()
            _my_params["max_new_branch_capacity"] = _extractor.get_max_new_branch_capacity()
            _my_params["max_gen_power"] = _extractor.get_max_gen_power()
            _my_params["max_load"] = _extractor.get_max_load()

            (
                _my_params["IC_new_cables_coefs"],
                _my_params["IC_new_capacity_coefs"],
            ) = _extractor.get_investment_cost_coefs()
            _my_params["OC_gen_coefs"], _my_params["OC_load_shed_coefs"] = _extractor.get_operation_cost_coefs()
            _my_params["branch_losses"] = _extractor.get_branches_losses()

            self.params_per_scenario[iscen] = _my_params

    def create_multi_horizon_problem(self, USE_BIN_EXPANS=False, USE_FIXED_CAP_LINES=False):
        """Create non-decomposed MH optimization problem.

        Requires:
        - MH tree (`create_ancestors_struct()`)
        - MH parameters (`get_pgim_params_per_scenario()`)
        """

        try:
            self.params_per_scenario
        except NameError:
            print("Parameters per scenario for multi-horizon fomrulation are not defined.")
        else:
            print("---------------------------------\n \n Creating multi-horizon formulation...\n \n")

        self.USE_BIN_EXPANS = USE_BIN_EXPANS
        self.USE_FIXED_CAP_LINES = USE_FIXED_CAP_LINES

        self.s_branches = len(self.pgim_case.ref_grid_data.branch)
        self.s_generators = len(self.pgim_case.ref_grid_data.generator)
        self.s_loads = len(self.pgim_case.ref_grid_data.consumer)
        if self.USE_BIN_EXPANS:
            self.k_bin_expans = int(math.log2(5))

        model = pyo.ConcreteModel(name="Non-decomposed problem")

        # --------------------------------SETS
        model.t = pyo.RangeSet(0, self.s_sample_size - 1)  # Set of operational timesteps with duration T (Horizon)
        model.n_elctr = pyo.Set(initialize=self.pgim_case.ref_pgim_model.s_node)  # Set of electrical (physical) nodes
        model.n_strgc = pyo.RangeSet(0, self.s_strategic_nodes - 1)  # Set strategic nodes (multi-Horizon formulation)
        model.n_op = pyo.RangeSet(1, self.s_operational_nodes)  # Set operational nodes (multi-Horizon formulation)
        model.gen = pyo.RangeSet(0, self.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.s_loads - 1)  # Set of loads
        model.branch = pyo.RangeSet(0, self.s_branches - 1)  # Set of branches
        if self.USE_BIN_EXPANS:
            model.bin_expans_var = pyo.RangeSet(0, self.k_bin_expans)  # Set of binary expansion variables

        # ----------------------------VARIABLES
        # (INVESTEMENT)
        if self.USE_BIN_EXPANS:
            model.z_new_lines_bin = pyo.Var(model.branch, model.n_strgc, model.bin_expans_var, within=pyo.Binary)
        else:
            model.z_new_lines = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeIntegers)

        model.z_new_capacity = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)
        model.z_capacity_total = pyo.Var(
            model.branch, model.n_strgc, within=pyo.NonNegativeReals
        )  # This is the only complicating variable

        # (OPERATION)
        model.z_generation = pyo.Var(model.gen, model.n_op, model.t, within=pyo.NonNegativeReals)
        model.z_load_shed = pyo.Var(model.ld, model.n_op, model.t, within=pyo.NonNegativeReals)
        model.z_flow_12 = pyo.Var(model.branch, model.n_op, model.t, within=pyo.NonNegativeReals)
        model.z_flow_21 = pyo.Var(model.branch, model.n_op, model.t, within=pyo.NonNegativeReals)

        # --------------------------CONTRAINTS

        if self.USE_BIN_EXPANS:
            # Define rule to represent integer values using binary variables
            def rule_integer_representation_z_new_lines(model, branch, n_strgc):
                return sum(2**i * model.z_new_lines_bin[branch, n_strgc, i] for i in range(self.k_bin_expans))

        # CUMULATIVE NEW CAPACITY PER STRATEGIC NODE
        def rule_aggregate_new_capacity(model, branch, n_strgc):
            if n_strgc == 0:
                existing_capacity = self.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)])
                ]

                return (
                    model.z_capacity_total[branch, n_strgc] == existing_capacity + model.z_new_capacity[branch, n_strgc]
                )

            else:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                if self.ancestors_include_current:
                    ancestor_nodes = self.dict_ancestors_set[current_strgc_node][:-1]
                else:
                    ancestor_nodes = self.dict_ancestors_set[current_strgc_node]

                previous_periods = [
                    self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)] for n_strgc in ancestor_nodes
                ]
                existing_capacity = self.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)] for pr in previous_periods
                )

                return (
                    model.z_capacity_total[branch, n_strgc]
                    == existing_capacity
                    + sum(model.z_new_capacity[branch, node] for node in ancestor_nodes)
                    + model.z_new_capacity[branch, n_strgc]
                )

        model.c_rule_aggregate_new_capacity = pyo.Constraint(
            model.branch, model.n_strgc, rule=rule_aggregate_new_capacity
        )

        # NODAL (electrical) POWER BALANCE
        def rule_nodal_power_balance(model):
            # For the current operational node, scann electrical nodes and add corresponding variables to the node depedning if this has a gen, a load, or is connected to a branch.
            temp_pwr_expr = 0

            # Find branches that are node_from from the current electrical node (add power flow 12).
            branches_from = self.pgim_case.ref_grid_data.branch.loc[
                self.pgim_case.ref_grid_data.branch["node_from"] == n_elctr
            ]

            if len(branches_from) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]][
                            "branch_losses"
                        ].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_21[branch, n_op, t]
                    for branch in branches_from.index.to_list()
                ) - sum(model.z_flow_12[branch, n_op, t] for branch in branches_from.index.to_list())

            # Find branches that are node_to from the current electrical node (add power flow 21).
            branches_to = self.pgim_case.ref_grid_data.branch.loc[
                self.pgim_case.ref_grid_data.branch["node_to"] == n_elctr
            ]

            if len(branches_to) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]][
                            "branch_losses"
                        ].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_12[branch, n_op, t]
                    for branch in branches_to.index.to_list()
                ) - sum(model.z_flow_21[branch, n_op, t] for branch in branches_to.index.to_list())

            # Check if there are generators to add in the current electrical node (add power generation).
            if n_elctr in pd.Series(self.pgim_case.ref_grid_data.generator["node"].to_list()).unique().tolist():
                for n_electr_index in self.pgim_case.ref_grid_data.generator.index[
                    self.pgim_case.ref_grid_data.generator["node"] == n_elctr
                ]:
                    temp_pwr_expr += model.z_generation[n_electr_index, n_op, t]

            # Check if there are consumers to add, in the current electrical node (add load shedding, subtract demand).
            if n_elctr in pd.Series(self.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist():
                for n_electr_index in self.pgim_case.ref_grid_data.consumer.index[
                    self.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                ]:
                    temp_pwr_expr += (
                        model.z_load_shed[n_electr_index, n_op, t]
                        - self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]][
                            "max_load"
                        ].loc[
                            (n_electr_index, self.dict_operational_nodes_periods["n_op_" + str(n_op)], t),
                            "parameter_value",
                        ]
                    )

            return temp_pwr_expr == 0

        # Scann all operational nodes and enforce power balance for all electrical nodes.
        for n_op in model.n_op:
            for t in model.t:
                for n_elctr in model.n_elctr:
                    model.add_component(
                        f"Constraint_node_{n_elctr}_power_balance_at_{n_op}_operational_node_time_{t}",
                        pyo.Constraint(rule=rule_nodal_power_balance),
                    )

        # LOAD SHEDDING UPPER BOUND
        def rule_load_shedding_limit(model):
            max_load = self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]][
                "max_load"
            ].loc[(n_electr_index, self.dict_operational_nodes_periods["n_op_" + str(n_op)], t), "parameter_value"]
            return model.z_load_shed[n_electr_index, n_op, t] <= max_load

        for n_op in model.n_op:
            for t in model.t:
                for n_elctr in model.n_elctr:
                    # Check where consumers exists to limit the corresponding load shedding variable.
                    if n_elctr in pd.Series(self.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist():
                        for n_electr_index in self.pgim_case.ref_grid_data.consumer.index[
                            self.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                        ]:
                            model.add_component(
                                f"Constraint_node_{n_elctr}_consumer_{n_electr_index}_load_shedding_upper_bound_at_{n_op}_operational_node_time_{t}",
                                pyo.Constraint(rule=rule_load_shedding_limit),
                            )

        # GENERATOR POWER UPPER BOUND
        def rule_gen_upper_bound(model):
            max_gen_output = self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]][
                "max_gen_power"
            ].loc[(n_electr_index, self.dict_operational_nodes_periods["n_op_" + str(n_op)], t), "parameter_value"]
            return model.z_generation[n_electr_index, n_op, t] <= max_gen_output

        for n_op in model.n_op:
            for t in model.t:
                for n_elctr in model.n_elctr:
                    # Check where generators exists to limit the corresponding generator output power variable.
                    if n_elctr in pd.Series(self.pgim_case.ref_grid_data.generator["node"].to_list()).unique().tolist():
                        for n_electr_index in self.pgim_case.ref_grid_data.generator.index[
                            self.pgim_case.ref_grid_data.generator["node"] == n_elctr
                        ]:
                            model.add_component(
                                f"Constraint_node_{n_elctr}_gen_{n_electr_index}_upper_bound_at_{n_op}_operational_node_time_{t}",
                                pyo.Constraint(rule=rule_gen_upper_bound),
                            )

        # BRANCH POWER FLOW LIMITS 12
        def rule_branch_max_flow12(model, branch, n_op, t):
            return (
                model.z_flow_12[branch, n_op, t]
                <= model.z_capacity_total[branch, self.dict_map_operational_to_strategic_node["n_op_" + str(n_op)]]
            )

        model.c_rule_branch_max_flow12 = pyo.Constraint(model.branch, model.n_op, model.t, rule=rule_branch_max_flow12)

        # BRANCH POWER FLOW LIMITS 21
        # NOTE: the only difference compared to the constraint above (# BRANCH POWER FLOW LIMITS 12) is that variables and constraint names are 21 instead of 12.
        def rule_branch_max_flow21(model, branch, n_op, t):
            return (
                model.z_flow_21[branch, n_op, t]
                <= model.z_capacity_total[branch, self.dict_map_operational_to_strategic_node["n_op_" + str(n_op)]]
            )

        model.c_rule_branch_max_flow21 = pyo.Constraint(model.branch, model.n_op, model.t, rule=rule_branch_max_flow21)

        # MAX NUMBER OF CABLES (new_lines) PER BRANCH
        def rule_max_cables_per_branch(model, branch, n_strgc):
            if self.USE_BIN_EXPANS:
                return (
                    rule_integer_representation_z_new_lines(model, branch, n_strgc)
                    <= self.params_per_scenario[self.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]][
                        "max_number_cables"
                    ].loc[(branch, self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]), "parameter_value"]
                )
            else:
                return (
                    model.z_new_lines[branch, n_strgc]
                    <= self.params_per_scenario[self.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]][
                        "max_number_cables"
                    ].loc[(branch, self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]), "parameter_value"]
                )

        model.c_rule_max_cables_per_branch = pyo.Constraint(
            model.branch, model.n_strgc, rule=rule_max_cables_per_branch
        )

        # MAX NEW CAPACITY PER CABLE (line)
        if self.USE_FIXED_CAP_LINES:

            def rule_max_capacity_per_cable(model, branch, n_strgc):
                if self.USE_BIN_EXPANS:
                    return model.z_new_capacity[branch, n_strgc] == self.params_per_scenario[
                        self.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                    ]["max_new_branch_capacity"].loc[
                        (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]), "parameter_value"
                    ] * rule_integer_representation_z_new_lines(
                        model, branch, n_strgc
                    )
                else:
                    return (
                        model.z_new_capacity[branch, n_strgc]
                        == self.params_per_scenario[
                            self.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]), "parameter_value"
                        ]
                        * model.z_new_lines[branch, n_strgc]
                    )

            model.c_rule_max_capacity_per_cable = pyo.Constraint(
                model.branch, model.n_strgc, rule=rule_max_capacity_per_cable
            )

        else:

            def rule_max_capacity_per_cable(model, branch, n_strgc):
                if self.USE_BIN_EXPANS:
                    return model.z_new_capacity[branch, n_strgc] <= self.params_per_scenario[
                        self.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                    ]["max_new_branch_capacity"].loc[
                        (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]), "parameter_value"
                    ] * rule_integer_representation_z_new_lines(
                        model, branch, n_strgc
                    )
                else:
                    return (
                        model.z_new_capacity[branch, n_strgc]
                        <= self.params_per_scenario[
                            self.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]), "parameter_value"
                        ]
                        * model.z_new_lines[branch, n_strgc]
                    )

            model.c_rule_max_capacity_per_cable = pyo.Constraint(
                model.branch, model.n_strgc, rule=rule_max_capacity_per_cable
            )

        # -------------------------OBJECTIVE FUNCTION
        def obj_expression(model):

            all_nodes_expr = 0
            # Loop trhough the strategic nodes.
            for node in model.n_strgc:
                node_expr = 0
                # Investment Costs (IC): Loop trhough the branches.
                for branch in model.branch:

                    if self.USE_BIN_EXPANS:
                        node_expr += self.params_per_scenario[
                            self.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                        ]["IC_new_cables_coefs"].loc[
                            (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(node)]), "parameter_value"
                        ] * rule_integer_representation_z_new_lines(
                            model, branch, node
                        )
                        node_expr += (
                            self.params_per_scenario[self.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]][
                                "IC_new_capacity_coefs"
                            ].loc[
                                (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(node)]), "parameter_value"
                            ]
                            * model.z_new_capacity[branch, node]
                        )
                    else:
                        node_expr += (
                            self.params_per_scenario[self.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]][
                                "IC_new_cables_coefs"
                            ].loc[
                                (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(node)]), "parameter_value"
                            ]
                            * model.z_new_lines[branch, node]
                        )
                        node_expr += (
                            self.params_per_scenario[self.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]][
                                "IC_new_capacity_coefs"
                            ].loc[
                                (branch, self.dict_strategic_nodes_periods["n_strgc_" + str(node)]), "parameter_value"
                            ]
                            * model.z_new_capacity[branch, node]
                        )

                node_expr *= self.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

                all_nodes_expr += node_expr

            # Loop trhough the operational nodes.
            for node in model.n_op:
                node_expr = 0
                # Operation Costs (OC): Loop trhough the timestep samples.
                for t in model.t:
                    for gen in (
                        self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(node)]][
                            "OC_gen_coefs"
                        ]
                        .index.get_level_values("gen")
                        .unique()
                        .to_list()
                    ):

                        node_expr += (
                            self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(node)]][
                                "OC_gen_coefs"
                            ].loc[(gen, self.dict_operational_nodes_periods["n_op_" + str(node)], t), "parameter_value"]
                            * model.z_generation[gen, node, t]
                        )

                    for ld in (
                        self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(node)]][
                            "OC_load_shed_coefs"
                        ]
                        .index.get_level_values("load")
                        .unique()
                        .to_list()
                    ):

                        node_expr += (
                            self.params_per_scenario[self.dict_map_operational_node_to_scenario["n_op_" + str(node)]][
                                "OC_load_shed_coefs"
                            ].loc[(ld, self.dict_operational_nodes_periods["n_op_" + str(node)], t), "parameter_value"]
                            * model.z_load_shed[ld, node, t]
                        )

                node_expr *= self.dict_operational_nodes_probabilities["n_op_" + str(node)]

                all_nodes_expr += node_expr

            return all_nodes_expr

        model.objective = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        self.non_decomposed_model = model

    # =======================================STATIC METHODS

    # Define rule to represent integer values using binary variables.
    @staticmethod
    def extract_z_new_lines_from_bin_expans(solution):
        try:
            solution["z_new_lines_bin"]
        except NameError:
            print("Binary expansion variables have not been defined in the multi-horizon formulation.")
        else:

            def _bin_to_int(bin_index_value, bin_value):
                return 2**bin_index_value * bin_value

            integer_z_new_lines = pd.Series(
                [_bin_to_int(i[-1], v) for i, v in solution["z_new_lines_bin"].items()],
                index=solution["z_new_lines_bin"].index,
                name=solution["z_new_lines_bin"].name,
            )
            return integer_z_new_lines.groupby(["branch", "n_strgc"]).sum()

    # Get the first stage solution of a MH 2-stage SO problem.
    def get_first_stage_decision(self, solution):
        n_strgc = 0
        if not self.USE_BIN_EXPANS:
            x_cbl = (
                solution["z_new_lines"]
                .xs(n_strgc, level=1)
                .loc[solution["z_new_lines"].xs(n_strgc, level=1).values > 0.01]
            )
        else:
            non_decomposed_integer_z_new_lines = MultiHorizonPgim.extract_z_new_lines_from_bin_expans(solution)
            x_cbl = non_decomposed_integer_z_new_lines.xs(n_strgc, level=1).loc[
                non_decomposed_integer_z_new_lines.xs(n_strgc, level=1).values > 0.01
            ]
        x = {"new_lines": x_cbl}

        if not self.USE_FIXED_CAP_LINES:
            x_cpt = (
                solution["z_new_capacity"]
                .xs(n_strgc, level=1)
                .loc[solution["z_new_capacity"].xs(n_strgc, level=1).values > 0.01]
            )
            # x_cpt_tot = solution['z_capacity_total'].xs(n_strgc,level=1).loc[solution['z_capacity_total'].xs(n_strgc,level=1).values>0.01]
            # x = {'new_lines':x_cbl, 'new_capacity':x_cpt, 'new_capacity_total':x_cpt_tot}
        else:
            x_cpt = pd.Series([None] * len(x_cbl), index=x_cbl.index)
        x["new_capacity"] = x_cpt

        return x


class BendersDecomp:
    """Setup for Benders decomposition methods (BD), algorithm settings, and relevant optimization problems.

    This is to be used with ancestors sets that do NOT include current.
    (`multi_horizon_instance.ancestors_include_current = False`)

    """

    def __init__(self, multi_hor_instance, INIT_LB=0, INIT_UB=350000, MAX_BD_ITER=1, CONVRG_EPS=50):
        self.INIT_LB = INIT_LB
        self.INIT_UB = INIT_UB
        self.MAX_BD_ITER = MAX_BD_ITER
        self.CONVRG_EPS = CONVRG_EPS

        if not isinstance(multi_hor_instance, MultiHorizonPgim):  # Check if 'custom_obj' is an instance of CustomClass
            raise TypeError("Argument must be of class type 'MultiHorizonPgim'")
        else:
            self.mh_model = multi_hor_instance

    def _integer_representation_z_new_lines_bin(self, model, branch, n_strgc):
        return sum(2**i * model.z_new_lines_bin[branch, n_strgc, i] for i in range(self.mh_model.k_bin_expans))

    # NOTE: this is the default master problem
    def create_master_problem(self, USE_MULTI_CUTS=False, solver="gurobi"):
        """
        Create Bender's Master problem (compact form).

        Requires:
        - `MultiHorizonPgim` instance with:
            - MH tree (`create_ancestors_struct()`).
            - MH parameters (`get_pgim_params_per_scenario()`).

        """
        model = pyo.ConcreteModel(name="Master problem")

        # ----------------------SETS
        model.n_strgc = pyo.RangeSet(
            0, self.mh_model.s_strategic_nodes - 1
        )  # Set strategic nodes (multi-Horizon formulation)
        model.n_op = pyo.RangeSet(
            1, self.mh_model.s_operational_nodes
        )  # Set operational nodes (multi-Horizon formulation)
        model.numSubProb = pyo.RangeSet(
            0, self.mh_model.s_sample_size * self.mh_model.s_operational_nodes - 1
        )  # Set of subproblems
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)
        if self.mh_model.USE_BIN_EXPANS:
            model.bin_expans_var = pyo.RangeSet(0, self.mh_model.k_bin_expans)  # Set of binary expansion variables

        # ----------------------VARIABLES (x)
        # (INVESTEMENT)
        if self.mh_model.USE_BIN_EXPANS:
            model.z_new_lines_bin = pyo.Var(model.branch, model.n_strgc, model.bin_expans_var, within=pyo.Binary)
        else:
            model.z_new_lines = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeIntegers)

        model.z_new_capacity = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)
        model.z_capacity_total = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)

        # Sub-problem cost approximation (cost-to-go)
        if USE_MULTI_CUTS:
            model.a = pyo.Var(model.numSubProb, within=pyo.Reals)
        else:
            model.a = pyo.Var(within=pyo.Reals)

        # Create an empty set of cuts (indexed by postive integers)
        model.CUTS = pyo.Set(within=pyo.PositiveIntegers, ordered=True)

        # Define parameters from sub-problem duals
        model.sb_current_master_solution_dual_z_capacity_total = pyo.Param(
            model.branch, model.numSubProb, model.CUTS, default=0.0, mutable=True
        )  # --> \rho dual
        # Define parameters from sub-problem cost
        if USE_MULTI_CUTS:
            model.sb_current_objective_value = pyo.Param(model.numSubProb, model.CUTS, default=0.0, mutable=True)
        else:
            model.sb_current_objective_value = pyo.Param(model.CUTS, default=0.0, mutable=True)

        # Define parameters from fixed master solution (x variables) at previous iteration
        model.x_fixed_z_capacity_total = pyo.Param(
            model.branch, model.n_strgc, model.CUTS, default=0.0, mutable=True
        )  # --> x_fixed

        # ----------------------CONTRAINTS

        # Cumulative new capacity per strategic node
        def rule_aggregate_new_capacity(model, branch, n_strgc):
            if n_strgc == 0:
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)])
                ]

                return (
                    model.z_capacity_total[branch, n_strgc] == existing_capacity + model.z_new_capacity[branch, n_strgc]
                )

            else:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]

                previous_periods = [
                    self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)] for n_strgc in ancestor_nodes
                ]
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )

                return (
                    model.z_capacity_total[branch, n_strgc]
                    == existing_capacity
                    + sum(model.z_new_capacity[branch, node] for node in ancestor_nodes)
                    + model.z_new_capacity[branch, n_strgc]
                )

        model.c_rule_aggregate_new_capacity = pyo.Constraint(
            model.branch, model.n_strgc, rule=rule_aggregate_new_capacity
        )

        # Max number of cables per branch
        def rule_max_cables_per_branch(model, branch, n_strgc):
            if self.mh_model.USE_BIN_EXPANS:
                return (
                    self._integer_representation_z_new_lines_bin(model, branch, n_strgc)
                    <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                    ]["max_number_cables"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                        "parameter_value",
                    ]
                )
            else:
                return (
                    model.z_new_lines[branch, n_strgc]
                    <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                    ]["max_number_cables"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                        "parameter_value",
                    ]
                )

        model.c_rule_max_cables_per_branch = pyo.Constraint(
            model.branch, model.n_strgc, rule=rule_max_cables_per_branch
        )

        # MAX NEW CAPACITY PER CABLE (line)
        if self.mh_model.USE_FIXED_CAP_LINES:

            def rule_max_capacity_per_cable(model, branch, n_strgc):
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_new_capacity[branch, n_strgc] == self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                    ]["max_new_branch_capacity"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                        "parameter_value",
                    ] * self._integer_representation_z_new_lines_bin(
                        model, branch, n_strgc
                    )
                else:
                    return (
                        model.z_new_capacity[branch, n_strgc]
                        == self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                            "parameter_value",
                        ]
                        * model.z_new_lines[branch, n_strgc]
                    )

            model.c_rule_max_capacity_per_cable = pyo.Constraint(
                model.branch, model.n_strgc, rule=rule_max_capacity_per_cable
            )

        else:

            def rule_max_capacity_per_cable(model, branch, n_strgc):
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_new_capacity[branch, n_strgc] <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                    ]["max_new_branch_capacity"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                        "parameter_value",
                    ] * self._integer_representation_z_new_lines_bin(
                        model, branch, n_strgc
                    )
                else:
                    return (
                        model.z_new_capacity[branch, n_strgc]
                        <= self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                            "parameter_value",
                        ]
                        * model.z_new_lines[branch, n_strgc]
                    )

            model.c_rule_max_capacity_per_cable = pyo.Constraint(
                model.branch, model.n_strgc, rule=rule_max_capacity_per_cable
            )

        # ONE LAST CONTRAINT: Initialize the LB
        if USE_MULTI_CUTS:

            def rule_multiCuts(model, sb_i):
                return model.a[sb_i] >= self.INIT_LB

            model.c_rule_cuts = pyo.Constraint(model.numSubProb, rule=rule_multiCuts)

        else:
            model.c_rule_cuts = pyo.Constraint(expr=model.a >= self.INIT_LB)

        # CUTS DEFINITION: Cuts are generated on-the-fly, so no rules are necessary.
        model.Cut_Defn = pyo.ConstraintList()

        # Objective Function
        def obj_expression(model):
            all_nodes_expr = 0
            # Loop trhough the strategic nodes.
            for node in model.n_strgc:
                node_expr = 0
                # Investment Costs (IC): Loop trhough the branches.
                for branch in model.branch:
                    if self.mh_model.USE_BIN_EXPANS:
                        node_expr += self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                        ]["IC_new_cables_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                            "parameter_value",
                        ] * self._integer_representation_z_new_lines_bin(
                            model, branch, node
                        )
                    else:
                        node_expr += (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_cables_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ]
                            * model.z_new_lines[branch, node]
                        )
                    node_expr += (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                        ]["IC_new_capacity_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                            "parameter_value",
                        ]
                        * model.z_new_capacity[branch, node]
                    )

                node_expr *= self.mh_model.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

                all_nodes_expr += node_expr

            if USE_MULTI_CUTS:
                all_nodes_expr += sum(model.a[sb_i] for sb_i in model.numSubProb)
            else:
                all_nodes_expr += model.a
            return all_nodes_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        # Create a solver for the Master
        mstr_solver = pyo.SolverFactory(solver)

        return model, mstr_solver

    def create_sub_problem(self, n_op, t):
        """Create Bender's Master sub-problems (compact form).

        Requires:
        - `MultiHorizonPgim` instance with:
            - MH tree (`create_ancestors_struct()`).
            - MH parameters (`get_pgim_params_per_scenario()`).
        """
        model = pyo.ConcreteModel(name=f"Sub-problem_n_op_{n_op}_t_{t}")  # This is just an LP

        # SETS
        model.n_elctr = pyo.Set(
            initialize=self.mh_model.pgim_case.ref_pgim_model.s_node
        )  # Set of electrical (physical) nodes

        model.n_strgc = pyo.RangeSet(
            0, self.mh_model.s_strategic_nodes - 1
        )  # Set strategic nodes (multi-Horizon formulation)
        model.n_op = pyo.RangeSet(
            1, self.mh_model.s_operational_nodes
        )  # Set operational nodes (multi-Horizon formulation)

        model.gen = pyo.RangeSet(0, self.mh_model.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.mh_model.s_loads - 1)  # Set of loads
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)  # Set of branches

        # Add dual infomration to the "model".
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # VARIABLES (y)
        model.z_generation = pyo.Var(model.gen, within=pyo.NonNegativeReals)
        model.z_load_shed = pyo.Var(model.ld, within=pyo.NonNegativeReals)
        model.z_flow_12 = pyo.Var(model.branch, within=pyo.NonNegativeReals)
        model.z_flow_21 = pyo.Var(model.branch, within=pyo.NonNegativeReals)

        # Define "replicas" of the relevant Master problem variables (x)
        model.z_capacity_total = pyo.Var(model.branch, within=pyo.NonNegativeReals)

        # Define parameters to fix the relevant Master variables
        model.z_capacity_total_fixed = pyo.Param(model.branch, within=pyo.NonNegativeReals, mutable=True)

        # CONTRAINTS

        # Nodal power balance
        def rule_nodal_power_balance(model):
            # Scann all nodes and add corresponding variables to the node depedning if this has a gen, a load, or is connected to a branch
            temp_pwr_expr = 0

            # Find branches that are node_from from the current node
            branches_from = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_from"] == n_elctr
            ]

            if len(branches_from) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_21[branch]
                    for branch in branches_from.index.to_list()
                ) - sum(model.z_flow_12[branch] for branch in branches_from.index.to_list())

            # Find branches that are node_to from the current node
            branches_to = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_to"] == n_elctr
            ]

            if len(branches_to) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_12[branch]
                    for branch in branches_to.index.to_list()
                ) - sum(model.z_flow_21[branch] for branch in branches_to.index.to_list())

            # Check where generators exists
            if (
                n_elctr
                in pd.Series(self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list()).unique().tolist()
            ):
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                    self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                ]:
                    temp_pwr_expr += model.z_generation[n_electr_index]

            # Check where consumers exists, subtract demand and allow for load shedding
            if n_elctr in pd.Series(self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist():
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                    self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                ]:
                    temp_pwr_expr += (
                        model.z_load_shed[n_electr_index]
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                        ]["max_load"].loc[
                            (n_electr_index, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t),
                            "parameter_value",
                        ]
                    )
            return temp_pwr_expr == 0

        for n_elctr in model.n_elctr:
            model.add_component(
                f"Constraint_node_{n_elctr}_power_balance_at_{n_op}_operational_node_time_{t}",
                pyo.Constraint(rule=rule_nodal_power_balance),
            )

        # Load shedding upper bound
        def rule_load_shedding_limit(model):
            # Check where consumers exists to limit the corresponding load shedding variable.
            max_load = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
            ]["max_load"].loc[
                (n_electr_index, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t),
                "parameter_value",
            ]

            return model.z_load_shed[n_electr_index] <= max_load

        for n_elctr in model.n_elctr:
            if n_elctr in pd.Series(self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist():
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                    self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                ]:
                    model.add_component(
                        f"Constraint_node_{n_elctr}_consumer_{n_electr_index}_load_shedding_upper_bound_at_{n_op}_operational_node_time_{t}",
                        pyo.Constraint(rule=rule_load_shedding_limit),
                    )

        def rule_gen_upper_bound(model):

            max_gen_output = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
            ]["max_gen_power"].loc[
                (n_electr_index, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t),
                "parameter_value",
            ]
            return model.z_generation[n_electr_index] <= max_gen_output

        for n_elctr in model.n_elctr:
            if (
                n_elctr
                in pd.Series(self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list()).unique().tolist()
            ):
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                    self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                ]:
                    model.add_component(
                        f"Constraint_node_{n_elctr}_gen_{n_electr_index}_upper_bound_at_{n_op}_operational_node_time_{t}",
                        pyo.Constraint(rule=rule_gen_upper_bound),
                    )

        # Branch power flow limits 12
        def rule_branch_max_flow12(model, branch):
            return (
                model.z_flow_12[branch] <= model.z_capacity_total[branch]
            )  # it gets only the component of x corresponding to the specified indices.

        model.c_rule_branch_max_flow12 = pyo.Constraint(model.branch, rule=rule_branch_max_flow12)

        # Branch power flow limits 21
        def rule_branch_max_flow21(model, branch):
            return (
                model.z_flow_21[branch] <= model.z_capacity_total[branch]
            )  # it gets only the component of x corresponding to the specified indices.

        model.c_rule_branch_max_flow21 = pyo.Constraint(model.branch, rule=rule_branch_max_flow21)

        # ONE LAST CONTRAINT: Fix the Master variables (x) (associated dual variables)
        def rule_fix_master_var_z_capacity_total(model, b):
            return (
                model.z_capacity_total[b] == model.z_capacity_total_fixed[b]
            )  # I get the infromation for all x components, and use only the ones i need above.

        model.c_rule_fix_master_var_z_capacity_total = pyo.Constraint(
            model.branch, rule=rule_fix_master_var_z_capacity_total
        )

        # Objective Function
        def obj_expression(model):
            node_t_expr = 0
            # Operation Costs (OC) per time step:
            for gen in (
                self.mh_model.params_per_scenario[
                    self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                ]["OC_gen_coefs"]
                .index.get_level_values("gen")
                .unique()
                .to_list()
            ):

                node_t_expr += (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                    ]["OC_gen_coefs"].loc[
                        (gen, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t), "parameter_value"
                    ]
                    * model.z_generation[gen]
                )

            for ld in (
                self.mh_model.params_per_scenario[
                    self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                ]["OC_load_shed_coefs"]
                .index.get_level_values("load")
                .unique()
                .to_list()
            ):  # for ld in model.ld: could also work

                node_t_expr += (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                    ]["OC_load_shed_coefs"].loc[
                        (ld, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t), "parameter_value"
                    ]
                    * model.z_load_shed[ld]
                )

            node_t_expr *= self.mh_model.dict_operational_nodes_probabilities["n_op_" + str(n_op)]

            return node_t_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        return model

    def create_single_sub_problem(self):
        """Create Bender's non-decomposed (single) sub-problem (compact form).

        Requires:
        - `MultiHorizonPgim` instance with:
            - MH tree (`create_ancestors_struct()`).
            - MH parameters (`get_pgim_params_per_scenario()`).
        """
        model = pyo.ConcreteModel(name="Sub-problem")  # This is just an LP

        # SETS
        model.t = pyo.RangeSet(
            0, self.mh_model.s_sample_size - 1
        )  # Set of operational timesteps with duration T (Horizon)
        model.n_elctr = pyo.Set(
            initialize=self.mh_model.pgim_case.ref_pgim_model.s_node
        )  # Set of electrical (physical) nodes
        model.n_strgc = pyo.RangeSet(
            0, self.mh_model.s_strategic_nodes - 1
        )  # Set strategic nodes (multi-Horizon formulation)
        model.n_op = pyo.RangeSet(
            1, self.mh_model.s_operational_nodes
        )  # Set operational nodes (multi-Horizon formulation)

        model.gen = pyo.RangeSet(0, self.mh_model.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.mh_model.s_loads - 1)  # Set of loads
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)  # Set of branches

        # Add dual infomration to the "model".
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # -------------------VARIABLES (y)
        model.z_generation = pyo.Var(model.gen, model.n_op, model.t, within=pyo.NonNegativeReals)
        model.z_load_shed = pyo.Var(model.ld, model.n_op, model.t, within=pyo.NonNegativeReals)
        model.z_flow_12 = pyo.Var(model.branch, model.n_op, model.t, within=pyo.NonNegativeReals)
        model.z_flow_21 = pyo.Var(model.branch, model.n_op, model.t, within=pyo.NonNegativeReals)

        # Define "replicas" of the relevant Master problem variables (x)
        model.z_capacity_total = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)

        # Define parameters to fix the relevant Master variables
        model.z_capacity_total_fixed = pyo.Param(model.branch, model.n_strgc, within=pyo.NonNegativeReals, mutable=True)

        # -------------------CONTRAINTS
        # Nodal power balance
        def rule_nodal_power_balance(model):
            # Scann all nodes and add corresponding variables to the node depedning if this has a gen, a load, or is connected to a branch
            temp_pwr_expr = 0

            # Find branches that are node_from from the current node
            branches_from = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_from"] == n_elctr
            ]

            if len(branches_from) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_21[branch, n_op, t]
                    for branch in branches_from.index.to_list()
                ) - sum(model.z_flow_12[branch, n_op, t] for branch in branches_from.index.to_list())

            # Find branches that are node_to from the current node
            branches_to = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_to"] == n_elctr
            ]

            if len(branches_to) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_12[branch, n_op, t]
                    for branch in branches_to.index.to_list()
                ) - sum(model.z_flow_21[branch, n_op, t] for branch in branches_to.index.to_list())

            # Check where generators exists
            if (
                n_elctr
                in pd.Series(self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list()).unique().tolist()
            ):
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                    self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                ]:
                    temp_pwr_expr += model.z_generation[n_electr_index, n_op, t]

            # Check where consumers exists, subtract demand and allow for load shedding
            if n_elctr in pd.Series(self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist():
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                    self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                ]:
                    temp_pwr_expr += (
                        model.z_load_shed[n_electr_index, n_op, t]
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
                        ]["max_load"].loc[
                            (n_electr_index, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t),
                            "parameter_value",
                        ]
                    )

            return temp_pwr_expr == 0

        for n_op in model.n_op:
            for t in model.t:
                for n_elctr in model.n_elctr:
                    model.add_component(
                        f"Constraint_node_{n_elctr}_power_balance_at_{n_op}_operational_node_time_{t}",
                        pyo.Constraint(rule=rule_nodal_power_balance),
                    )

        # Load shedding upper bound
        def rule_load_shedding_limit(model):
            # Check where consumers exists to limit the corresponding load shedding variable.
            max_load = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
            ]["max_load"].loc[
                (n_electr_index, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t),
                "parameter_value",
            ]

            return model.z_load_shed[n_electr_index, n_op, t] <= max_load

        for n_op in model.n_op:
            for t in model.t:
                for n_elctr in model.n_elctr:
                    if (
                        n_elctr
                        in pd.Series(self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist()
                    ):
                        for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                            self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                        ]:
                            model.add_component(
                                f"Constraint_node_{n_elctr}_consumer_{n_electr_index}_load_shedding_upper_bound_at_{n_op}_operational_node_time_{t}",
                                pyo.Constraint(rule=rule_load_shedding_limit),
                            )

        def rule_gen_upper_bound(model):

            max_gen_output = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(n_op)]
            ]["max_gen_power"].loc[
                (n_electr_index, self.mh_model.dict_operational_nodes_periods["n_op_" + str(n_op)], t),
                "parameter_value",
            ]
            return model.z_generation[n_electr_index, n_op, t] <= max_gen_output

        for n_op in model.n_op:
            for t in model.t:
                for n_elctr in model.n_elctr:
                    if (
                        n_elctr
                        in pd.Series(self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list())
                        .unique()
                        .tolist()
                    ):
                        for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                            self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                        ]:
                            model.add_component(
                                f"Constraint_node_{n_elctr}_gen_{n_electr_index}_upper_bound_at_{n_op}_operational_node_time_{t}",
                                pyo.Constraint(rule=rule_gen_upper_bound),
                            )

        # Branch power flow limits 12
        def rule_branch_max_flow12(model, branch, n_op, t):
            return (
                model.z_flow_12[branch, n_op, t]
                <= model.z_capacity_total[
                    branch, self.mh_model.dict_map_operational_to_strategic_node["n_op_" + str(n_op)]
                ]
            )

        model.c_rule_branch_max_flow12 = pyo.Constraint(model.branch, model.n_op, model.t, rule=rule_branch_max_flow12)

        # Branch power flow limits 21
        def rule_branch_max_flow21(model, branch, n_op, t):
            return (
                model.z_flow_21[branch, n_op, t]
                <= model.z_capacity_total[
                    branch, self.mh_model.dict_map_operational_to_strategic_node["n_op_" + str(n_op)]
                ]
            )

        model.c_rule_branch_max_flow21 = pyo.Constraint(model.branch, model.n_op, model.t, rule=rule_branch_max_flow21)

        # ONE LAST CONTRAINT: Fix the Master variables (x) (associated dual variables)
        def rule_fix_master_var_z_capacity_total(model, b, n):
            return model.z_capacity_total[b, n] == model.z_capacity_total_fixed[b, n]

        model.c_rule_fix_master_var_z_capacity_total = pyo.Constraint(
            model.branch, model.n_strgc, rule=rule_fix_master_var_z_capacity_total
        )

        # Objective Function
        def obj_expression(model):
            all_nodes_expr = 0
            # Loop trhough the strategic nodes.
            for node in model.n_op:
                node_expr = 0
                # Operation Costs (OC): Loop trhough the timestep samples.
                for t in model.t:
                    for gen in (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(node)]
                        ]["OC_gen_coefs"]
                        .index.get_level_values("gen")
                        .unique()
                        .to_list()
                    ):

                        node_expr += (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(node)]
                            ]["OC_gen_coefs"].loc[
                                (gen, self.mh_model.dict_operational_nodes_periods["n_op_" + str(node)], t),
                                "parameter_value",
                            ]
                            * model.z_generation[gen, node, t]
                        )

                    for ld in (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(node)]
                        ]["OC_load_shed_coefs"]
                        .index.get_level_values("load")
                        .unique()
                        .to_list()
                    ):

                        node_expr += (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_operational_node_to_scenario["n_op_" + str(node)]
                            ]["OC_load_shed_coefs"].loc[
                                (ld, self.mh_model.dict_operational_nodes_periods["n_op_" + str(node)], t),
                                "parameter_value",
                            ]
                            * model.z_load_shed[ld, node, t]
                        )

                node_expr *= self.mh_model.dict_operational_nodes_probabilities["n_op_" + str(node)]
                all_nodes_expr += node_expr

            return all_nodes_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        return model

    # Make a correspondence between operational node of subproblem and index in the list of subproblems.
    def assign_operational_nodes_to_subproblems(self, original_list, repeat_every):
        repeated_list = []
        for item in original_list:
            repeated_list.extend([item] * repeat_every)
        return repeated_list


# CHECKME: IMPORTANT: THIS IS VALID ONLY FOR THE _DETERMINISTIC_ CASE ---> NOT TO BE USED FOR THE 'STOCHASTIC' CASE.
class DantzigWolfeDecomp:
    """
    ----------------------------------------------------------------------------------
    IMPORTANT: THIS IS VALID ONLY FOR THE 'DETERMINISTIC' CASE
    ----------------------------------------------------------------------------------


    Setup for Dantzig–Wolfe decomposition methods, algorithm settings, and relevant optimization problems.

    This is to be used with ancestors sets that **include current node**.
    (`multi_horizon_instance.ancestors_include_current = True`)

    """

    def __init__(self, multi_horizon_instance, INIT_LB=0, INIT_UB=350000, MAX_DW_ITER=1, CONVRG_EPS=1):
        self.INIT_LB = INIT_LB
        self.INIT_UB = INIT_UB
        self.MAX_DW_ITER = MAX_DW_ITER
        self.CONVRG_EPS = CONVRG_EPS

        if not isinstance(
            multi_horizon_instance, MultiHorizonPgim
        ):  # Check if 'custom_obj' is an instance of CustomClass
            raise TypeError("Argument must be of class type 'MultiHorizonPgim'")
        else:
            self.mh_model = multi_horizon_instance

    # CHECKME: should i make this a common function (static) in mh class?
    def _integer_representation_z_new_lines_bin(self, model, branch, n_strgc):
        return sum(2**i * model.z_new_lines_bin[branch, n_strgc, i] for i in range(self.mh_model.k_bin_expans))

    # Define rule to represent integer values using binary variables
    def _integer_representation_z_new_lines_bin_req(model, branch, a_strgc):
        return sum(2**i * model.z_new_lines_bin_req[branch, a_strgc, i] for i in model.bin_expans_var)

    # NOTE: the following function is specifically designed for this class.
    def calculate_costs(self, model, solution):
        """
        This evaluates objective function value for a given solution.

        """
        # Define rule to represent integer values using binary variables
        # CHECKME: This could be defined in a better way, outside
        def _integer_representation_z_new_lines_bin(branch, n_strgc):
            return sum(2**i * solution["z_new_lines_bin"][branch, n_strgc, i] for i in model.bin_expans_var)

        IC_per_node = []
        OC_per_node = []
        # Loop trhough the strategic nodes.
        for node in model.n_strgc:
            current_strgc_node = "n_strgc_" + str(node)
            node_IC_expr = 0
            # Investment Costs (IC): Loop trhough the branches.
            for branch in model.branch:
                if self.mh_model.USE_FIXED_CAP_LINES:
                    if self.mh_model.USE_BIN_EXPANS:
                        node_IC_expr += self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["IC_new_cables_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods[current_strgc_node]), "parameter_value"
                        ] * sum(
                            solution["w_new_FEP"][node, j] * _integer_representation_z_new_lines_bin(branch, node)
                            for j in model.J_columns
                        )
                        node_IC_expr += self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["IC_new_capacity_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods[current_strgc_node]), "parameter_value"
                        ] * sum(
                            solution["w_new_FEP"][node, j]
                            * _integer_representation_z_new_lines_bin(branch, node)
                            * self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["max_new_branch_capacity"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ]
                            for j in model.J_columns
                        )
                    else:
                        node_IC_expr += self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["IC_new_cables_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods[current_strgc_node]), "parameter_value"
                        ] * sum(
                            solution["w_new_FEP"][node, j] * solution["z_new_lines"][branch, node]
                            for j in model.J_columns
                        )
                        node_IC_expr += self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["IC_new_capacity_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods[current_strgc_node]), "parameter_value"
                        ] * sum(
                            solution["w_new_FEP"][node, j]
                            * solution["z_new_lines"][branch, node]
                            * self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["max_new_branch_capacity"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ]
                            for j in model.J_columns
                        )
                else:

                    if self.mh_model.USE_BIN_EXPANS:
                        node_IC_expr += self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                        ]["IC_new_cables_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                            "parameter_value",
                        ] * sum(
                            solution["w_new_FEP"][node, j] * _integer_representation_z_new_lines_bin(branch, node)
                            for j in model.J_columns
                        )
                    else:
                        node_IC_expr += self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["IC_new_cables_coefs"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods[current_strgc_node]), "parameter_value"
                        ] * sum(
                            solution["w_new_FEP"][node, j] * solution["z_new_lines"][branch, node]
                            for j in model.J_columns
                        )

                    node_IC_expr += self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                    ]["IC_new_capacity_coefs"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods[current_strgc_node]), "parameter_value"
                    ] * sum(
                        solution["w_new_FEP"][node, j] * solution["z_new_capacity"][branch, node]
                        for j in model.J_columns
                    )

            node_IC_expr *= self.mh_model.dict_strategic_nodes_probabilities[current_strgc_node]

            IC_per_node.append(node_IC_expr)

            node_OC_expr = 0
            # Operation Costs (OC): Loop trhough the timestep samples.
            for t in model.t:
                for gen in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                    ]["OC_gen_coefs"]
                    .index.get_level_values("gen")
                    .unique()
                    .to_list()
                ):

                    node_OC_expr += self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                    ]["OC_gen_coefs"].loc[
                        (gen, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t), "parameter_value"
                    ] * sum(
                        solution["w_new_FEP"][node, j] * model.FEP_data_opt_gen_op_plan[node, t, gen, j].value
                        for j in model.J_columns
                    )

                for ld in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                    ]["OC_load_shed_coefs"]
                    .index.get_level_values("load")
                    .unique()
                    .to_list()
                ):

                    node_OC_expr += self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                    ]["OC_load_shed_coefs"].loc[
                        (ld, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t), "parameter_value"
                    ] * sum(
                        solution["w_new_FEP"][node, j] * model.FEP_data_opt_ld_shed_plan[node, t, ld, j].value
                        for j in model.J_columns
                    )

            node_OC_expr *= self.mh_model.dict_strategic_nodes_probabilities[current_strgc_node]

            OC_per_node.append(node_OC_expr)

        return IC_per_node, OC_per_node

    def create_RRMP(self, columns_indexes=[0]):
        model = pyo.ConcreteModel(name="Relaxed-Restricted-Master-Problem")

        # SETS
        model.n_strgc = pyo.RangeSet(
            0, self.mh_model.s_strategic_nodes - 1
        )  # Set strategic nodes (multi-Horizon formulation)
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)  # Set of branches
        if self.mh_model.USE_BIN_EXPANS:
            model.bin_expans_var = pyo.RangeSet(0, self.mh_model.k_bin_expans)  # Set of binary expansion variables

        # Create an empty set of columns (indexed by postive integers)
        model.J_columns = pyo.Set(initialize=columns_indexes, within=pyo.NonNegativeIntegers, ordered=True)

        model.t = pyo.RangeSet(
            0, self.mh_model.s_sample_size - 1
        )  # Set of operational timesteps with duration T (Horizon)
        model.gen = pyo.RangeSet(0, self.mh_model.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.mh_model.s_loads - 1)  # Set of loads

        # Add dual infomration to the "model".
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # VARIABLES (x)
        if self.mh_model.USE_BIN_EXPANS:
            # CHECKME: should those be binaries or continous in the relaxed problem?
            # NOTE: If they are binaries, this is not "relaxed". If they are continous, the binary expansion does not hold.
            model.z_new_lines_bin = pyo.Var(
                model.branch, model.n_strgc, model.bin_expans_var, within=pyo.NonNegativeReals
            )
        else:
            model.z_new_lines = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)  # "grant"

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.z_new_capacity = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)  # "grant"

        # Feasible Expansion Plan (FEP) vars ("choice")
        model.w_new_FEP = pyo.Var(model.n_strgc, model.J_columns, within=pyo.NonNegativeReals)

        # Define parameters from "generated" columns
        model.FEP_data_opt_gen_op_plan = pyo.Param(
            model.n_strgc, model.t, model.gen, model.J_columns, default=0.0, mutable=True
        )
        model.FEP_data_opt_ld_shed_plan = pyo.Param(
            model.n_strgc, model.t, model.ld, model.J_columns, default=0.0, mutable=True
        )

        model.FEP_data_new_lines_req = pyo.Param(
            model.branch, model.n_strgc, model.n_strgc, model.J_columns, default=0.0, mutable=True
        )

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.FEP_data_new_capacity_req = pyo.Param(
                model.branch, model.n_strgc, model.n_strgc, model.J_columns, default=0.0, mutable=True
            )

        # CONSTRAINTS

        # A "grant" of new lines at a node should comply with the selected new lines "requests" from the history of that node
        if self.mh_model.USE_BIN_EXPANS:

            def rule_nodal_grants_meet_accepted_history_requests_lines(model):
                return (
                    sum(
                        model.FEP_data_new_lines_req[b, a_strgc, n_strgc, i, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_lines_bin[b, a_strgc, i]
                )

            for n_strgc in model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in model.branch:
                        for i in model.bin_expans_var:
                            model.add_component(
                                f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_bin_{i}",
                                pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_lines),
                            )

        else:

            def rule_nodal_grants_meet_accepted_history_requests_lines(model):
                return (
                    sum(
                        model.FEP_data_new_lines_req[b, a_strgc, n_strgc, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_lines[b, a_strgc]
                )

            for n_strgc in model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in model.branch:
                        model.add_component(
                            f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}",
                            pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_lines),
                        )

        if not self.mh_model.USE_FIXED_CAP_LINES:
            # A "grant" of new capacity at a node should comply with the selected new capacity "requests" from the history of that node
            def rule_nodal_grants_meet_accepted_history_requests_capacity(model):

                return (
                    sum(
                        model.FEP_data_new_capacity_req[b, a_strgc, n_strgc, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_capacity[b, a_strgc]
                )

            for n_strgc in model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in model.branch:
                        model.add_component(
                            f"Constraint_nodal_grants_meet_accepted_history_requests_capacity_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}",
                            pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_capacity),
                        )

        # Convexity contraints for FEPs
        def rule_FEP_convexity(model, n_strgc):
            if len(model.J_columns) > 0:
                return sum(model.w_new_FEP[n_strgc, j] for j in model.J_columns) == 1
                # return sum(model.w_new_FEP[n_strgc, j] for j in model.J_columns) <= 1
            else:
                return pyo.Constraint.Skip

        model.c_rule_FEP_convexity = pyo.Constraint(model.n_strgc, rule=rule_FEP_convexity)

        # Objective Function
        def obj_expression(model):
            all_nodes_expr = 0
            # Loop trhough the strategic nodes.
            for node in model.n_strgc:
                node_IC_expr = 0
                # Investment Costs (IC): Loop trhough the branches.
                for branch in model.branch:
                    if self.mh_model.USE_FIXED_CAP_LINES:
                        if self.mh_model.USE_BIN_EXPANS:
                            node_IC_expr += self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_cables_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ] * self._integer_representation_z_new_lines_bin(
                                model, branch, node
                            )
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_capacity_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * self._integer_representation_z_new_lines_bin(model, branch, node)
                                * self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                                ]["max_new_branch_capacity"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                                    "parameter_value",
                                ]
                            )
                        else:
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_cables_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                            )
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_capacity_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                                * self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["max_new_branch_capacity"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                            )

                    else:

                        if self.mh_model.USE_BIN_EXPANS:
                            node_IC_expr += self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_cables_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ] * self._integer_representation_z_new_lines_bin(
                                model, branch, node
                            )
                        else:
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_cables_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                            )
                        node_IC_expr += (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_capacity_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ]
                            * model.z_new_capacity[branch, node]
                        )

                node_IC_expr *= self.mh_model.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

                all_nodes_expr += node_IC_expr

                node_OC_expr = 0
                # Operation Costs (OC): Loop trhough the timestep samples.
                for t in model.t:
                    for j in model.J_columns:
                        for gen in (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["OC_gen_coefs"]
                            .index.get_level_values("gen")
                            .unique()
                            .to_list()
                        ):

                            node_OC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["OC_gen_coefs"].loc[
                                    (gen, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t),
                                    "parameter_value",
                                ]
                                * model.FEP_data_opt_gen_op_plan[node, t, gen, j]
                                * model.w_new_FEP[node, j]
                            )

                        for ld in (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["OC_load_shed_coefs"]
                            .index.get_level_values("load")
                            .unique()
                            .to_list()
                        ):

                            node_OC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["OC_load_shed_coefs"].loc[
                                    (ld, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t),
                                    "parameter_value",
                                ]
                                * model.FEP_data_opt_ld_shed_plan[node, t, ld, j]
                                * model.w_new_FEP[node, j]
                            )

                node_OC_expr *= self.mh_model.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

                all_nodes_expr += node_OC_expr

            return all_nodes_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        self.RRMP_model = model

        return self
        # return model

    def create_initial_RRMP(self):
        """_This creates the basic RRMP template before any columns are added_

        Returns:
            _self.RRMP_: _The RRMP problem_
        """
        model = pyo.ConcreteModel(name="Relaxed-Restricted-Master-Problem")

        # SETS
        model.n_strgc = pyo.RangeSet(
            0, self.mh_model.s_strategic_nodes - 1
        )  # Set strategic nodes (multi-Horizon formulation)
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)  # Set of branches
        if self.mh_model.USE_BIN_EXPANS:
            model.bin_expans_var = pyo.RangeSet(0, self.mh_model.k_bin_expans)  # Set of binary expansion variables

        # Create an empty set of columns (indexed by postive integers)
        model.J_columns = pyo.Set(within=pyo.NonNegativeIntegers, ordered=True)

        model.t = pyo.RangeSet(
            0, self.mh_model.s_sample_size - 1
        )  # Set of operational timesteps with duration T (Horizon)
        model.gen = pyo.RangeSet(0, self.mh_model.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.mh_model.s_loads - 1)  # Set of loads

        # Add dual infomration to the "model".
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # VARIABLES (x)
        if self.mh_model.USE_BIN_EXPANS:
            # CHECKME: should those be binaries or continous in the relaxed problem?
            # NOTE: If they are binaries, this is not "relaxed". If they are continous, the binary expansion does not hold.
            model.z_new_lines_bin = pyo.Var(
                model.branch, model.n_strgc, model.bin_expans_var, within=pyo.NonNegativeReals
            )
        else:
            model.z_new_lines = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)  # "grant"

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.z_new_capacity = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)  # "grant"

        # Feasible Expansion Plan (FEP) vars ("choice")
        model.w_new_FEP = pyo.Var(model.n_strgc, model.J_columns, within=pyo.NonNegativeReals)

        # Define parameters from "generated" columns
        model.FEP_data_opt_gen_op_plan = pyo.Param(
            model.n_strgc, model.t, model.gen, model.J_columns, default=0.0, mutable=True
        )
        model.FEP_data_opt_ld_shed_plan = pyo.Param(
            model.n_strgc, model.t, model.ld, model.J_columns, default=0.0, mutable=True
        )

        model.FEP_data_new_lines_req = pyo.Param(
            model.branch, model.n_strgc, model.n_strgc, model.J_columns, default=0.0, mutable=True
        )

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.FEP_data_new_capacity_req = pyo.Param(
                model.branch, model.n_strgc, model.n_strgc, model.J_columns, default=0.0, mutable=True
            )

        # Objective Function
        def obj_expression(model):
            all_nodes_expr = 0
            # Loop trhough the strategic nodes.
            for node in model.n_strgc:
                node_IC_expr = 0
                # Investment Costs (IC): Loop trhough the branches.
                for branch in model.branch:
                    if self.mh_model.USE_FIXED_CAP_LINES:
                        if self.mh_model.USE_BIN_EXPANS:
                            node_IC_expr += self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_cables_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ] * self._integer_representation_z_new_lines_bin(
                                model, branch, node
                            )
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_capacity_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * self._integer_representation_z_new_lines_bin(model, branch, node)
                                * self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["max_new_branch_capacity"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                            )
                        else:
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_cables_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                            )
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_capacity_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                                * self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["max_new_branch_capacity"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                            )

                    else:

                        if self.mh_model.USE_BIN_EXPANS:
                            node_IC_expr += self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_cables_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ] * self._integer_representation_z_new_lines_bin(
                                model, branch, node
                            )
                        else:
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_cables_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                            )
                        node_IC_expr += (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_capacity_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ]
                            * model.z_new_capacity[branch, node]
                        )

                node_IC_expr *= self.mh_model.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

                all_nodes_expr += node_IC_expr

            return all_nodes_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        self.RRMP_model = model

        return self

    def update_RRMP(self, current_DW_iter):
        """_This updates the RRMP pyomo model, adding the new variables and associated constraint and objective terms_"""
        # ------------------------CLEAN THE CONSTRAINTS DEFINED OVER THE PREVIOUS COLUMN SET
        if current_DW_iter > 0:
            # A "grant" of new lines at a node should comply with the selected new lines "requests" from the history of that node
            if self.mh_model.USE_BIN_EXPANS:

                for n_strgc in self.RRMP_model.n_strgc:
                    current_strgc_node = "n_strgc_" + str(n_strgc)
                    ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                    for a_strgc in ancestor_nodes:
                        for b in self.RRMP_model.branch:
                            for i in self.RRMP_model.bin_expans_var:
                                self.RRMP_model.del_component(
                                    self.RRMP_model.find_component(
                                        f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_bin_{i}_iter_{current_DW_iter-1}"
                                    )
                                )

            else:

                for n_strgc in self.RRMP_model.n_strgc:
                    current_strgc_node = "n_strgc_" + str(n_strgc)
                    ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                    for a_strgc in ancestor_nodes:
                        for b in self.RRMP_model.branch:
                            self.RRMP_model.del_component(
                                self.RRMP_model.find_component(
                                    f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_iter_{current_DW_iter-1}"
                                )
                            )

            if not self.mh_model.USE_FIXED_CAP_LINES:
                # A "grant" of new capacity at a node should comply with the selected new capacity "requests" from the history of that node
                for n_strgc in self.RRMP_model.n_strgc:
                    current_strgc_node = "n_strgc_" + str(n_strgc)
                    ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                    for a_strgc in ancestor_nodes:
                        for b in self.RRMP_model.branch:
                            self.RRMP_model.del_component(
                                self.RRMP_model.find_component(
                                    f"Constraint_nodal_grants_meet_accepted_history_requests_capacity_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_iter_{current_DW_iter-1}"
                                )
                            )

            # Convexity contraints for FEPs
            for n_strgc in self.RRMP_model.n_strgc:
                self.RRMP_model.del_component(
                    self.RRMP_model.find_component(
                        f"Constraint_FEP_convexity_at_node_{n_strgc}_iter_{current_DW_iter-1}"
                    )
                )

        # ------------------------ADD TERMS RELATED TO THE NEW VARIABLE (GENERATED COLUMN) IN THE OBJECTIVE
        for node in self.RRMP_model.n_strgc:
            node_OC_expr = 0
            # Operation Costs (OC): Loop trhough the timestep samples.
            for t in self.RRMP_model.t:
                for gen in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                    ]["OC_gen_coefs"]
                    .index.get_level_values("gen")
                    .unique()
                    .to_list()
                ):

                    node_OC_expr += (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                        ]["OC_gen_coefs"].loc[
                            (gen, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t),
                            "parameter_value",
                        ]
                        * self.RRMP_model.FEP_data_opt_gen_op_plan[node, t, gen, current_DW_iter]
                        * self.RRMP_model.w_new_FEP[node, current_DW_iter]
                    )

                for ld in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                    ]["OC_load_shed_coefs"]
                    .index.get_level_values("load")
                    .unique()
                    .to_list()
                ):

                    node_OC_expr += (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                        ]["OC_load_shed_coefs"].loc[
                            (ld, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t),
                            "parameter_value",
                        ]
                        * self.RRMP_model.FEP_data_opt_ld_shed_plan[node, t, ld, current_DW_iter]
                        * self.RRMP_model.w_new_FEP[node, current_DW_iter]
                    )

            node_OC_expr *= self.mh_model.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

            self.RRMP_model.obj += node_OC_expr

        # ------------------------RE-DEFINE THE CONSTRAINTS OVER THE NEW COLUMN SET (INCLUDING THE NEW DEFINED VARIABLE)
        # A "grant" of new lines at a node should comply with the selected new lines "requests" from the history of that node
        if self.mh_model.USE_BIN_EXPANS:

            def rule_nodal_grants_meet_accepted_history_requests_lines(model):
                return (
                    sum(
                        model.FEP_data_new_lines_req[b, a_strgc, n_strgc, i, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_lines_bin[b, a_strgc, i]
                )

            for n_strgc in self.RRMP_model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in self.RRMP_model.branch:
                        for i in self.RRMP_model.bin_expans_var:
                            self.RRMP_model.add_component(
                                f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_bin_{i}_iter_{current_DW_iter}",
                                pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_lines),
                            )

        else:

            def rule_nodal_grants_meet_accepted_history_requests_lines(model):
                return (
                    sum(
                        model.FEP_data_new_lines_req[b, a_strgc, n_strgc, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_lines[b, a_strgc]
                )

            for n_strgc in self.RRMP_model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in self.RRMP_model.branch:
                        self.RRMP_model.add_component(
                            f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_iter_{current_DW_iter}",
                            pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_lines),
                        )

        if not self.mh_model.USE_FIXED_CAP_LINES:
            # A "grant" of new capacity at a node should comply with the selected new capacity "requests" from the history of that node
            def rule_nodal_grants_meet_accepted_history_requests_capacity(model):

                return (
                    sum(
                        model.FEP_data_new_capacity_req[b, a_strgc, n_strgc, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_capacity[b, a_strgc]
                )

            for n_strgc in self.RRMP_model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in self.RRMP_model.branch:
                        self.RRMP_model.add_component(
                            f"Constraint_nodal_grants_meet_accepted_history_requests_capacity_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_iter_{current_DW_iter}",
                            pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_capacity),
                        )

        # Convexity contraints for FEPs
        def rule_FEP_convexity(model):
            return sum(model.w_new_FEP[n_strgc, j] for j in model.J_columns) == 1

        for n_strgc in self.RRMP_model.n_strgc:
            self.RRMP_model.add_component(
                f"Constraint_FEP_convexity_at_node_{n_strgc}_iter_{current_DW_iter}",
                pyo.Constraint(rule=rule_FEP_convexity),
            )

        return self

    # NOTE: this is maybe too verbose. Can i integrate it with create_DW_RRMP?
    def create_IRMP(self, columns_indexes=[0]):
        model = pyo.ConcreteModel(name="Relaxed-Restricted-Master-Problem")

        # SETS
        model.n_strgc = pyo.RangeSet(
            0, self.mh_model.s_strategic_nodes - 1
        )  # Set strategic nodes (multi-Horizon formulation)
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)  # Set of branches
        if self.mh_model.USE_BIN_EXPANS:
            model.bin_expans_var = pyo.RangeSet(0, self.mh_model.k_bin_expans)  # Set of binary expansion variables

        # Create an empty set of columns (indexed by postive integers)
        model.J_columns = pyo.Set(initialize=columns_indexes, within=pyo.NonNegativeIntegers, ordered=True)

        model.t = pyo.RangeSet(
            0, self.mh_model.s_sample_size - 1
        )  # Set of operational timesteps with duration T (Horizon)
        model.gen = pyo.RangeSet(0, self.mh_model.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.mh_model.s_loads - 1)  # Set of loads

        # Add dual infomration to the "model".
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # VARIABLES (x)
        if self.mh_model.USE_BIN_EXPANS:
            model.z_new_lines_bin = pyo.Var(model.branch, model.n_strgc, model.bin_expans_var, within=pyo.Binary)
        else:
            model.z_new_lines = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeIntegers)  # "grant"

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.z_new_capacity = pyo.Var(model.branch, model.n_strgc, within=pyo.NonNegativeReals)  # "grant"

        # Feasible Expansion Plan (FEP) vars ("choice")
        model.w_new_FEP = pyo.Var(model.n_strgc, model.J_columns, within=pyo.Binary)

        # Define parameters from "generated" columns
        model.FEP_data_opt_gen_op_plan = pyo.Param(
            model.n_strgc, model.t, model.gen, model.J_columns, default=0.0, mutable=True
        )
        model.FEP_data_opt_ld_shed_plan = pyo.Param(
            model.n_strgc, model.t, model.ld, model.J_columns, default=0.0, mutable=True
        )

        model.FEP_data_new_lines_req = pyo.Param(
            model.branch, model.n_strgc, model.n_strgc, model.J_columns, default=0.0, mutable=True
        )

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.FEP_data_new_capacity_req = pyo.Param(
                model.branch, model.n_strgc, model.n_strgc, model.J_columns, default=0.0, mutable=True
            )

        # CONSTRAINTS

        # A "grant" of new lines at a node should comply with the selected new lines "requests" from the history of that node
        if self.mh_model.USE_BIN_EXPANS:

            def rule_nodal_grants_meet_accepted_history_requests_lines(model):
                return (
                    sum(
                        model.FEP_data_new_lines_req[b, a_strgc, n_strgc, i, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_lines_bin[b, a_strgc, i]
                )

            for n_strgc in model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in model.branch:
                        for i in model.bin_expans_var:
                            model.add_component(
                                f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}_bin_{i}",
                                pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_lines),
                            )

        else:

            def rule_nodal_grants_meet_accepted_history_requests_lines(model):
                return (
                    sum(
                        model.FEP_data_new_lines_req[b, a_strgc, n_strgc, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_lines[b, a_strgc]
                )

            for n_strgc in model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in model.branch:
                        model.add_component(
                            f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}",
                            pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_lines),
                        )

        if not self.mh_model.USE_FIXED_CAP_LINES:
            # A "grant" of new capacity at a node should comply with the selected new capacity "requests" from the history of that node
            def rule_nodal_grants_meet_accepted_history_requests_capacity(model):

                return (
                    sum(
                        model.FEP_data_new_capacity_req[b, a_strgc, n_strgc, j] * model.w_new_FEP[n_strgc, j]
                        for j in model.J_columns
                    )
                    <= model.z_new_capacity[b, a_strgc]
                )

            for n_strgc in model.n_strgc:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
                for a_strgc in ancestor_nodes:
                    for b in model.branch:
                        model.add_component(
                            f"Constraint_nodal_grants_meet_accepted_history_requests_capacity_at_node_{n_strgc}_ancestor_{a_strgc}_branch_{b}",
                            pyo.Constraint(rule=rule_nodal_grants_meet_accepted_history_requests_capacity),
                        )

        # Convexity contraints for FEPs
        def rule_FEP_convexity(model, n_strgc):
            return sum(model.w_new_FEP[n_strgc, j] for j in model.J_columns) == 1
            # if len(model.J_columns) > 0:
            #     return sum(model.w_new_FEP[n_strgc, j] for j in model.J_columns) == 1
            #     # return sum(model.w_new_FEP[n_strgc, j] for j in model.J_columns) <= 1
            # else:
            #     return pyo.Constraint.Skip

        model.c_rule_FEP_convexity = pyo.Constraint(model.n_strgc, rule=rule_FEP_convexity)

        # Objective Function
        def obj_expression(model):
            all_nodes_expr = 0
            # Loop trhough the strategic nodes.
            for node in model.n_strgc:
                node_IC_expr = 0
                # Investment Costs (IC): Loop trhough the branches.
                for branch in model.branch:
                    if self.mh_model.USE_FIXED_CAP_LINES:
                        if self.mh_model.USE_BIN_EXPANS:
                            node_IC_expr += self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_cables_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ] * self._integer_representation_z_new_lines_bin(
                                model, branch, node
                            )
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_capacity_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * self._integer_representation_z_new_lines_bin(model, branch, node)
                                * self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                                ]["max_new_branch_capacity"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                                    "parameter_value",
                                ]
                            )
                        else:
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_cables_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                            )
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_capacity_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                                * self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["max_new_branch_capacity"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                            )

                    else:

                        if self.mh_model.USE_BIN_EXPANS:
                            node_IC_expr += self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_cables_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ] * self._integer_representation_z_new_lines_bin(
                                model, branch, node
                            )
                        else:
                            node_IC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["IC_new_cables_coefs"].loc[
                                    (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                    "parameter_value",
                                ]
                                * model.z_new_lines[branch, node]
                            )
                        node_IC_expr += (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["IC_new_capacity_coefs"].loc[
                                (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)]),
                                "parameter_value",
                            ]
                            * model.z_new_capacity[branch, node]
                        )

                node_IC_expr *= self.mh_model.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

                all_nodes_expr += node_IC_expr

                node_OC_expr = 0
                # Operation Costs (OC): Loop trhough the timestep samples.
                for t in model.t:
                    for j in model.J_columns:
                        for gen in (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["OC_gen_coefs"]
                            .index.get_level_values("gen")
                            .unique()
                            .to_list()
                        ):

                            node_OC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["OC_gen_coefs"].loc[
                                    (gen, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t),
                                    "parameter_value",
                                ]
                                * model.FEP_data_opt_gen_op_plan[node, t, gen, j]
                                * model.w_new_FEP[node, j]
                            )

                        for ld in (
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                            ]["OC_load_shed_coefs"]
                            .index.get_level_values("load")
                            .unique()
                            .to_list()
                        ):

                            node_OC_expr += (
                                self.mh_model.params_per_scenario[
                                    self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(node)]
                                ]["OC_load_shed_coefs"].loc[
                                    (ld, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(node)], t),
                                    "parameter_value",
                                ]
                                * model.FEP_data_opt_ld_shed_plan[node, t, ld, j]
                                * model.w_new_FEP[node, j]
                            )

                node_OC_expr *= self.mh_model.dict_strategic_nodes_probabilities["n_strgc_" + str(node)]

                all_nodes_expr += node_OC_expr

            return all_nodes_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        self.IRMP_model = model

        return self

    def create_sub_problem(self, n_strgc):
        model = pyo.ConcreteModel(name=f"Sub-problem_n_strgc_{n_strgc}")  # This is a MILP
        current_strgc_node = "n_strgc_" + str(n_strgc)

        ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]
        previous_periods = [self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)] for a in ancestor_nodes][
            :-1
        ]

        # SETS
        model.a_strgc = pyo.Set(
            initialize=ancestor_nodes
        )  # Set of ancestor strategic nodes (specific for each supb-problem)

        model.t = pyo.RangeSet(
            0, self.mh_model.s_sample_size - 1
        )  # Set of operational timesteps with duration T (Horizon)
        model.n_elctr = pyo.Set(
            initialize=self.mh_model.pgim_case.ref_pgim_model.s_node
        )  # Set of electrical (physical) nodes
        model.gen = pyo.RangeSet(0, self.mh_model.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.mh_model.s_loads - 1)  # Set of loads
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)  # Set of branches
        if self.mh_model.USE_BIN_EXPANS:
            model.bin_expans_var = pyo.RangeSet(0, self.mh_model.k_bin_expans)  # Set of binary expansion variables

        # VARIABLES (y)
        model.z_generation = pyo.Var(model.gen, model.t, within=pyo.NonNegativeReals)
        model.z_load_shed = pyo.Var(model.ld, model.t, within=pyo.NonNegativeReals)
        model.z_flow_12 = pyo.Var(model.branch, model.t, within=pyo.NonNegativeReals)
        model.z_flow_21 = pyo.Var(model.branch, model.t, within=pyo.NonNegativeReals)

        # Define "replicas" of the RRMP investment variables (x)
        if self.mh_model.USE_BIN_EXPANS:
            model.z_new_lines_bin_req = pyo.Var(
                model.branch, model.a_strgc, model.bin_expans_var, within=pyo.Binary
            )  # "request=grant"
        else:
            model.z_new_lines_req = pyo.Var(
                model.branch, model.a_strgc, within=pyo.NonNegativeIntegers
            )  # "request=grant"

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.z_new_capacity_req = pyo.Var(
                model.branch, model.a_strgc, within=pyo.NonNegativeReals
            )  # "request=grant"

        # Define parameters from RRMP duals
        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.dual_RRMP_capacity_requests_n_grants = pyo.Param(
                model.branch, model.a_strgc, default=0.0, mutable=True
            )  # --> \pi dual

        if self.mh_model.USE_BIN_EXPANS:
            model.dual_RRMP_lines_requests_n_grants = pyo.Param(
                model.branch, model.a_strgc, model.bin_expans_var, default=0.0, mutable=True
            )  # --> \mu dual
        else:
            model.dual_RRMP_lines_requests_n_grants = pyo.Param(
                model.branch, model.a_strgc, default=0.0, mutable=True
            )  # --> \mu dual

        model.dual_RRMP_convexity = pyo.Param(default=0.0, mutable=True)  # --> \lambda dual

        # CONTRAINTS
        # Nodal power balance
        def rule_nodal_power_balance(model):
            # Scann all nodes and add corresponding variables to the node depedning if this has a gen, a load, or is connected to a branch
            temp_pwr_expr = 0

            # Find branches that are node_from from the current node
            branches_from = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_from"] == n_elctr
            ]

            if len(branches_from) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_21[branch, t]
                    for branch in branches_from.index.to_list()
                ) - sum(model.z_flow_12[branch, t] for branch in branches_from.index.to_list())

            # Find branches that are node_to from the current node
            branches_to = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_to"] == n_elctr
            ]

            if len(branches_to) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_12[branch, t]
                    for branch in branches_to.index.to_list()
                ) - sum(model.z_flow_21[branch, t] for branch in branches_to.index.to_list())

            # Check where generators exists
            if n_elctr in self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list():
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                    self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                ]:
                    temp_pwr_expr += model.z_generation[n_electr_index, t]

            # Check where consumers exists, subtract demand and allow for load shedding
            if n_elctr in self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list():
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                    self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                ]:
                    temp_pwr_expr += (
                        model.z_load_shed[n_electr_index, t]
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["max_load"].loc[
                            (n_electr_index, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t),
                            "parameter_value",
                        ]
                    )

            return temp_pwr_expr == 0

        for t in model.t:
            for n_elctr in model.n_elctr:
                model.add_component(
                    f"Constraint_node_{n_elctr}_power_balance_at_{n_strgc}_strategic_node_time_{t}",
                    pyo.Constraint(rule=rule_nodal_power_balance),
                )

        # Load shedding upper bound
        def rule_load_shedding_limit(model):
            # Check where consumers exists to limit the corresponding load shedding variable.
            max_load = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
            ]["max_load"].loc[
                (n_electr_index, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
            ]

            return model.z_load_shed[n_electr_index, t] <= max_load

        for n_elctr in model.n_elctr:
            for t in model.t:
                if (
                    n_elctr
                    in pd.Series(self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist()
                ):
                    for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                        self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                    ]:
                        model.add_component(
                            f"Constraint_node_{n_elctr}_consumer_{n_electr_index}_load_shedding_upper_bound_at_{n_strgc}_strategic_node_time_{t}",
                            pyo.Constraint(rule=rule_load_shedding_limit),
                        )

        def rule_gen_upper_bound(model):

            max_gen_output = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
            ]["max_gen_power"].loc[
                (n_electr_index, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
            ]
            return model.z_generation[n_electr_index, t] <= max_gen_output

        for n_elctr in model.n_elctr:
            for t in model.t:
                if (
                    n_elctr
                    in pd.Series(self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list()).unique().tolist()
                ):
                    for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                        self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                    ]:
                        model.add_component(
                            f"Constraint_node_{n_elctr}_gen_{n_electr_index}_upper_bound_at_{n_strgc}_strategic_node_time_{t}",
                            pyo.Constraint(rule=rule_gen_upper_bound),
                        )

        # Branch power flow limits 12
        if self.mh_model.USE_FIXED_CAP_LINES:

            def rule_branch_max_flow12(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_flow_12[branch, t] <= existing_capacity + sum(
                        self._integer_representation_z_new_lines_bin_req(model, branch, a)
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )
                else:
                    return model.z_flow_12[branch, t] <= existing_capacity + sum(
                        model.z_new_lines_req[branch, a]
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )

            model.c_rule_branch_max_flow12 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow12)

        else:

            def rule_branch_max_flow12(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )

                return model.z_flow_12[branch, t] <= existing_capacity + sum(
                    model.z_new_capacity_req[branch, a] for a in ancestor_nodes
                )

            model.c_rule_branch_max_flow12 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow12)

        # Branch power flow limits 21
        if self.mh_model.USE_FIXED_CAP_LINES:

            def rule_branch_max_flow21(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_flow_21[branch, t] <= existing_capacity + sum(
                        self._integer_representation_z_new_lines_bin_req(model, branch, a)
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )
                else:
                    return model.z_flow_21[branch, t] <= existing_capacity + sum(
                        model.z_new_lines_req[branch, a]
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )

            model.c_rule_branch_max_flow21 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow21)

        else:

            def rule_branch_max_flow21(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )

                return model.z_flow_21[branch, t] <= existing_capacity + sum(
                    model.z_new_capacity_req[branch, a] for a in ancestor_nodes
                )

            model.c_rule_branch_max_flow21 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow21)

        # Max number of cables per branch
        def rule_max_cables_per_branch(model, branch, a_strgc):
            if self.mh_model.USE_BIN_EXPANS:
                return (
                    self._integer_representation_z_new_lines_bin_req(model, branch, a_strgc)
                    <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                    ]["max_number_cables"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                        "parameter_value",
                    ]
                )
            else:
                return (
                    model.z_new_lines_req[branch, a_strgc]
                    <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                    ]["max_number_cables"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                        "parameter_value",
                    ]
                )

        model.c_rule_max_cables_per_branch = pyo.Constraint(
            model.branch, model.a_strgc, rule=rule_max_cables_per_branch
        )

        # Max new capacity per cable
        if self.mh_model.USE_FIXED_CAP_LINES:
            ...
        else:

            def rule_max_capacity_per_cable(model, branch, a_strgc):
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_new_capacity_req[branch, a_strgc] <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                    ]["max_new_branch_capacity"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                        "parameter_value",
                    ] * self._integer_representation_z_new_lines_bin_req(
                        model, branch, a_strgc
                    )
                else:
                    return (
                        model.z_new_capacity_req[branch, a_strgc]
                        <= self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                            "parameter_value",
                        ]
                        * model.z_new_lines_req[branch, a_strgc]
                    )

            model.c_rule_max_capacity_per_cable = pyo.Constraint(
                model.branch, model.a_strgc, rule=rule_max_capacity_per_cable
            )

        # Objective Function
        def obj_expression(model):

            node_expr = 0
            # Operation Costs (OC): Loop trhough the timestep samples.
            for t in model.t:

                for gen in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                    ]["OC_gen_coefs"]
                    .index.get_level_values("gen")
                    .unique()
                    .to_list()
                ):

                    node_expr += (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["OC_gen_coefs"].loc[
                            (gen, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
                        ]
                        * model.z_generation[gen, t]
                    )

                for ld in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                    ]["OC_load_shed_coefs"]
                    .index.get_level_values("load")
                    .unique()
                    .to_list()
                ):

                    node_expr += (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["OC_load_shed_coefs"].loc[
                            (ld, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
                        ]
                        * model.z_load_shed[ld, t]
                    )

            node_expr *= self.mh_model.dict_strategic_nodes_probabilities[current_strgc_node]

            if not self.mh_model.USE_FIXED_CAP_LINES:
                node_expr -= sum(
                    model.dual_RRMP_capacity_requests_n_grants[branch, a] * model.z_new_capacity_req[branch, a]
                    for branch in model.branch
                    for a in ancestor_nodes
                )

            if self.mh_model.USE_BIN_EXPANS:
                node_expr -= sum(
                    model.dual_RRMP_lines_requests_n_grants[branch, a, i] * model.z_new_lines_bin_req[branch, a, i]
                    for branch in model.branch
                    for a in ancestor_nodes
                    for i in model.bin_expans_var
                )
            else:
                node_expr -= sum(
                    model.dual_RRMP_lines_requests_n_grants[branch, a] * model.z_new_lines_req[branch, a]
                    for branch in model.branch
                    for a in ancestor_nodes
                )

            node_expr -= model.dual_RRMP_convexity

            return node_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        return model

    # NOTE: This is too verbose.
    # CHECKME: I do not need an "initial" version. I can just use a current iter param and avoid the duals related costs in the first iter.
    def create_initial_sub_problem(self, n_strgc):
        model = pyo.ConcreteModel(name=f"Sub-problem_n_strgc_{n_strgc}")  # This is a MILP
        current_strgc_node = "n_strgc_" + str(n_strgc)
        ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]

        previous_periods = [self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)] for a in ancestor_nodes][
            :-1
        ]

        # SETS
        model.a_strgc = pyo.Set(
            initialize=ancestor_nodes
        )  # Set of ancestor strategic nodes (specific for each supb-problem)

        model.t = pyo.RangeSet(
            0, self.mh_model.s_sample_size - 1
        )  # Set of operational timesteps with duration T (Horizon)
        model.n_elctr = pyo.Set(
            initialize=self.mh_model.pgim_case.ref_pgim_model.s_node
        )  # Set of electrical (physical) nodes
        model.gen = pyo.RangeSet(0, self.mh_model.s_generators - 1)  # Set of generators
        model.ld = pyo.RangeSet(0, self.mh_model.s_loads - 1)  # Set of loads
        model.branch = pyo.RangeSet(0, self.mh_model.s_branches - 1)  # Set of branches
        if self.mh_model.USE_BIN_EXPANS:
            model.bin_expans_var = pyo.RangeSet(0, self.mh_model.k_bin_expans)  # Set of binary expansion variables

        # VARIABLES (y)
        model.z_generation = pyo.Var(model.gen, model.t, within=pyo.NonNegativeReals)
        model.z_load_shed = pyo.Var(model.ld, model.t, within=pyo.NonNegativeReals)
        model.z_flow_12 = pyo.Var(model.branch, model.t, within=pyo.NonNegativeReals)
        model.z_flow_21 = pyo.Var(model.branch, model.t, within=pyo.NonNegativeReals)

        # Define "replicas" of the RRMP investment variables (x)
        if self.mh_model.USE_BIN_EXPANS:
            model.z_new_lines_bin_req = pyo.Var(
                model.branch, model.a_strgc, model.bin_expans_var, within=pyo.Binary
            )  # "request=grant"
        else:
            model.z_new_lines_req = pyo.Var(
                model.branch, model.a_strgc, within=pyo.NonNegativeIntegers
            )  # "request=grant"

        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.z_new_capacity_req = pyo.Var(
                model.branch, model.a_strgc, within=pyo.NonNegativeReals
            )  # "request=grant"

        # Define parameters from RRMP duals
        if not self.mh_model.USE_FIXED_CAP_LINES:
            model.dual_RRMP_capacity_requests_n_grants = pyo.Param(
                model.branch, model.a_strgc, default=0.0, mutable=True
            )  # --> \pi dual

        if self.mh_model.USE_BIN_EXPANS:
            model.dual_RRMP_lines_requests_n_grants = pyo.Param(
                model.branch, model.a_strgc, model.bin_expans_var, default=0.0, mutable=True
            )  # --> \mu dual
        else:
            model.dual_RRMP_lines_requests_n_grants = pyo.Param(
                model.branch, model.a_strgc, default=0.0, mutable=True
            )  # --> \mu dual

        model.dual_RRMP_convexity = pyo.Param(default=0.0, mutable=True)  # --> \lambda dual

        # CONTRAINTS
        # Nodal power balance
        def rule_nodal_power_balance(model):
            # Scann all nodes and add corresponding variables to the node depedning if this has a gen, a load, or is connected to a branch
            temp_pwr_expr = 0

            # Find branches that are node_from from the current node
            branches_from = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_from"] == n_elctr
            ]

            if len(branches_from) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_21[branch, t]
                    for branch in branches_from.index.to_list()
                ) - sum(model.z_flow_12[branch, t] for branch in branches_from.index.to_list())

            # Find branches that are node_to from the current node
            branches_to = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                self.mh_model.pgim_case.ref_grid_data.branch["node_to"] == n_elctr
            ]

            if len(branches_to) > 0:
                temp_pwr_expr += sum(
                    (
                        1
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["branch_losses"].loc[branch, "parameter_value"]
                    )
                    * model.z_flow_12[branch, t]
                    for branch in branches_to.index.to_list()
                ) - sum(model.z_flow_21[branch, t] for branch in branches_to.index.to_list())

            # Check where generators exists
            if n_elctr in self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list():
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                    self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                ]:
                    temp_pwr_expr += model.z_generation[n_electr_index, t]

            # Check where consumers exists, subtract demand and allow for load shedding
            if n_elctr in self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list():
                for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                    self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                ]:
                    temp_pwr_expr += (
                        model.z_load_shed[n_electr_index, t]
                        - self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["max_load"].loc[
                            (n_electr_index, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t),
                            "parameter_value",
                        ]
                    )

            return temp_pwr_expr == 0

        for t in model.t:
            for n_elctr in model.n_elctr:
                model.add_component(
                    f"Constraint_node_{n_elctr}_power_balance_at_{n_strgc}_strategic_node_time_{t}",
                    pyo.Constraint(rule=rule_nodal_power_balance),
                )

        # Load shedding upper bound
        def rule_load_shedding_limit(model):
            # Check where consumers exists to limit the corresponding load shedding variable.
            max_load = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
            ]["max_load"].loc[
                (n_electr_index, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
            ]

            return model.z_load_shed[n_electr_index, t] <= max_load

        for n_elctr in model.n_elctr:
            for t in model.t:
                if (
                    n_elctr
                    in pd.Series(self.mh_model.pgim_case.ref_grid_data.consumer["node"].to_list()).unique().tolist()
                ):
                    for n_electr_index in self.mh_model.pgim_case.ref_grid_data.consumer.index[
                        self.mh_model.pgim_case.ref_grid_data.consumer["node"] == n_elctr
                    ]:
                        model.add_component(
                            f"Constraint_node_{n_elctr}_consumer_{n_electr_index}_load_shedding_upper_bound_at_{n_strgc}_strategic_node_time_{t}",
                            pyo.Constraint(rule=rule_load_shedding_limit),
                        )

        def rule_gen_upper_bound(model):

            max_gen_output = self.mh_model.params_per_scenario[
                self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
            ]["max_gen_power"].loc[
                (n_electr_index, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
            ]
            return model.z_generation[n_electr_index, t] <= max_gen_output

        for n_elctr in model.n_elctr:
            for t in model.t:
                if (
                    n_elctr
                    in pd.Series(self.mh_model.pgim_case.ref_grid_data.generator["node"].to_list()).unique().tolist()
                ):
                    for n_electr_index in self.mh_model.pgim_case.ref_grid_data.generator.index[
                        self.mh_model.pgim_case.ref_grid_data.generator["node"] == n_elctr
                    ]:
                        model.add_component(
                            f"Constraint_node_{n_elctr}_gen_{n_electr_index}_upper_bound_at_{n_strgc}_strategic_node_time_{t}",
                            pyo.Constraint(rule=rule_gen_upper_bound),
                        )

        # Branch power flow limits 12
        if self.mh_model.USE_FIXED_CAP_LINES:

            def rule_branch_max_flow12(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_flow_12[branch, t] <= existing_capacity + sum(
                        self._integer_representation_z_new_lines_bin_req(model, branch, a)
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )
                else:
                    return model.z_flow_12[branch, t] <= existing_capacity + sum(
                        model.z_new_lines_req[branch, a]
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )

            model.c_rule_branch_max_flow12 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow12)
        else:

            def rule_branch_max_flow12(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )

                return model.z_flow_12[branch, t] <= existing_capacity + sum(
                    model.z_new_capacity_req[branch, a] for a in ancestor_nodes
                )

            model.c_rule_branch_max_flow12 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow12)

        # Branch power flow limits 21
        if self.mh_model.USE_FIXED_CAP_LINES:

            def rule_branch_max_flow21(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_flow_21[branch, t] <= existing_capacity + sum(
                        self._integer_representation_z_new_lines_bin_req(model, branch, a)
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )
                else:
                    return model.z_flow_21[branch, t] <= existing_capacity + sum(
                        model.z_new_lines_req[branch, a]
                        * self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a)]), "parameter_value"
                        ]
                        for a in ancestor_nodes
                    )

            model.c_rule_branch_max_flow21 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow21)
        else:

            def rule_branch_max_flow21(model, branch, t):
                existing_capacity = self.mh_model.pgim_case.ref_grid_data.branch.loc[
                    branch, "capacity_" + str(self.mh_model.dict_strategic_nodes_periods[current_strgc_node])
                ] + sum(
                    self.mh_model.pgim_case.ref_grid_data.branch.loc[branch, "capacity_" + str(pr)]
                    for pr in previous_periods
                )

                return model.z_flow_21[branch, t] <= existing_capacity + sum(
                    model.z_new_capacity_req[branch, a] for a in ancestor_nodes
                )

            model.c_rule_branch_max_flow21 = pyo.Constraint(model.branch, model.t, rule=rule_branch_max_flow21)

        # Max number of cables per branch
        def rule_max_cables_per_branch(model, branch, a_strgc):
            if self.mh_model.USE_BIN_EXPANS:
                return (
                    self._integer_representation_z_new_lines_bin_req(model, branch, a_strgc)
                    <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                    ]["max_number_cables"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                        "parameter_value",
                    ]
                )
            else:
                return (
                    model.z_new_lines_req[branch, a_strgc]
                    <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                    ]["max_number_cables"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                        "parameter_value",
                    ]
                )

        model.c_rule_max_cables_per_branch = pyo.Constraint(
            model.branch, model.a_strgc, rule=rule_max_cables_per_branch
        )

        # Max new capacity per cable
        if self.mh_model.USE_FIXED_CAP_LINES:
            ...
        else:

            def rule_max_capacity_per_cable(model, branch, a_strgc):
                if self.mh_model.USE_BIN_EXPANS:
                    return model.z_new_capacity_req[branch, a_strgc] <= self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                    ]["max_new_branch_capacity"].loc[
                        (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                        "parameter_value",
                    ] * self._integer_representation_z_new_lines_bin_req(
                        model, branch, a_strgc
                    )
                else:
                    return (
                        model.z_new_capacity_req[branch, a_strgc]
                        <= self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(a_strgc)]
                        ]["max_new_branch_capacity"].loc[
                            (branch, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(a_strgc)]),
                            "parameter_value",
                        ]
                        * model.z_new_lines_req[branch, a_strgc]
                    )

            model.c_rule_max_capacity_per_cable = pyo.Constraint(
                model.branch, model.a_strgc, rule=rule_max_capacity_per_cable
            )

        # Objective Function
        def obj_expression(model):

            node_expr = 0
            # Operation Costs (OC): Loop trhough the timestep samples.
            for t in model.t:

                for gen in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                    ]["OC_gen_coefs"]
                    .index.get_level_values("gen")
                    .unique()
                    .to_list()
                ):

                    node_expr += (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["OC_gen_coefs"].loc[
                            (gen, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
                        ]
                        * model.z_generation[gen, t]
                    )

                for ld in (
                    self.mh_model.params_per_scenario[
                        self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                    ]["OC_load_shed_coefs"]
                    .index.get_level_values("load")
                    .unique()
                    .to_list()
                ):

                    node_expr += (
                        self.mh_model.params_per_scenario[
                            self.mh_model.dict_map_strategic_node_to_scenario[current_strgc_node]
                        ]["OC_load_shed_coefs"].loc[
                            (ld, self.mh_model.dict_strategic_nodes_periods[current_strgc_node], t), "parameter_value"
                        ]
                        * model.z_load_shed[ld, t]
                    )

            node_expr *= self.mh_model.dict_strategic_nodes_probabilities[current_strgc_node]

            return node_expr

        model.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        return model

    def assign_RRMP_duals_to_sub_problems(self, n, current_DW_iter, first_FEP=False):
        if first_FEP is True:
            # Assign SP parameters based on a first-guess FEP (trivial flat solution)
            sub_problem = self.create_initial_sub_problem(n)
            for b in sub_problem.branch:
                for a in sub_problem.a_strgc:
                    if self.mh_model.USE_BIN_EXPANS:
                        for i in sub_problem.bin_expans_var:
                            sub_problem.z_new_lines_bin_req[b, a, i].set_value(1)
                    else:
                        if self.mh_model.USE_FIXED_CAP_LINES:
                            sub_problem.z_new_lines_req[b, a].set_value(5)  # NOTE: THIS WORKS WITH FIXED CAPACITY LINES
                        else:
                            sub_problem.z_new_lines_req[b, a].set_value(
                                0
                            )  # NOTE: THIS WORKS WITHOUT FIXED CAPACITY LINES

                        # sub_problem.z_new_lines_req[b,a].set_value(self.mh_model.params_per_scenario[self.mh_model.dict_map_strategic_node_to_scenario['n_strgc_' + str(n)]]['max_number_cables'].loc[(b, self.mh_model.dict_strategic_nodes_periods['n_strgc_' + str(n)]), 'parameter_value'])

                    if not self.mh_model.USE_FIXED_CAP_LINES:
                        # sub_problem.z_new_capacity_req[b,a].set_value(0)
                        sub_problem.z_new_capacity_req[b, a].set_value(
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n)]
                            ]["max_new_branch_capacity"].loc[
                                (b, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n)]), "parameter_value"
                            ]
                        )

        else:
            # Requires the attribute self.RRMP_model, whcih comes after running self.create_DW_RRMP()
            # Assign SP parameters based on current iteration FEP
            sub_problem = self.create_sub_problem(n)
            for b in sub_problem.branch:
                for a in sub_problem.a_strgc:
                    if not self.mh_model.USE_FIXED_CAP_LINES:
                        constraint_name = f"Constraint_nodal_grants_meet_accepted_history_requests_capacity_at_node_{n}_ancestor_{a}_branch_{b}_iter_{current_DW_iter}"
                        sub_problem.dual_RRMP_capacity_requests_n_grants[b, a].set_value(
                            self.RRMP_model.dual[self.RRMP_model.find_component(constraint_name)]
                        )

                    if self.mh_model.USE_BIN_EXPANS:
                        for i in sub_problem.bin_expans_var:
                            constraint_name = f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n}_ancestor_{a}_branch_{b}_bin_{i}_iter_{current_DW_iter}"
                            sub_problem.dual_RRMP_lines_requests_n_grants[b, a, i].set_value(
                                self.RRMP_model.dual[self.RRMP_model.find_component(constraint_name)]
                            )
                    else:
                        constraint_name = f"Constraint_nodal_grants_meet_accepted_history_requests_lines_at_node_{n}_ancestor_{a}_branch_{b}_iter_{current_DW_iter}"
                        sub_problem.dual_RRMP_lines_requests_n_grants[b, a].set_value(
                            self.RRMP_model.dual[self.RRMP_model.find_component(constraint_name)]
                        )

            constraint_name = f"Constraint_FEP_convexity_at_node_{n}_iter_{current_DW_iter}"
            sub_problem.dual_RRMP_convexity.set_value(
                self.RRMP_model.dual[self.RRMP_model.find_component(constraint_name)]
            )

        return sub_problem

    # CHECKME: convergence of DW is highly sensitive on the initial columns selection.
    def assign_initial_y_hat_to_RRMP(self, current_DW_iter, y_hat):
        # Assign y_hat_init to the corresponding parameters of the RRMP
        for n_strgc in self.RRMP_model.n_strgc:
            for t in self.RRMP_model.t:
                for g in self.RRMP_model.gen:

                    self.RRMP_model.FEP_data_opt_gen_op_plan[n_strgc, t, g, current_DW_iter].set_value(
                        y_hat[n_strgc]["z_generation"][g, t]
                    )

                for my_ld in self.RRMP_model.ld:

                    self.RRMP_model.FEP_data_opt_ld_shed_plan[n_strgc, t, my_ld, current_DW_iter].set_value(
                        y_hat[n_strgc]["z_load_shed"][my_ld, t]
                    )

            # x_hat_init is already defined (from initial FEP but i defined it to get the duals)
            # Assign x_hat_init to the corresponding parameters of the RRMP
            for b in self.RRMP_model.branch:
                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]

                for a in ancestor_nodes:
                    if self.mh_model.USE_BIN_EXPANS:
                        for i in self.RRMP_model.bin_expans_var:
                            self.RRMP_model.FEP_data_new_lines_req[b, a, n_strgc, i, current_DW_iter].set_value(1)
                    else:
                        self.RRMP_model.FEP_data_new_lines_req[b, a, n_strgc, current_DW_iter].set_value(
                            5
                        )  # THIS WORKS WITHOUT AND WITH FIXED CAPACITY LINES
                        # self.RRMP_model.FEP_data_new_lines_req[b,a,n_strgc,current_DW_iter].set_value(0) # THIS WORKS WITHOUT FIXED CAPACITY LINES
                        # self.RRMP_model.FEP_data_new_lines_req[b,a,n_strgc,current_DW_iter].set_value(self.mh_model.params_per_scenario[self.mh_model.dict_map_strategic_node_to_scenario['n_strgc_' + str(n_strgc)]]['max_number_cables'].loc[(b, self.mh_model.dict_strategic_nodes_periods['n_strgc_' + str(n_strgc)]), 'parameter_value'])

                    if not self.mh_model.USE_FIXED_CAP_LINES:
                        # self.RRMP_model.FEP_data_new_capacity_req[b,a,n_strgc,current_DW_iter].set_value(0)
                        self.RRMP_model.FEP_data_new_capacity_req[b, a, n_strgc, current_DW_iter].set_value(
                            self.mh_model.params_per_scenario[
                                self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                            ]["max_new_branch_capacity"].loc[
                                (b, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                                "parameter_value",
                            ]
                        )

    # NOTE: DW loop needs to update collumns (add one column) before calling this method.
    def assign_columns_data_to_RMP(self, current_DW_iter, current_column_data_from_subproblems):

        # Current CG iteration which builts an additional column in the RRMP
        for n_strgc in self.RRMP_model.n_strgc:

            # Assign x_hat from sub-problems solution to the corresponding parameters of the RRMP
            for b in self.RRMP_model.branch:

                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]

                for a in ancestor_nodes:
                    if not self.mh_model.USE_FIXED_CAP_LINES:
                        self.RRMP_model.FEP_data_new_capacity_req[b, a, n_strgc, current_DW_iter].set_value(
                            current_column_data_from_subproblems[n_strgc]["z_new_capacity_req"][b, a]
                        )

                    if self.mh_model.USE_BIN_EXPANS:
                        for i in self.RRMP_model.bin_expans_var:
                            self.RRMP_model.FEP_data_new_lines_req[b, a, n_strgc, i, current_DW_iter].set_value(
                                current_column_data_from_subproblems[n_strgc]["z_new_lines_bin_req"][b, a, i]
                            )
                    else:
                        self.RRMP_model.FEP_data_new_lines_req[b, a, n_strgc, current_DW_iter].set_value(
                            current_column_data_from_subproblems[n_strgc]["z_new_lines_req"][b, a]
                        )

            # Assign y_hat from sub-problems solution to the corresponding parameters of the RRMP
            for t in self.mh_model.non_decomposed_model.t:
                for g in self.mh_model.non_decomposed_model.gen:
                    self.RRMP_model.FEP_data_opt_gen_op_plan[n_strgc, t, g, current_DW_iter].set_value(
                        current_column_data_from_subproblems[n_strgc]["z_generation"][g, t]
                    )

                for my_ld in self.mh_model.non_decomposed_model.ld:
                    self.RRMP_model.FEP_data_opt_ld_shed_plan[n_strgc, t, my_ld, current_DW_iter].set_value(
                        current_column_data_from_subproblems[n_strgc]["z_load_shed"][my_ld, t]
                    )

        return self

    def assign_columns_data_to_RMP_OLD_implementation(
        self,
        current_DW_iter,
        past_column_data_from_subproblems,
        current_column_data_from_subproblems,
        FINAL_ITER_FLAG=False,
    ):
        # Previous DW-CG iterations that have built past columns in the RRMP
        if FINAL_ITER_FLAG:
            _temp_model = self.IRMP_model
        else:
            _temp_model = self.RRMP_model

        for column_iter in range(current_DW_iter):
            for n_strgc in _temp_model.n_strgc:

                # Assign x_hat from sub-problems solution to the corresponding parameters of the RMP
                for b in _temp_model.branch:

                    current_strgc_node = "n_strgc_" + str(n_strgc)
                    ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]

                    for a in ancestor_nodes:

                        # Intial solution (user defined)
                        if column_iter == 0:

                            if not self.mh_model.USE_FIXED_CAP_LINES:
                                # _temp_model.FEP_data_new_capacity_req[b,a,n_strgc,column_iter].set_value(0)
                                _temp_model.FEP_data_new_capacity_req[b, a, n_strgc, column_iter].set_value(
                                    self.mh_model.params_per_scenario[
                                        self.mh_model.dict_map_strategic_node_to_scenario["n_strgc_" + str(n_strgc)]
                                    ]["max_new_branch_capacity"].loc[
                                        (b, self.mh_model.dict_strategic_nodes_periods["n_strgc_" + str(n_strgc)]),
                                        "parameter_value",
                                    ]
                                )

                            if self.mh_model.USE_BIN_EXPANS:
                                for i in _temp_model.bin_expans_var:
                                    _temp_model.FEP_data_new_lines_req[b, a, n_strgc, i, column_iter].set_value(1)
                            else:
                                # _temp_model.FEP_data_new_lines_req[b,a,n_strgc,column_iter].set_value(self.mh_model.params_per_scenario[self.mh_model.dict_map_strategic_node_to_scenario['n_strgc_' + str(n_strgc)]]['max_number_cables'].loc[(b, self.mh_model.dict_strategic_nodes_periods['n_strgc_' + str(n_strgc)]), 'parameter_value'])
                                if self.mh_model.USE_FIXED_CAP_LINES:
                                    _temp_model.FEP_data_new_lines_req[b, a, n_strgc, column_iter].set_value(
                                        5
                                    )  # THIS WORKS WITH FIXED CAPACITY LINES
                                else:
                                    _temp_model.FEP_data_new_lines_req[b, a, n_strgc, column_iter].set_value(
                                        0
                                    )  # THIS WORKS WITHOUT FIXED CAPACITY LINES

                        else:

                            if not self.mh_model.USE_FIXED_CAP_LINES:
                                _temp_model.FEP_data_new_capacity_req[b, a, n_strgc, column_iter].set_value(
                                    past_column_data_from_subproblems[column_iter][n_strgc]["z_new_capacity_req"][b, a]
                                )

                            if self.mh_model.USE_BIN_EXPANS:
                                _temp_model.FEP_data_new_lines_req[b, a, n_strgc, i, column_iter].set_value(
                                    past_column_data_from_subproblems[column_iter][n_strgc]["z_new_lines_bin_req"][
                                        b, a, i
                                    ]
                                )
                            else:
                                _temp_model.FEP_data_new_lines_req[b, a, n_strgc, column_iter].set_value(
                                    past_column_data_from_subproblems[column_iter][n_strgc]["z_new_lines_req"][b, a]
                                )

                # Assign y_hat from sub-problems solution to the corresponding parameters of the RRMP
                for t in self.mh_model.non_decomposed_model.t:
                    for g in self.mh_model.non_decomposed_model.gen:

                        _temp_model.FEP_data_opt_gen_op_plan[n_strgc, t, g, column_iter].set_value(
                            past_column_data_from_subproblems[column_iter][n_strgc]["z_generation"][g, t]
                        )

                    for my_ld in self.mh_model.non_decomposed_model.ld:

                        _temp_model.FEP_data_opt_ld_shed_plan[n_strgc, t, my_ld, column_iter].set_value(
                            past_column_data_from_subproblems[column_iter][n_strgc]["z_load_shed"][my_ld, t]
                        )

        # Current CG iteration which builts an additional column in the RRMP
        for n_strgc in _temp_model.n_strgc:

            # Assign x_hat from sub-problems solution to the corresponding parameters of the RRMP
            for b in _temp_model.branch:

                current_strgc_node = "n_strgc_" + str(n_strgc)
                ancestor_nodes = self.mh_model.dict_ancestors_set[current_strgc_node]

                for a in ancestor_nodes:
                    if not self.mh_model.USE_FIXED_CAP_LINES:
                        _temp_model.FEP_data_new_capacity_req[b, a, n_strgc, current_DW_iter].set_value(
                            current_column_data_from_subproblems[n_strgc]["z_new_capacity_req"][b, a]
                        )

                    if self.mh_model.USE_BIN_EXPANS:
                        for i in _temp_model.bin_expans_var:
                            _temp_model.FEP_data_new_lines_req[b, a, n_strgc, i, current_DW_iter].set_value(
                                current_column_data_from_subproblems[n_strgc]["z_new_lines_bin_req"][b, a, i]
                            )
                    else:
                        _temp_model.FEP_data_new_lines_req[b, a, n_strgc, current_DW_iter].set_value(
                            current_column_data_from_subproblems[n_strgc]["z_new_lines_req"][b, a]
                        )

            # Assign y_hat from sub-problems solution to the corresponding parameters of the RRMP
            for t in self.mh_model.non_decomposed_model.t:
                for g in self.mh_model.non_decomposed_model.gen:
                    _temp_model.FEP_data_opt_gen_op_plan[n_strgc, t, g, current_DW_iter].set_value(
                        current_column_data_from_subproblems[n_strgc]["z_generation"][g, t]
                    )

                for my_ld in self.mh_model.non_decomposed_model.ld:
                    _temp_model.FEP_data_opt_ld_shed_plan[n_strgc, t, my_ld, current_DW_iter].set_value(
                        current_column_data_from_subproblems[n_strgc]["z_load_shed"][my_ld, t]
                    )

        if FINAL_ITER_FLAG:
            self.IRMP_model = _temp_model
        else:
            self.RRMP_model = _temp_model

        return self


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
