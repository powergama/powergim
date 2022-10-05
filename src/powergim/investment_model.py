"""
Module for power grid investment analyses

"""

import copy

import mpisppy.scenario_tree as scenario_tree
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .utils import annuityfactor


class const:
    baseMVA = 100  # MVA


def _slice_to_list(component_slice):
    """expand slice to list of components"""
    mylist = [v for k, v in component_slice.expanded_items()]
    return mylist


class SipModel(pyo.ConcreteModel):
    """
    Power Grid Investment Module - Stochastic Investment Problem
    """

    _NUMERICAL_THRESHOLD_ZERO = 1e-6
    _HOURS_PER_YEAR = 8760

    def __init__(self, grid_data, parameter_data, M_const=1000):
        """Create Abstract Pyomo model for PowerGIM

        Parameters
        ----------
        M_const : int
            large constant
        """

        super().__init__()
        self.M_const = M_const
        self.grid_data = grid_data
        self.nodetypes = parameter_data["nodetype"]
        self.branchtypes = parameter_data["branchtype"]
        self.gentypes = parameter_data["gentype"]
        self.parameters = parameter_data["parameters"]
        self.investment_years = parameter_data["parameters"]["investment_years"]
        self.finance_years = parameter_data["parameters"]["finance_years"]
        self.finance_interest_rate = parameter_data["parameters"]["finance_interest_rate"]
        self.operation_maintenance_rate = parameter_data["parameters"]["operation_maintenance_rate"]
        self.load_shed_penalty = parameter_data["parameters"]["load_shed_penalty"]
        self.CO2_price = parameter_data["parameters"]["CO2_price"]
        self.MAX_BRANCH_NEW_NUM = 5  # Fixme: get from parameter input
        self.MAX_GEN_NEW_CAPACITY = 10000  # Fixme: get from parameter input

        # sample factor scales from sample to annual value
        self.sample_factor = pd.Series(
            index=self.grid_data.profiles.index, data=self._HOURS_PER_YEAR / self.grid_data.profiles.shape[0]
        )
        if "frequency" in self.grid_data.profiles:
            # a column describing weight for each time-step
            self.sample_factor = self.grid_data.profiles["frequency"]

        self.create_sets()
        self.create_variables()
        self.create_objective()
        self.create_constraints()

    def create_sets(self):
        """Initialise Pyomo model sets"""
        timerange = range(self.grid_data.profiles.shape[0])
        self.s_node = pyo.Set(initialize=self.grid_data.node["id"].tolist())
        self.s_branch = pyo.Set(initialize=self.grid_data.branch.index.tolist())
        self.s_gen = pyo.Set(initialize=self.grid_data.generator.index.tolist())
        self.s_load = pyo.Set(initialize=self.grid_data.consumer.index.tolist())
        self.s_time = pyo.Set(initialize=timerange)
        self.s_area = pyo.Set(initialize=list(self.grid_data.node["area"].unique()))
        self.s_period = pyo.Set(initialize=self.investment_years)
        self.s_branchtype = pyo.Set(initialize=self.branchtypes.keys())
        self.s_branch_cost_element = pyo.Set(initialize=["B", "Bd", "Bdp", "CLp", "CL", "CSp", "CS"])
        self.s_nodetype = pyo.Set()
        self.s_gentype = pyo.Set()
        self.s_node_cost_element = pyo.Set(initialize=["L", "S"])

    def create_variables(self):
        # NOTE: bounds vs constraints
        # In general, variable limits are given by constraints.
        # But bounds are needed when linearising proximal terms in stochastic
        # optimisation and are therefore included. Note also that Pyomo gives error if lb=ub (a bug),
        # so instead of specifying bounds lb=ub, we do it as constraint.

        # TODO: Initialize variables=0 if they may be omitted

        def bounds_branch_new_capacity(model, branch, period):
            # default max capacity is given by branch type and max number of cables
            branchtype = self.grid_data.branch.loc[branch, "type"]
            cap_branchtype = self.branchtypes[branchtype]["max_cap"]
            maxcap = self.MAX_BRANCH_NEW_NUM * cap_branchtype
            if self.grid_data.branch.loc[branch, "max_newCap"] > 0:
                maxcap = self.grid_data.branch.loc[branch, "max_newCap"]
            return (0, maxcap)

        self.v_branch_new_capacity = pyo.Var(
            self.s_branch,
            self.s_period,
            within=pyo.NonNegativeReals,
            bounds=bounds_branch_new_capacity,  # needed for proximal term linearisation in stochastic optimisation
        )

        # investment: new branch cables (needed for linearisation, see also model.cMaxNumberCables)
        def bounds_branch_new_cables(model, branch, period):
            return (0, self.MAX_BRANCH_NEW_NUM)

        self.v_branch_new_cables = pyo.Var(
            self.s_branch,
            self.s_period,
            within=pyo.NonNegativeIntegers,
            initialize=0,
            bounds=bounds_branch_new_cables,
        )

        # investment: new nodes
        self.v_new_nodes = pyo.Var(self.s_node, self.s_period, within=pyo.Binary)

        # investment: generation capacity
        def bounds_gen_new_capacity(model, gen, period):
            maxcap = self.MAX_GEN_NEW_CAPACITY
            if self.grid_data.generator.loc[gen, "p_maxNew"] > 0:
                maxcap = self.grid_data.generator.loc[gen, "p_maxNew"]
            # this does not work here, as it in some cases gives ub=0=lb -> using constraints instead
            # max_value = maxcap * self.grid_data.generator.loc[gen, f"expand_{period}"]
            return (0, maxcap)

        self.v_gen_new_capacity = pyo.Var(
            self.s_gen, self.s_period, within=pyo.NonNegativeReals, bounds=bounds_gen_new_capacity, initialize=0
        )

        # branch flows in both directions
        self.v_branch_flow12 = pyo.Var(
            self.s_branch,
            self.s_period,
            self.s_time,
            within=pyo.NonNegativeReals,
        )
        self.v_branch_flow21 = pyo.Var(
            self.s_branch,
            self.s_period,
            self.s_time,
            within=pyo.NonNegativeReals,
        )

        # generator output (bounds set by constraint)
        self.v_generation = pyo.Var(self.s_gen, self.s_period, self.s_time, within=pyo.NonNegativeReals)

        # load shedding
        def bounds_load_shed(model, consumer, period, time):
            ref = self.grid_data.consumer.loc[consumer, "demand_ref"]
            profile = self.grid_data.profiles.loc[time, ref]
            demand_avg = self.grid_data.consumer.loc[consumer, "demand_avg"]
            ub = max(0, demand_avg * profile)
            return (0, ub)

        self.v_load_shed = pyo.Var(
            self.s_load,
            self.s_period,
            self.s_time,
            domain=pyo.NonNegativeReals,
            bounds=bounds_load_shed,
        )

    def create_objective(self):
        self.v_investment_cost = pyo.Var(self.s_period, within=pyo.Reals)
        self.v_operating_cost = pyo.Var(self.s_period, within=pyo.Reals)

        def investment_cost_rule(model, period):
            """Investment cost, including lifetime O&M costs (NPV)"""
            expr = self.costInvestments(period)
            return self.v_investment_cost[period] == expr

        self.c_investment_cost = pyo.Constraint(self.s_period, rule=investment_cost_rule)

        def operating_cost_rule(model, period):
            """Operational costs: cost of gen, load shed (NPV)"""
            opcost = self.costOperation(period)
            return self.v_operating_cost[period] == opcost

        self.c_operating_costs = pyo.Constraint(self.s_period, rule=operating_cost_rule)

        def total_cost_objective_rule(model):
            investment = pyo.summation(self.v_investment_cost)
            operation = pyo.summation(self.v_operating_cost)
            return investment + operation

        self.OBJ = pyo.Objective(rule=total_cost_objective_rule, sense=pyo.minimize)

    def create_constraints(self):

        # Power flow limited by installed (existing or new) capacity
        def _branch_flow_limit(branch, period, t):
            branch_existing_capacity = 0
            branch_new_capacity = 0
            previous_periods = (p for p in self.s_period if p <= period)
            for p in previous_periods:
                branch_existing_capacity += self.grid_data.branch.loc[branch, f"capacity_{p}"]
                if self.grid_data.branch.loc[branch, f"expand_{period}"] == 1:
                    branch_new_capacity += self.v_branch_new_capacity[branch, p]
            return branch_existing_capacity + branch_new_capacity

        def rule_max_flow12(model, branch, period, t):
            expr = self.v_branch_flow12[branch, period, t] <= _branch_flow_limit(branch, period, t)
            return expr

        def rule_max_flow21(model, branch, period, t):
            expr = self.v_branch_flow21[branch, period, t] <= _branch_flow_limit(branch, period, t)
            return expr

        self.c_max_flow12 = pyo.Constraint(self.s_branch, self.s_period, self.s_time, rule=rule_max_flow12)
        self.c_max_flow21 = pyo.Constraint(self.s_branch, self.s_period, self.s_time, rule=rule_max_flow21)

        # number of new cables is limited
        def rule_max_new_cables(model, branch, period):
            max_num = self.MAX_BRANCH_NEW_NUM * self.grid_data.branch.loc[branch, f"expand_{period}"]
            expr = self.v_branch_new_cables[branch, period] <= max_num
            return expr

        self.c_max_number_cables = pyo.Constraint(self.s_branch, self.s_period, rule=rule_max_new_cables)

        # No new branch capacity without new cables
        def rule_max_new_cap(model, branch, period):
            branchtype = self.grid_data.branch.loc[branch, "type"]
            cap_branchtype = self.branchtypes[branchtype]["max_cap"]
            expr = (
                self.v_branch_new_capacity[branch, period] <= cap_branchtype * self.v_branch_new_cables[branch, period]
            )
            return expr

        self.c_max_new_branch_capacity = pyo.Constraint(self.s_branch, self.s_period, rule=rule_max_new_cap)

        # A node required at each branch endpoint
        def rule_new_nodes(model, node, period):
            num_nodes = self.grid_data.node.loc[node, "existing"]
            previous_periods = (p for p in self.s_period if p <= period)
            for p in previous_periods:
                num_nodes += self.v_new_nodes[node, p]
            connected_branches = 0
            for branch in self.s_branch:
                node_from = self.grid_data.branch.loc[branch, "node_from"]
                node_to = self.grid_data.branch.loc[branch, "node_to"]
                if node_from == node or node_to == node:
                    connected_branches += self.v_branch_new_cables[branch, period]
            expr = connected_branches <= self.M_const * num_nodes
            if (type(expr) is bool) and (expr is True):
                expr = pyo.Constraint.Skip
            return expr

        self.c_new_nodes = pyo.Constraint(self.s_node, self.s_period, rule=rule_new_nodes)

        # Limit new generator capacity
        def rule_gen_new_capacity(model, gen, period):
            maxcap = self.MAX_GEN_NEW_CAPACITY
            if self.grid_data.generator.loc[gen, "p_maxNew"] > 0:
                maxcap = self.grid_data.generator.loc[gen, "p_maxNew"]
            max_value = maxcap * self.grid_data.generator.loc[gen, f"expand_{period}"]
            return self.v_gen_new_capacity[gen, period] <= max_value

        self.c_max_new_gen_capacity = pyo.Constraint(self.s_gen, self.s_period, rule=rule_gen_new_capacity)

        # Generator output limitations
        # TODO: add option to set minimum output = timeseries for renewable,
        # i.e. disallov curtaliment (could be global parameter)
        def rule_max_gen_power(model, gen, period, t):
            cap_existing = 0
            cap_new = 0
            previous_periods = (p for p in self.s_period if p <= period)
            for p in previous_periods:
                cap_existing += self.grid_data.generator.loc[gen, f"capacity_{p}"]
                if self.grid_data.generator.loc[gen, f"expand_{p}"] == 1:
                    cap_new += self.v_gen_new_capacity[gen, p]
            cap = cap_existing + cap_new
            gentype = self.grid_data.generator.loc[gen, "type"]
            profile_ref = self.grid_data.generator.loc[gen, "inflow_ref"]
            profile_fac = self.grid_data.generator.loc[gen, "inflow_fac"]
            profile_value = self.grid_data.profiles.loc[t, profile_ref] * profile_fac
            allow_curtailment = self.gentypes[gentype]["allow_curtailment"]
            if allow_curtailment:
                expr = self.v_generation[gen, period, t] <= (profile_value * cap)
            else:
                # don't allow curtailment of generator output - output fixed by profile
                expr = self.v_generation[gen, period, t] == (profile_value * cap)
            return expr

        self.c_max_gen_power = pyo.Constraint(self.s_gen, self.s_period, self.s_time, rule=rule_max_gen_power)

        # Generator maximum average output (energy sum)
        # (relevant e.g. for hydro with storage)
        def rule_max_energy(model, gen, period):
            cap_existing = 0
            cap_new = 0
            previous_periods = (p for p in self.s_period if p <= period)
            for p in previous_periods:
                cap_existing += self.grid_data.generator.loc[gen, f"capacity_{p}"]
                cap_new += self.v_gen_new_capacity[gen, p]
            cap = cap_existing + cap_new
            max_p_avg = self.grid_data.generator.loc[gen, "pavg"]
            if max_p_avg > 0:
                # TODO: Weighted average according to sample factor
                expr = sum(self.v_generation[gen, period, t] for t in self.s_time) <= (
                    max_p_avg * cap * len(self.s_time)
                )
            else:
                expr = pyo.Constraint.Skip
            return expr

        self.c_max_energy = pyo.Constraint(self.s_gen, self.s_period, rule=rule_max_energy)

        # Emissions restriction per country/load
        # TODO: deal with situation when no emission cap has been given (-1)
        def rule_emission_cap(model, area, period):
            if self.CO2_price > 0:
                area_emission = 0
                for n in self.s_node:
                    node_area = self.grid_data.node.loc[n, "area"]
                    if node_area == area:
                        for gen in self.s_gen:
                            gen_node = self.grid_data.generator.loc[gen, "node"]
                            if gen_node == n:
                                gentype = self.grid_data.generator.loc[gen, "type"]
                                emission_rate = self.gentypes[gentype]["CO2"]
                                area_emission += sum(
                                    self.v_generation[gen, period, t] * emission_rate * self.sample_factor[t]
                                    for t in self.s_time
                                )
                area_cap = 0
                for cons in self.s_load:
                    load_node = self.grid_data.consumer.loc[cons, "node"]
                    load_area = self.grid_data.node.loc[load_node, "area"]
                    if load_area == area:
                        cons_cap = self.grid_data.consumer.loc[cons, "emission_cap"]
                        area_cap += cons_cap
                expr = area_emission <= area_cap
            else:
                expr = pyo.Constraint.Skip
            return expr

        self.c_emission_cap = pyo.Constraint(self.s_area, self.s_period, rule=rule_emission_cap)

        # Power balance in nodes : gen+demand+flow into node=0
        def rule_powerbalance(model, node, period, t):
            flow_into_node = 0
            # flow of power into node (subtrating losses)
            for branch in self.s_branch:
                node_from = self.grid_data.branch.loc[branch, "node_from"]
                node_to = self.grid_data.branch.loc[branch, "node_to"]
                if node_from == node:
                    # branch out of node
                    branchtype = self.grid_data.branch.loc[branch, "type"]
                    dist = self.grid_data.branch.loc[branch, "distance"]
                    loss_fix = self.branchtypes[branchtype]["loss_fix"]
                    loss_slope = self.branchtypes[branchtype]["loss_slope"]
                    flow_into_node -= self.v_branch_flow12[branch, period, t]
                    flow_into_node += self.v_branch_flow21[branch, period, t] * (1 - (loss_fix + loss_slope * dist))
                if node_to == node:
                    # branch into node
                    branchtype = self.grid_data.branch.loc[branch, "type"]
                    dist = self.grid_data.branch.loc[branch, "distance"]
                    loss_fix = self.branchtypes[branchtype]["loss_fix"]
                    loss_slope = self.branchtypes[branchtype]["loss_slope"]
                    flow_into_node -= self.v_branch_flow21[branch, period, t]
                    flow_into_node += self.v_branch_flow12[branch, period, t] * (1 - (loss_fix + loss_slope * dist))

            # generated power
            for gen in self.s_gen:
                node_gen = self.grid_data.generator.loc[gen, "node"]
                if node_gen == node:
                    flow_into_node += self.v_generation[gen, period, t]

            # load shedding
            for cons in self.s_load:
                node_load = self.grid_data.consumer.loc[cons, "node"]
                if node_load == node:
                    flow_into_node += self.v_load_shed[cons, period, t]

            # consumed power
            for cons in self.s_load:
                node_load = self.grid_data.consumer.loc[cons, "node"]
                if node_load == node:
                    dem_avg = self.grid_data.consumer.loc[cons, "demand_avg"]
                    dem_profile_ref = self.grid_data.consumer.loc[cons, "demand_ref"]
                    profile = self.grid_data.profiles.loc[t, dem_profile_ref]
                    flow_into_node += -dem_avg * profile

            expr = flow_into_node == 0

            if (type(expr) is bool) and (expr is True):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr

        self.c_powerbalance = pyo.Constraint(self.s_node, self.s_period, self.s_time, rule=rule_powerbalance)

    def costNode(self, node, period):
        """Expression for cost of node, investment cost no discounting"""
        n_cost = 0
        var_num = self.v_new_nodes
        is_offshore = self.grid_data.node.loc[node, "offshore"]  # 1 or 0
        nodetype = self.grid_data.node.loc[node, "type"]
        nodetype_costs = self.nodetypes[nodetype]
        scale = self.grid_data.node.loc[node, "cost_scaling"]
        n_cost += is_offshore * (nodetype_costs["S"] * var_num[node, period])
        n_cost += (1 - is_offshore) * (nodetype_costs["L"] * var_num[node, period])
        return scale * n_cost

    def costBranch(self, branch, period):
        """Expression for cost of branch, investment cost no discounting"""
        b_cost = 0

        var_num = self.v_branch_new_cables
        var_cap = self.v_branch_new_capacity
        branchtype = self.grid_data.branch.loc[branch, "type"]
        branchtype_costs = self.branchtypes[branchtype]
        distance = self.grid_data.branch.loc[branch, "distance"]
        b_cost += branchtype_costs["B"] * var_num[branch, period]
        b_cost += branchtype_costs["Bd"] * distance * var_num[branch, period]
        b_cost += branchtype_costs["Bdp"] * distance * var_cap[branch, period]

        # endpoint costs (difference onshore/offshore)
        node1 = self.grid_data.branch.loc[branch, "node_from"]
        node2 = self.grid_data.branch.loc[branch, "node_to"]
        is_offshore1 = self.grid_data.node.loc[node1, "offshore"]
        is_offshore2 = self.grid_data.node.loc[node2, "offshore"]
        for N in [is_offshore1, is_offshore2]:
            b_cost += N * (
                branchtype_costs["CS"] * var_num[branch, period] + branchtype_costs["CSp"] * var_cap[branch, period]
            )
            b_cost += (1 - N) * (
                branchtype_costs["CL"] * var_num[branch, period] + branchtype_costs["CLp"] * var_cap[branch, period]
            )
        scale = self.grid_data.branch.loc[branch, "cost_scaling"]
        return scale * b_cost

    def costGen(self, gen, period):
        """Expression for cost of generator, investment cost no discounting"""
        g_cost = 0
        var_cap = self.v_gen_new_capacity
        gentype = self.grid_data.generator.loc[gen, "type"]
        gentype_cost = self.gentypes[gentype]
        scale = self.grid_data.generator.loc[gen, "cost_scaling"]
        g_cost += gentype_cost["CX"] * var_cap[gen, period]
        return scale * g_cost

    def npvInvestment(self, period, investment, include_om=True, subtract_residual_value=True):
        """NPV of investment cost including lifetime O&M and salvage value

        Parameters
        ----------
        model : object
            Pyomo model
        stage : int
            Investment or operation stage (1 or 2)
        investment :
            cost of e.g. node, branch or gen

        O&M is computed via a parameter giving O&M costs per year relative to investment
        """
        om_factor = 0
        residual_factor = 0
        delta_years = period - self.investment_years[0]
        if subtract_residual_value:
            # Remaining value of investment at end of period considered (self.finance_years)
            # if delta_years=0, then residual value should be zero.
            # if delta_years=finance_years, then residual value factor should be 1
            residual_factor = (delta_years / self.finance_years) * (
                1 / ((1 + self.finance_interest_rate) ** (self.finance_years - delta_years))
            )
        if include_om:
            # NPV of all O&M from investment made to end of time period considered
            om_factor = self.operation_maintenance_rate * (
                annuityfactor(self.finance_interest_rate, self.finance_years)
                - annuityfactor(self.finance_interest_rate, delta_years)
            )

        # discount investments in the future (after period 0)
        # present value vs future value: pv = fv/(1+r)^n
        discount_t0 = 1 / ((1 + self.finance_interest_rate) ** (delta_years))

        investment = investment * discount_t0
        pv_cost = investment * (1 + om_factor - residual_factor)
        return pv_cost

    def costInvestments(self, period, include_om=True, subtract_residual_value=True):
        """Investment cost, including lifetime O&M costs (NPV)"""
        investment = 0
        # add branch, node and generator investment costs:
        for b in self.s_branch:
            investment += self.costBranch(b, period)
        for n in self.s_node:
            investment += self.costNode(n, period)
        for g in self.s_gen:
            investment += self.costGen(g, period)
        # add O&M costs and compute NPV:
        cost = self.npvInvestment(period, investment, include_om, subtract_residual_value)
        return cost

    def costOperation(self, period):
        """Operational costs: cost of gen, load shed (NPV)"""
        opcost = 0
        # discount_t0 = (1/((1+model.financeInterestrate)
        #    **(model.stage2TimeDelta*int(stage-1))))

        # operation cost for single year:
        opcost = 0
        for gen in self.s_gen:
            fuelcost = self.grid_data.generator.loc[gen, "fuelcost"]
            cost_profile_ref = self.grid_data.generator.loc[gen, "fuelcost_ref"]
            cost_profile = self.grid_data.profiles[cost_profile_ref]
            gentype = self.grid_data.generator.loc[gen, "type"]
            emission_rate = self.gentypes[gentype]["CO2"]
            opcost += sum(
                self.v_generation[gen, period, t]
                * (fuelcost * cost_profile[t] + emission_rate * self.CO2_price)
                * self.sample_factor[t]
                for t in self.s_time
            )
        for cons in self.s_load:
            opcost += sum(
                self.v_load_shed[cons, period, t] * self.load_shed_penalty * self.sample_factor[t] for t in self.s_time
            )

        # compute present value of future annual costs
        year_0 = self.investment_years[0]
        N_this = period - year_0
        # Number of years since start
        if period == self.investment_years[-1]:
            # last period - lasts until finance_years
            N_next = self.finance_years  # e.g. 30 years
        else:
            #
            ind_this_period = self.investment_years.index(period)
            N_next = self.investment_years[ind_this_period + 1] - year_0
        opcost = opcost * (
            annuityfactor(self.finance_interest_rate, N_next) - annuityfactor(self.finance_interest_rate, N_this)
        )
        return opcost

    def costOperationSingleGen(self, model, g, stage):
        """Operational costs: cost of gen, load shed (NPV)"""
        opcost = 0
        # operation cost per year:
        opcost = sum(
            model.generation[g, t, stage]
            * model.samplefactor[t]
            * (
                model.genCostAvg[g] * model.genCostProfile[g, t]
                + model.genTypeEmissionRate[model.genType[g]] * model.CO2price
            )
            for t in model.TIME
        )

        # compute present value of future annual costs
        if stage == len(model.STAGE):
            opcost = opcost * (
                annuityfactor(model.financeInterestrate, model.financeYears)
                - annuityfactor(model.financeInterestrate, int(stage - 1) * model.stage2TimeDelta)
            )
        else:
            opcost = opcost * annuityfactor(model.financeInterestrate, model.stage2TimeDelta)
        # opcost = opcost*discount_t0
        return opcost

    # TODO: Check if use_integer and sense is needed.
    # Perhaps needed when solved with command line mpi?
    def scenario_creator(self, scenario_name, probability, dict_data):
        """
        Creates a list of non-leaf tree node objects associated with this scenario

        Returns a Pyomo Concrete Model
        """

        # dict_data = scenario_creator_kwargs["dict_data"]
        # grid_data = scenario_creator_kwargs["grid_data"]
        # probabilities = scenario_creator_kwargs["cond_prob"]
        # scenario_data_callback = scenario_creator_kwargs["scenario_data_callback"]
        # dict_data = scenario_data_callback(scenario_name, grid_data, dict_data)
        model = self.createConcreteModel(dict_data=dict_data)
        model._mpisppy_probability = probability  # probabilities[scenario_name]

        # Convert slices to lists as slices otherwise give error
        stage1vars = (
            _slice_to_list(model.branchNewCables[:, 1])
            + _slice_to_list(model.branchNewCapacity[:, 1])
            + _slice_to_list(model.newNodes[:, 1])
            + _slice_to_list(model.genNewCapacity[:, 1])
        )
        # Create the list of nodes associated with the scenario (for two stage,
        # there is only one node associated with the scenario--leaf nodes are
        # ignored).
        root_node = scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.opCost[1] + model.investmentCost[1],
            nonant_list=stage1vars,
            scen_model=model,
        )
        model._mpisppy_node_list = [root_node]
        return model

    def computeBranchCongestionRent(self, model, b, stage=1):
        """
        Compute annual congestion rent for a given branch
        """
        # TODO: use nodal price, not area price.
        N1 = model.branchNodeFrom[b]
        N2 = model.branchNodeTo[b]

        area1 = model.nodeArea[N1]
        area2 = model.nodeArea[N2]

        flow = []
        deltaP = []

        for t in model.TIME:
            deltaP.append(
                abs(self.computeAreaPrice(model, area1, t, stage) - self.computeAreaPrice(model, area2, t, stage))
                * model.samplefactor[t]
            )
            flow.append(model.branchFlow21[b, t, stage].value + model.branchFlow12[b, t, stage].value)

        return sum(deltaP[i] * flow[i] for i in range(len(deltaP)))

    def computeCostBranch(self, model, b, stage=2, include_om=False):
        """Investment cost of single branch NPV"""
        cost1 = self.costBranch(model, b, stage=stage)
        cost1npv = self.npvInvestment(
            model,
            stage=stage,
            investment=cost1,
            includeOM=include_om,
            subtractSalvage=True,
        )
        cost_value = pyo.value(cost1npv)
        return cost_value

    def computeCostNode(self, model, n, include_om=False):
        """Investment cost of single node

        corresponds to cost in abstract model"""

        # Re-use method used in optimisation
        # NOTE: adding "()" after the expression gives the value
        cost = {}
        costnpv = {}
        cost_value = 0
        for stage in model.STAGE:
            cost[stage] = self.costNode(model, n, stage=stage)
            costnpv[stage] = self.npvInvestment(
                model,
                stage=stage,
                investment=cost[stage],
                includeOM=include_om,
                subtractSalvage=True,
            )
            cost_value = cost_value + pyo.value(costnpv[stage])
        return cost_value

    def computeCostGenerator(self, model, g, stage=2, include_om=False):
        """Investment cost of generator NPV"""
        cost1 = self.costGen(model, g, stage=stage)
        cost1npv = self.npvInvestment(
            model,
            stage=stage,
            investment=cost1,
            includeOM=include_om,
            subtractSalvage=True,
        )
        cost_value = pyo.value(cost1npv)
        return cost_value

    def computeGenerationCost(self, model, g, stage):
        """compute NPV cost of generation (+ CO2 emissions)"""
        cost1 = self.costOperationSingleGen(model, g, stage=stage)
        cost_value = pyo.value(cost1)
        return cost_value

    def computeDemand(self, model, c, t):
        """compute demand at specified load ant time"""
        return model.demandAvg[c] * model.demandProfile[c, t]

    def computeCurtailment(self, model, g, t, stage=2):
        """compute curtailment [MWh] per generator per hour"""
        cur = 0
        gen_max = 0
        if model.generation[g, t, stage].value > 0 and model.genCostAvg[g] * model.genCostProfile[g, t] < 1:
            if stage == 1:
                gen_max = model.genCapacity[g]
                if model.genNewCapacity[g, stage].value is not None:
                    gen_max = gen_max + model.genNewCapacity[g, stage].value
                cur = gen_max * model.genCapacityProfile[g, t] - model.generation[g, t, stage].value
            if stage == 2:
                gen_max = model.genCapacity[g] + model.genCapacity2[g]
                if model.genNewCapacity[g, stage - 1].value is not None:
                    gen_max = gen_max + model.genNewCapacity[g, stage - 1].value
                if model.genNewCapacity[g, stage].value is not None:
                    gen_max = gen_max + model.genNewCapacity[g, stage].value
                cur = gen_max * model.genCapacityProfile[g, t] - model.generation[g, t, stage].value
        return cur

    def computeAreaEmissions(self, model, c, stage=2, cost=False):
        """compute total emissions from a load/country"""
        # TODO: ensure that all nodes are mapped to a country/load
        n = model.demNode[c]
        expr = 0
        gen = model.generation
        if stage == 1:
            ar = annuityfactor(model.financeInterestrate, model.stage2TimeDelta)
        elif stage == 2:
            ar = annuityfactor(model.financeInterestrate, model.financeYears) - annuityfactor(
                model.financeInterestrate, model.stage2TimeDelta
            )

        for g in model.GEN:
            if model.genNode[g] == n:
                expr += sum(
                    gen[g, t, stage].value * model.samplefactor[t] * model.genTypeEmissionRate[model.genType[g]]
                    for t in model.TIME
                )
        if cost:
            expr = expr * model.CO2price.value * ar
        return expr

    def computeAreaRES(self, model, j, shareof, stage=2):
        """compute renewable share of demand or total generation capacity"""
        node = model.demNode[j]
        area = model.nodeArea[node]
        Rgen = 0
        costlimit_RES = 1  # limit for what to consider renewable generator
        gen_p = model.generation
        gen = 0
        dem = sum(model.demandAvg[j] * model.demandProfile[j, t] for t in model.TIME)
        for g in model.GEN:
            if model.nodeArea[model.genNode[g]] == area:
                if model.genCostAvg[g] <= costlimit_RES:
                    Rgen += sum(gen_p[g, t, stage].value for t in model.TIME)
                else:
                    gen += sum(gen_p[g, t, stage].value for t in model.TIME)

        if shareof == "dem":
            return Rgen / dem
        elif shareof == "gen":
            if gen + Rgen > 0:
                return Rgen / (gen + Rgen)
            else:
                return np.nan
        else:
            print("Choose shareof dem or gen")

    def computeAreaPrice(self, model, area, t, stage=2):
        """cumpute the approximate area price based on max marginal cost"""
        mc = []
        for g in model.GEN:
            gen = model.generation[g, t, stage].value
            if gen > 0:
                if model.nodeArea[model.genNode[g]] == area:
                    mc.append(
                        model.genCostAvg[g] * model.genCostProfile[g, t]
                        + model.genTypeEmissionRate[model.genType[g]] * model.CO2price.value
                    )
        if len(mc) == 0:
            # no generators in area, so no price...
            price = np.nan
        else:
            price = max(mc)
        return price

    def computeAreaWelfare(self, model, c, t, stage=2):
        """compute social welfare for a given area and time step

        Returns: Welfare, ProducerSurplus, ConsumerSurplus,
                 CongestionRent, IMport, eXport
        """
        node = model.demNode[c]
        area = model.nodeArea[node]
        PS = 0
        CS = 0
        CR = 0
        GC = 0
        gen = 0
        dem = model.demandAvg[c] * model.demandProfile[c, t]
        price = self.computeAreaPrice(model, area, t, stage)
        # branch_capex = self.computeAreaCostBranch(model,c,include_om=True) #npv
        # gen_capex = self.computeAreaCostGen(model,c) #annualized

        # TODO: check phase1 vs phase2
        gen_p = model.generation
        flow12 = model.branchFlow12
        flow21 = model.branchFlow21

        for g in model.GEN:
            if model.nodeArea[model.genNode[g]] == area:
                gen += gen_p[g, t, stage].value
                GC += gen_p[g, t, stage].value * (
                    model.genCostAvg[g] * model.genCostProfile[g, t]
                    + model.genTypeEmissionRate[model.genType[g]] * model.CO2price.value
                )
        CS = (model.VOLL - price) * dem
        CC = price * dem
        PS = price * gen - GC
        if gen > dem:
            X = price * (gen - dem)
            IM = 0
            flow = []
            price2 = []
            for j in model.BRANCH:
                if model.nodeArea[model.branchNodeFrom[j]] == area and model.nodeArea[model.branchNodeTo[j]] != area:
                    flow.append(flow12[j, t, stage].value)
                    price2.append(self.computeAreaPrice(model, model.nodeArea[model.branchNodeTo[j]], t, stage))
                if model.nodeArea[model.branchNodeTo[j]] == area and model.nodeArea[model.branchNodeFrom[j]] != area:
                    flow.append(flow21[j, t, stage].value)
                    price2.append(self.computeAreaPrice(model, model.nodeArea[model.branchNodeFrom[j]], t, stage))
            CR = sum(flow[i] * (price2[i] - price) for i in range(len(flow))) / 2
        elif gen < dem:
            X = 0
            IM = price * (dem - gen)
            flow = []
            price2 = []
            for j in model.BRANCH:
                if model.nodeArea[model.branchNodeFrom[j]] == area and model.nodeArea[model.branchNodeTo[j]] != area:
                    flow.append(flow21[j, t, stage].value)
                    price2.append(self.computeAreaPrice(model, model.nodeArea[model.branchNodeTo[j]], t, stage))
                if model.nodeArea[model.branchNodeTo[j]] == area and model.nodeArea[model.branchNodeFrom[j]] != area:
                    flow.append(flow12[j, t, stage].value)
                    price2.append(self.computeAreaPrice(model, model.nodeArea[model.branchNodeFrom[j]], t, stage))
            CR = sum(flow[i] * (price - price2[i]) for i in range(len(flow))) / 2
        else:
            X = 0
            IM = 0
            flow = [0]
            price2 = [0]
            CR = 0
        W = PS + CS + CR
        return {
            "W": W,
            "PS": PS,
            "CS": CS,
            "CC": CC,
            "GC": GC,
            "CR": CR,
            "IM": IM,
            "X": X,
        }

    def computeAreaCostBranch(self, model, c, stage, include_om=False):
        """Investment cost for branches connected to an given area"""
        node = model.demNode[c]
        area = model.nodeArea[node]
        cost = 0

        for b in model.BRANCH:
            if model.nodeArea[model.branchNodeTo[b]] == area:
                cost += self.computeCostBranch(model, b, stage, include_om)
            elif model.nodeArea[model.branchNodeFrom[b]] == area:
                cost += self.computeCostBranch(model, b, stage, include_om)

        # assume 50/50 cost sharing
        return cost / 2

    def computeAreaCostGen(self, model, c):
        """compute capital costs for new generator capacity"""
        node = model.demNode[c]
        area = model.nodeArea[node]
        gen_capex = 0

        for g in model.GEN:
            if model.nodeArea[model.genNode[g]] == area:
                typ = model.genType[g]
                gen_capex += model.genTypeCost[typ] * model.genNewCapacity[g].value * model.genCostScale[g]

        return gen_capex

    def saveDeterministicResults(self, model, excel_file):
        """export results to excel file

        Parameters
        ==========
        model : Pyomo model
            concrete instance of optimisation model
        excel_file : string
            name of Excel file to create

        """
        df_branches = pd.DataFrame()
        df_nodes = pd.DataFrame()
        df_gen = pd.DataFrame()
        df_load = pd.DataFrame()
        # Specifying columns is not necessary, but useful to get the wanted
        # ordering
        df_branches = pd.DataFrame(
            columns=[
                "from",
                "to",
                "fArea",
                "tArea",
                "type",
                "existingCapacity",
                "expand",
                "newCables",
                "newCapacity",
                "flow12avg_1",
                "flow21avg_1",
                "existingCapacity2",
                "expand2",
                "newCables2",
                "newCapacity2",
                "flow12avg_2",
                "flow21avg_2",
                "cost_withOM",
                "congestion_rent",
            ]
        )
        df_nodes = pd.DataFrame(columns=["num", "area", "newNodes1", "newNodes2", "cost", "cost_withOM"])
        df_gen = pd.DataFrame(
            columns=[
                "num",
                "node",
                "area",
                "type",
                "pmax",
                "expand",
                "newCapacity",
                "pmax2",
                "expand2",
                "newCapacity2",
            ]
        )
        df_load = pd.DataFrame(
            columns=[
                "num",
                "node",
                "area",
                "Pavg",
                "Pmin",
                "Pmax",
                "emissions",
                "emissionCap",
                "emission_cost",
                "price_avg",
                "RES%dem",
                "RES%gen",
                "IM",
                "EX",
                "CS",
                "PS",
                "CR",
                "CAPEX",
                "Welfare",
            ]
        )

        for j in model.BRANCH:
            df_branches.loc[j, "num"] = j
            df_branches.loc[j, "from"] = model.branchNodeFrom[j]
            df_branches.loc[j, "to"] = model.branchNodeTo[j]
            df_branches.loc[j, "fArea"] = model.nodeArea[model.branchNodeFrom[j]]
            df_branches.loc[j, "tArea"] = model.nodeArea[model.branchNodeTo[j]]
            df_branches.loc[j, "type"] = model.branchType[j]
            df_branches.loc[j, "existingCapacity"] = model.branchExistingCapacity[j]
            df_branches.loc[j, "expand"] = model.branchExpand[j]
            df_branches.loc[j, "existingCapacity2"] = model.branchExistingCapacity2[j]
            df_branches.loc[j, "expand2"] = model.branchExpand2[j]
            for s in model.STAGE:
                if s == 1:
                    df_branches.loc[j, "newCables"] = model.branchNewCables[j, s].value
                    df_branches.loc[j, "newCapacity"] = model.branchNewCapacity[j, s].value
                    df_branches.loc[j, "flow12avg_1"] = np.mean(
                        [model.branchFlow12[(j, t, s)].value for t in model.TIME]
                    )
                    df_branches.loc[j, "flow21avg_1"] = np.mean(
                        [model.branchFlow21[(j, t, s)].value for t in model.TIME]
                    )
                    cap1 = model.branchExistingCapacity[j] + model.branchNewCapacity[j, s].value
                    if cap1 > 0:
                        df_branches.loc[j, "flow12%_1"] = df_branches.loc[j, "flow12avg_1"] / cap1
                        df_branches.loc[j, "flow21%_1"] = df_branches.loc[j, "flow21avg_1"] / cap1
                elif s == 2:
                    df_branches.loc[j, "newCables2"] = model.branchNewCables[j, s].value
                    df_branches.loc[j, "newCapacity2"] = model.branchNewCapacity[j, s].value
                    df_branches.loc[j, "flow12avg_2"] = np.mean(
                        [model.branchFlow12[(j, t, s)].value for t in model.TIME]
                    )
                    df_branches.loc[j, "flow21avg_2"] = np.mean(
                        [model.branchFlow21[(j, t, s)].value for t in model.TIME]
                    )
                    cap1 = model.branchExistingCapacity[j] + model.branchNewCapacity[j, s - 1].value
                    cap2 = cap1 + model.branchExistingCapacity2[j] + model.branchNewCapacity[j, s].value
                    if cap2 > 0:
                        df_branches.loc[j, "flow12%_2"] = df_branches.loc[j, "flow12avg_2"] / cap2
                        df_branches.loc[j, "flow21%_2"] = df_branches.loc[j, "flow21avg_2"] / cap2
            # branch costs
            df_branches.loc[j, "cost"] = sum(self.computeCostBranch(model, j, stage) for stage in model.STAGE)
            df_branches.loc[j, "cost_withOM"] = sum(
                self.computeCostBranch(model, j, stage, include_om=True) for stage in model.STAGE
            )
            df_branches.loc[j, "congestion_rent"] = self.computeBranchCongestionRent(
                model, j, stage=len(model.STAGE)
            ) * annuityfactor(model.financeInterestrate, model.financeYears)

        for j in model.NODE:
            df_nodes.loc[j, "num"] = j
            df_nodes.loc[j, "area"] = model.nodeArea[j]
            for s in model.STAGE:
                if s == 1:
                    df_nodes.loc[j, "newNodes1"] = model.newNodes[j, s].value
                elif s == 2:
                    df_nodes.loc[j, "newNodes2"] = model.newNodes[j, s].value
            df_nodes.loc[j, "cost"] = self.computeCostNode(model, j)
            df_nodes.loc[j, "cost_withOM"] = self.computeCostNode(model, j, include_om=True)

        for j in model.GEN:
            df_gen.loc[j, "num"] = j
            df_gen.loc[j, "area"] = model.nodeArea[model.genNode[j]]
            df_gen.loc[j, "node"] = model.genNode[j]
            df_gen.loc[j, "type"] = model.genType[j]
            df_gen.loc[j, "pmax"] = model.genCapacity[j]
            df_gen.loc[j, "pmax2"] = model.genCapacity2[j]
            df_gen.loc[j, "expand"] = model.genExpand[j]
            df_gen.loc[j, "expand2"] = model.genExpand2[j]
            df_gen.loc[j, "emission_rate"] = model.genTypeEmissionRate[model.genType[j]]
            for s in model.STAGE:
                if s == 1:
                    df_gen.loc[j, "emission1"] = model.genTypeEmissionRate[model.genType[j]] * sum(
                        model.generation[j, t, s].value * model.samplefactor[t] for t in model.TIME
                    )
                    df_gen.loc[j, "Pavg1"] = np.mean([model.generation[(j, t, s)].value for t in model.TIME])
                    df_gen.loc[j, "Pmin1"] = np.min([model.generation[(j, t, s)].value for t in model.TIME])
                    df_gen.loc[j, "Pmax1"] = np.max([model.generation[(j, t, s)].value for t in model.TIME])
                    df_gen.loc[j, "curtailed_avg1"] = np.mean(
                        [self.computeCurtailment(model, j, t, stage=1) for t in model.TIME]
                    )
                    df_gen.loc[j, "newCapacity"] = model.genNewCapacity[j, s].value
                    df_gen.loc[j, "cost_NPV1"] = self.computeGenerationCost(model, j, stage=1)
                elif s == 2:
                    df_gen.loc[j, "emission2"] = model.genTypeEmissionRate[model.genType[j]] * sum(
                        model.generation[j, t, s].value * model.samplefactor[t] for t in model.TIME
                    )
                    df_gen.loc[j, "Pavg2"] = np.mean([model.generation[(j, t, s)].value for t in model.TIME])
                    df_gen.loc[j, "Pmin2"] = np.min([model.generation[(j, t, s)].value for t in model.TIME])
                    df_gen.loc[j, "Pmax2"] = np.max([model.generation[(j, t, s)].value for t in model.TIME])
                    df_gen.loc[j, "curtailed_avg2"] = np.mean(
                        [self.computeCurtailment(model, j, t, stage=2) for t in model.TIME]
                    )
                    df_gen.loc[j, "newCapacity2"] = model.genNewCapacity[j, s].value
                    df_gen.loc[j, "cost_NPV2"] = self.computeGenerationCost(model, j, stage=2)

            df_gen.loc[j, "cost_investment"] = sum(self.computeCostGenerator(model, j, stage) for stage in model.STAGE)
            df_gen.loc[j, "cost_investment_withOM"] = sum(
                self.computeCostGenerator(model, j, stage, include_om=True) for stage in model.STAGE
            )

        print("TODO: powergim.saveDeterministicResults LOAD:" "only showing phase 2 (after 2nd stage investments)")
        # stage=2
        stage = max(model.STAGE)

        def _n(n, p):
            return n + str(p)

        for j in model.LOAD:
            df_load.loc[j, "num"] = j
            df_load.loc[j, "node"] = model.demNode[j]
            df_load.loc[j, "area"] = model.nodeArea[model.demNode[j]]
            df_load.loc[j, "price_avg"] = np.mean(
                [self.computeAreaPrice(model, df_load.loc[j, "area"], t, stage=stage) for t in model.TIME]
            )
            df_load.loc[j, "Pavg"] = np.mean([self.computeDemand(model, j, t) for t in model.TIME])
            df_load.loc[j, "Pmin"] = np.min([self.computeDemand(model, j, t) for t in model.TIME])
            df_load.loc[j, "Pmax"] = np.max([self.computeDemand(model, j, t) for t in model.TIME])
            if model.CO2price.value > 0:
                df_load.loc[j, "emissionCap"] = model.emissionCap[j]
            df_load.loc[j, _n("emissions", stage)] = self.computeAreaEmissions(model, j, stage=stage)
            df_load.loc[j, _n("emission_cost", stage)] = self.computeAreaEmissions(model, j, stage=stage, cost=True)
            df_load.loc[j, _n("RES%dem", stage)] = self.computeAreaRES(model, j, stage=stage, shareof="dem")
            df_load.loc[j, _n("RES%gen", stage)] = self.computeAreaRES(model, j, stage=stage, shareof="gen")
            df_load.loc[j, _n("Welfare", stage)] = sum(
                self.computeAreaWelfare(model, j, t, stage=stage)["W"] * model.samplefactor[t] for t in model.TIME
            )
            df_load.loc[j, _n("PS", stage)] = sum(
                self.computeAreaWelfare(model, j, t, stage=stage)["PS"] * model.samplefactor[t] for t in model.TIME
            )
            df_load.loc[j, _n("CS", stage)] = sum(
                self.computeAreaWelfare(model, j, t, stage=stage)["CS"] * model.samplefactor[t] for t in model.TIME
            )
            df_load.loc[j, _n("CR", stage)] = sum(
                self.computeAreaWelfare(model, j, t, stage=stage)["CR"] * model.samplefactor[t] for t in model.TIME
            )
            df_load.loc[j, _n("IM", stage)] = sum(
                self.computeAreaWelfare(model, j, t, stage=stage)["IM"] * model.samplefactor[t] for t in model.TIME
            )
            df_load.loc[j, _n("EX", stage)] = sum(
                self.computeAreaWelfare(model, j, t, stage=stage)["X"] * model.samplefactor[t] for t in model.TIME
            )
            df_load.loc[j, _n("CAPEX1", stage)] = self.computeAreaCostBranch(model, j, stage, include_om=False)

        df_cost = pd.DataFrame(columns=["value", "unit"])
        df_cost.loc["InvestmentCosts", "value"] = sum(model.investmentCost[s].value for s in model.STAGE) / 1e9
        df_cost.loc["OperationalCosts", "value"] = sum(model.opCost[s].value for s in model.STAGE) / 1e9
        df_cost.loc["newTransmission", "value"] = (
            sum(self.computeCostBranch(model, b, stage, include_om=True) for b in model.BRANCH for stage in model.STAGE)
            / 1e9
        )
        df_cost.loc["newGeneration", "value"] = (
            sum(self.computeCostGenerator(model, g, stage, include_om=True) for g in model.GEN for stage in model.STAGE)
            / 1e9
        )
        df_cost.loc["newNodes", "value"] = (
            sum(self.computeCostNode(model, n, include_om=True) for n in model.NODE) / 1e9
        )
        df_cost.loc["InvestmentCosts", "unit"] = "10^9 EUR"
        df_cost.loc["OperationalCosts", "unit"] = "10^9 EUR"
        df_cost.loc["newTransmission", "unit"] = "10^9 EUR"
        df_cost.loc["newGeneration", "unit"] = "10^9 EUR"
        df_cost.loc["newNodes", "unit"] = "10^9 EUR"

        with pd.ExcelWriter(excel_file) as writer:
            df_cost.to_excel(excel_writer=writer, sheet_name="cost")
            df_branches.to_excel(excel_writer=writer, sheet_name="branches")
            df_nodes.to_excel(excel_writer=writer, sheet_name="nodes")
            df_gen.to_excel(excel_writer=writer, sheet_name="generation")
            df_load.to_excel(excel_writer=writer, sheet_name="demand")

    def extractResultingGridData(self, grid_data, model=None, file_ph=None, stage=1, scenario=None, newData=False):
        """Extract resulting optimal grid layout from simulation results

        Parameters
        ==========
        grid_data : powergama.GridData
            grid data class
        model : Pyomo model
            concrete instance of optimisation model containing det. results
        file_ph : string
            CSV file containing results from stochastic solution
        stage : int
            Which stage to extract data for (1 or 2).
            1: only stage one investments included (default)
            2: both stage one and stage two investments included
        scenario : int
            which stage 2 scenario to get data for (only relevant when stage=2)
        newData : Boolean
            Choose whether to use only new data (True) or add new data to
            existing data (False)

        Use either model or file_ph parameter

        Returns
        =======
        GridData object reflecting optimal solution
        """

        grid_res = copy.deepcopy(grid_data)
        res_brC = pd.DataFrame(data=grid_res.branch["capacity"])
        res_N = pd.DataFrame(data=grid_res.node["existing"])
        res_G = pd.DataFrame(data=grid_res.generator["pmax"])
        if newData:
            res_brC[res_brC > 0] = 0
            #            res_N[res_N>0] = 0
            res_G[res_G > 0] = 0

        if model is not None:
            # Deterministic optimisation results
            if stage >= 1:
                stage1 = 1
                for j in model.BRANCH:
                    res_brC.loc[j, "capacity"] += model.branchNewCapacity[j, stage1].value
                for j in model.NODE:
                    res_N.loc[j, "existing"] += int(model.newNodes[j, stage1].value)
                for j in model.GEN:
                    newgen = model.genNewCapacity[j, stage1].value
                    if newgen is not None:
                        res_G.loc[j, "pmax"] += model.genNewCapacity[j, stage1].value
            if stage >= 2:
                # add to investments in stage 1
                # Need to read from model, as data may differ between scenarios
                # res_brC["capacity"] += grid_res.branch["capacity2"]
                # res_G["pmax"] += grid_res.generator["pmax2"]
                for j in model.BRANCH:
                    res_brC.loc[j, "capacity"] += pyo.value(model.branchExistingCapacity2[j])
                    res_brC.loc[j, "capacity"] += pyo.value(model.branchNewCapacity[j, stage])
                for j in model.GEN:
                    res_G.loc[j, "pmax"] += pyo.value(model.genCapacity2[j])
                    newgen = model.genNewCapacity[j, stage].value
                    if newgen is not None:
                        res_G.loc[j, "pmax"] += model.genNewCapacity[j, stage].value
                for j in model.NODE:
                    res_N.loc[j, "existing"] += int(model.newNodes[j, stage].value)
        elif file_ph is not None:
            # Stochastic optimisation results
            df_ph = pd.read_csv(
                file_ph,
                header=None,
                skipinitialspace=True,
                names=["stage", "node", "var", "var_indx", "value"],
                na_values=["None"],
            ).fillna(0)
            # var_indx consists of e.g. [ind,stage], or [ind,time,stage]
            # expand into three columns [ind,time,stage]
            ind_split = df_ph["var_indx"].str.split(":", expand=True)
            if ind_split.shape[1] != 3:
                raise Exception("Was assuming different format" " - implementation error")
            mask_only2 = ind_split[2] is None
            ind_split.loc[mask_only2, 2] = ind_split.loc[mask_only2, 2]
            ind_split.loc[mask_only2, 1] = ind_split.loc[mask_only2, 2]
            df_ph[["ind_i", "ind_time", "ind_stage"]] = ind_split
            df_ph["ind_i"] = df_ph["ind_i"].str.replace("'", "")
            if stage >= 1:
                stage1 = 1
                df_branchNewCapacity = df_ph[(df_ph["var"] == "branchNewCapacity") & (df_ph["stage"] == stage1)]
                df_newNodes = df_ph[(df_ph["var"] == "newNodes") & (df_ph["stage"] == stage1)]
                df_newGen = df_ph[(df_ph["var"] == "genNewCapacity") & (df_ph["stage"] == stage1)]
                for k, row in df_branchNewCapacity.iterrows():
                    res_brC.loc[int(row["ind_i"]), "capacity"] += float(row["value"])
                for k, row in df_newNodes.iterrows():
                    res_N.loc[row["ind_i"], "existing"] += int(float(row["value"]))
                for k, row in df_newGen.iterrows():
                    res_G.loc[int(row["ind_i"]), "pmax"] += float(row["value"])
            if stage >= 2:
                if scenario is None:
                    raise Exception('Missing input "scenario"')
                res_brC["capacity"] += grid_res.branch["capacity2"]
                res_G["pmax"] += grid_res.generator["pmax2"]

                df_branchNewCapacity = df_ph[
                    (df_ph["var"] == "branchNewCapacity")
                    & (df_ph["stage"] == stage)
                    & (df_ph["node"] == "Scenario{}".format(scenario))
                ]
                df_newNodes = df_ph[
                    (df_ph["var"] == "newNodes")
                    & (df_ph["stage"] == stage)
                    & (df_ph["node"] == "Scenario{}".format(scenario))
                ]
                df_newGen = df_ph[
                    (df_ph["var"] == "genNewCapacity")
                    & (df_ph["stage"] == stage)
                    & (df_ph["node"] == "Scenario{}".format(scenario))
                ]

                for k, row in df_branchNewCapacity.iterrows():
                    res_brC.loc[int(row["ind_i"]), "capacity"] += float(row["value"])
                for k, row in df_newNodes.iterrows():
                    res_N.loc[row["ind_i"], "existing"] += int(float(row["value"]))
                for k, row in df_newGen.iterrows():
                    res_G.loc[int(row["ind_i"]), "pmax"] += float(row["value"])
        else:
            raise Exception("Missing input parameter")

        grid_res.branch["capacity"] = res_brC["capacity"]
        grid_res.node["existing"] = res_N["existing"]
        grid_res.generator["pmax"] = res_G["pmax"]
        grid_res.branch = grid_res.branch[grid_res.branch["capacity"] > self._NUMERICAL_THRESHOLD_ZERO]
        grid_res.node = grid_res.node[grid_res.node["existing"] > self._NUMERICAL_THRESHOLD_ZERO]
        return grid_res

    def extract_all_variable_values(self):
        """Extract variable values and return as a dictionary of pandas milti-index series"""
        all_values = {}
        all_obj = self.component_objects(ctype=pyo.Var)
        for myvar in all_obj:
            # extract the variable index names in the right order
            if myvar._implicit_subsets is None:
                index_names = None
            else:
                index_names = [index_set.name for index_set in myvar._implicit_subsets]
            var_values = myvar.get_values()
            if not var_values:
                # empty dictionary, so no variables to store
                all_values[myvar.name] = None
                continue
            # This creates a pandas.Series:
            df = pd.DataFrame.from_dict(var_values, orient="index", columns=["value"])["value"]
            if index_names is not None:
                df.index = pd.MultiIndex.from_tuples(df.index, names=index_names)

            # ignore NA values
            df = df.dropna()
            if df.empty:
                all_values[myvar.name] = None
                continue

            all_values[myvar.name] = df
        return all_values

    def model_info(self, model):
        """Return info about model as dictionary"""
        model_vars = model.component_data_objects(ctype=pyo.Var)
        varlist = list(model_vars)
        integers = [v.is_integer() for v in varlist]
        continuous = [v.is_continuous() for v in varlist]
        constraints = list(model.component_data_objects(ctype=pyo.Constraint))
        objectives = list(model.component_data_objects(ctype=pyo.Objective))
        info = dict()
        info["number of variables"] = len(varlist)
        info["number of integer variables"] = sum(integers)
        info["number of continuous variables"] = sum(continuous)
        info["number of constraints"] = len(constraints)
        info["number of objectives"] = len(objectives)
        var_overview = {data.name: len(list(data)) for data in model.component_map(pyo.Var, active=True).values()}
        info["variables"] = var_overview
        return info
