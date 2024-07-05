"""
Module for power grid investment analyses

"""

import mpisppy.scenario_tree as scenario_tree
import pandas as pd
import pyomo.environ as pyo

import powergim

from .utils import annuityfactor

# import mpisppy.utils.sputils # alternative to scenario_tree


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
        self.CO2_cap = parameter_data["parameters"]["CO2_cap"]

        # sample factor scales from sample to annual value
        self.sample_factor = pd.Series(
            index=self.grid_data.profiles.index,
            data=self._HOURS_PER_YEAR / self.grid_data.profiles.shape[0],
        )
        if "frequency" in self.grid_data.profiles:
            # a column describing weight for each time-step
            self.sample_factor = self.grid_data.profiles["frequency"]

        self.has_load_flex_shift = "load_flex_shift_frac" in self.parameters
        self.has_load_flex_price = "load_flex_price_frac" in self.parameters

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
        # self.s_branchtype = pyo.Set(initialize=self.branchtypes.keys())
        # self.s_nodetype = pyo.Set()
        # self.s_gentype = pyo.Set()

    def create_variables(self):
        # NOTE: bounds vs constraints
        # In general, variable limits are given by constraints.
        # But bounds are needed when linearising proximal terms in stochastic
        # optimisation and are therefore included. Note also that Pyomo gives error if lb=ub (a bug),
        # so instead of specifying bounds lb=ub, we do it as constraint.

        # TODO: Initialize variables=0 if they may be omitted

        def bounds_branch_new_capacity(model, branch, period):
            # default max capacity is given by branch type and max number of cables
            branchtype = self.grid_data.branch.at[branch, "type"]
            cap_branchtype = self.branchtypes[branchtype]["max_cap"]
            maxnum_branchtype = self.branchtypes[branchtype]["max_num"]
            maxcap = maxnum_branchtype * cap_branchtype
            if self.grid_data.branch.at[branch, "max_newCap"] > 0:
                maxcap = self.grid_data.branch.at[branch, "max_newCap"]
            return (0, maxcap)

        self.v_branch_new_capacity = pyo.Var(
            self.s_branch,
            self.s_period,
            within=pyo.NonNegativeReals,
            bounds=bounds_branch_new_capacity,  # needed for proximal term linearisation in stochastic optimisation
        )

        # investment: new branch cables (needed for linearisation, see also model.cMaxNumberCables)
        def bounds_branch_new_cables(model, branch, period):
            branchtype = self.grid_data.branch.at[branch, "type"]
            maxnum_branchtype = self.branchtypes[branchtype]["max_num"]
            return (0, maxnum_branchtype)

        self.v_branch_new_cables = pyo.Var(
            self.s_branch,
            self.s_period,
            within=pyo.NonNegativeIntegers,
            initialize=0,
            bounds=bounds_branch_new_cables,
        )

        # investment: new nodes
        self.v_new_nodes = pyo.Var(self.s_node, self.s_period, within=pyo.Binary)

        def bounds_node_new_capacity(model, node, period):
            nodetype = self.grid_data.node.loc[node, "type"]
            maxcap = self.nodetypes[nodetype]["max_cap"]
            return (0, maxcap)

        self.v_node_new_capacity = pyo.Var(
            self.s_node,
            self.s_period,
            within=pyo.NonNegativeReals,
            bounds=bounds_node_new_capacity,  # needed for proximal term linearisation in stochastic optimisation
        )

        # investment: generation capacity
        def bounds_gen_new_capacity(model, gen, period):
            gentype = self.grid_data.generator.at[gen, "type"]
            maxcap = self.gentypes[gentype]["max_cap"]  # max sum
            if self.grid_data.generator.at[gen, "p_maxNew"] > 0:
                maxcap = self.grid_data.generator.at[gen, "p_maxNew"]
            # this does not work here, as it in some cases gives ub=0=lb -> using constraints instead
            # max_value = maxcap * self.grid_data.generator.loc[gen, f"expand_{period}"]
            return (0, maxcap)

        self.v_gen_new_capacity = pyo.Var(
            self.s_gen,
            self.s_period,
            within=pyo.NonNegativeReals,
            bounds=bounds_gen_new_capacity,
            initialize=0,
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
            ref = self.grid_data.consumer.at[consumer, "demand_ref"]
            if self.parameters["profiles_period_suffix"]:
                ref = f"{ref}_{period}"
            profile = self.grid_data.profiles.at[time, ref]

            demand_avg = 0
            previous_periods = [p for p in self.s_period if p <= period]
            for p in previous_periods:
                demand_avg += self.grid_data.consumer.at[consumer, f"demand_{p}"]
            ub = max(0, demand_avg * profile)
            return (0, ub)

        self.v_load_shed = pyo.Var(
            self.s_load,
            self.s_period,
            self.s_time,
            domain=pyo.NonNegativeReals,
            bounds=bounds_load_shed,
        )

        # Flexible load (shiftable and price sensitive)
        def bounds_load_flex_shift(model, consumer, period, t):
            ub = 0
            lb = 0
            if self.has_load_flex_shift:
                flex_frac = self.parameters["load_flex_shift_frac"][period]
                flex_onoff = self.parameters["load_flex_shift_max"][period]
                demand_avg = 0
                previous_periods = [p for p in self.s_period if p <= period]
                for p in previous_periods:
                    demand_avg += self.grid_data.consumer.at[consumer, f"demand_{p}"]
                ref = self.grid_data.consumer.at[consumer, "demand_ref"]
                if self.parameters["profiles_period_suffix"]:
                    ref = f"{ref}_{period}"
                profile = self.grid_data.profiles.at[t, ref]
                # need to set a lower bound on flex_demand (x) such that both
                # 1) x >- demand_avg*flex_frac (max downward flexibility) and
                # 2) x + demand_avg*profile > 0 (sum demand should not be negative)
                lb = -demand_avg * min(flex_frac, profile)
                ub = demand_avg * flex_frac * flex_onoff
            return (lb, ub)

        def bounds_load_flex_price(model, consumer, period, t):
            ub = 0
            if self.has_load_flex_price:
                flex_frac = self.parameters["load_flex_price_frac"][period]
                demand_avg = 0
                previous_periods = [p for p in self.s_period if p <= period]
                for p in previous_periods:
                    demand_avg += self.grid_data.consumer.at[consumer, f"demand_{p}"]
                ub = demand_avg * flex_frac
            return (0, ub)

        self.v_load_flex_shift = pyo.Var(
            self.s_load, self.s_period, self.s_time, domain=pyo.Reals, bounds=bounds_load_flex_shift
        )
        self.v_load_flex_price = pyo.Var(
            self.s_load, self.s_period, self.s_time, domain=pyo.NonNegativeReals, bounds=bounds_load_flex_price
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
            # TODO: Why does this make any difference?
            # THIS WORKS (more often) with CBC:
            # investment = 0
            # operation = 0
            # for period in self.s_period:
            #    investment += self.costInvestments(period)
            #    operation += self.costOperation(period)
            # This does NOT (always) work with CBC
            investment = pyo.summation(self.v_investment_cost)
            operation = pyo.summation(self.v_operating_cost)
            return investment + operation

        self.OBJ = pyo.Objective(rule=total_cost_objective_rule, sense=pyo.minimize)

    def create_constraints(self):
        # Power flow limited by installed (existing or new) capacity
        def _branch_flow_limit(branch, period, t):
            branch_existing_capacity = 0
            branch_new_capacity = 0
            previous_periods = [p for p in self.s_period if p <= period]
            for p in previous_periods:
                branch_existing_capacity += self.grid_data.branch.at[branch, f"capacity_{p}"]
                if self.grid_data.branch.at[branch, f"expand_{p}"] == 1:
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
            branchtype = self.grid_data.branch.at[branch, "type"]
            maxnum_branchtype = self.branchtypes[branchtype]["max_num"]
            max_num = maxnum_branchtype * self.grid_data.branch.at[branch, f"expand_{period}"]
            expr = self.v_branch_new_cables[branch, period] <= max_num
            return expr

        self.c_max_number_cables = pyo.Constraint(self.s_branch, self.s_period, rule=rule_max_new_cables)

        # No new branch capacity without new cables
        def rule_max_new_cap(model, branch, period):
            branchtype = self.grid_data.branch.at[branch, "type"]
            cap_branchtype = self.branchtypes[branchtype]["max_cap"]
            expr = (
                self.v_branch_new_capacity[branch, period] <= cap_branchtype * self.v_branch_new_cables[branch, period]
            )
            return expr

        self.c_max_new_branch_capacity = pyo.Constraint(self.s_branch, self.s_period, rule=rule_max_new_cap)

        # No new node capacity without new nodes
        def rule_new_nodes(model, node, period):
            nodetype = self.grid_data.node.at[node, "type"]
            max_node_cap = self.nodetypes[nodetype]["max_cap"]  # max (node num is binary, not integer)
            expr = self.v_node_new_capacity[node, period] <= max_node_cap * self.v_new_nodes[node, period]
            return expr

        self.c_new_nodes = pyo.Constraint(self.s_node, self.s_period, rule=rule_new_nodes)

        # Limit new generator capacity
        def rule_gen_new_capacity(model, gen, period):
            gentype = self.grid_data.generator.at[gen, "type"]
            maxcap = self.gentypes[gentype]["max_cap"]  # max sum (generators have no unit costs)
            if self.grid_data.generator.at[gen, "p_maxNew"] > 0:
                maxcap = self.grid_data.generator.at[gen, "p_maxNew"]
            max_value = maxcap * self.grid_data.generator.at[gen, f"expand_{period}"]
            return self.v_gen_new_capacity[gen, period] <= max_value

        self.c_max_new_gen_capacity = pyo.Constraint(self.s_gen, self.s_period, rule=rule_gen_new_capacity)

        # Generator output limitations
        # TODO: add option to set minimum output = timeseries for renewable,
        # i.e. disallov curtaliment (could be global parameter)
        def rule_max_gen_power(model, gen, period, t):
            cap_existing = 0
            cap_new = 0
            previous_periods = [p for p in self.s_period if p <= period]
            for p in previous_periods:
                cap_existing += self.grid_data.generator.at[gen, f"capacity_{p}"]
                if self.grid_data.generator.at[gen, f"expand_{p}"] == 1:
                    cap_new += self.v_gen_new_capacity[gen, p]
            cap = cap_existing + cap_new
            profile_ref = self.grid_data.generator.at[gen, "inflow_ref"]
            if self.parameters["profiles_period_suffix"]:
                profile_ref = f"{profile_ref}_{period}"
            profile_fac = self.grid_data.generator.at[gen, "inflow_fac"]
            profile_value = self.grid_data.profiles.at[t, profile_ref] * profile_fac
            allow_curtailment = self.grid_data.generator.at[gen, "allow_curtailment"]
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
            previous_periods = [p for p in self.s_period if p <= period]
            for p in previous_periods:
                cap_existing += self.grid_data.generator.at[gen, f"capacity_{p}"]
                cap_new += self.v_gen_new_capacity[gen, p]
            cap = cap_existing + cap_new
            max_p_avg = self.grid_data.generator.at[gen, "pavg"]
            if max_p_avg > 0:
                expr = sum(
                    self.v_generation[gen, period, t] * self.sample_factor[t] for t in self.s_time
                ) / self._HOURS_PER_YEAR <= (max_p_avg * cap)
            else:
                expr = pyo.Constraint.Skip
            return expr

        self.c_max_energy = pyo.Constraint(self.s_gen, self.s_period, rule=rule_max_energy)

        # fixed overall average demand for shiftable loads is zero
        def rule_load_shift_sum(model, cons, period):
            if not self.has_load_flex_shift:
                return pyo.Constraint.Skip

            loadshift_avg_mw = (
                sum(self.v_load_flex_shift[cons, period, t] * self.sample_factor[t] for t in self.s_time)
                / self._HOURS_PER_YEAR
            )
            expr = loadshift_avg_mw == 0
            return expr

        self.c_load_flex_shift_sum = pyo.Constraint(self.s_load, self.s_period, rule=rule_load_shift_sum)

        def rule_emission_cap_global(model, period):
            global_emission = 0
            for gen in self.s_gen:
                gentype = self.grid_data.generator.at[gen, "type"]
                emission_rate = self.gentypes[gentype]["CO2"]
                global_emission += sum(
                    self.v_generation[gen, period, t] * emission_rate * self.sample_factor[t] for t in self.s_time
                )
            return global_emission <= self.CO2_cap

        def rule_emission_cap_area(model, area, period):
            area_emission = 0
            # loop through all nodes to get right area
            # then loop through generators connected to these nodes
            for n in self.s_node:
                node_area = self.grid_data.node.at[n, "area"]
                if node_area == area:
                    for gen in self.s_gen:
                        gen_node = self.grid_data.generator.at[gen, "node"]
                        if gen_node == n:
                            gentype = self.grid_data.generator.at[gen, "type"]
                            emission_rate = self.gentypes[gentype]["CO2"]
                            area_emission += sum(
                                self.v_generation[gen, period, t] * emission_rate * self.sample_factor[t]
                                for t in self.s_time
                            )
            expr = area_emission <= self.CO2_cap[area]
            if (type(expr) is bool) and (expr is True):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr

        if self.CO2_cap is None:
            self.c_emission_cap = pyo.Constraint.Skip
        elif isinstance(self.CO2_cap, dict):
            # specified per area
            self.c_emission_cap = pyo.Constraint(self.s_area, self.s_period, rule=rule_emission_cap_area)
        else:
            # global cap
            self.c_emission_cap = pyo.Constraint(self.s_period, rule=rule_emission_cap_global)

        # Power balance in nodes : gen+demand+flow into node=0
        def rule_powerbalance(model, node, period, t):
            flow_into_node = 0
            # flow of power into node (subtrating losses)
            for branch in self.s_branch:
                node_from = self.grid_data.branch.at[branch, "node_from"]
                node_to = self.grid_data.branch.at[branch, "node_to"]
                if node_from == node:
                    # branch out of node
                    branchtype = self.grid_data.branch.at[branch, "type"]
                    dist = self.grid_data.branch.at[branch, "distance"]
                    loss_fix = self.branchtypes[branchtype]["loss_fix"]
                    loss_slope = self.branchtypes[branchtype]["loss_slope"]
                    flow_into_node -= self.v_branch_flow12[branch, period, t]
                    flow_into_node += self.v_branch_flow21[branch, period, t] * (1 - (loss_fix + loss_slope * dist))
                if node_to == node:
                    # branch into node
                    branchtype = self.grid_data.branch.at[branch, "type"]
                    dist = self.grid_data.branch.at[branch, "distance"]
                    loss_fix = self.branchtypes[branchtype]["loss_fix"]
                    loss_slope = self.branchtypes[branchtype]["loss_slope"]
                    flow_into_node -= self.v_branch_flow21[branch, period, t]
                    flow_into_node += self.v_branch_flow12[branch, period, t] * (1 - (loss_fix + loss_slope * dist))

            # generated power
            for gen in self.s_gen:
                node_gen = self.grid_data.generator.at[gen, "node"]
                if node_gen == node:
                    flow_into_node += self.v_generation[gen, period, t]

            # load shedding
            for cons in self.s_load:
                node_load = self.grid_data.consumer.at[cons, "node"]
                if node_load == node:
                    flow_into_node += self.v_load_shed[cons, period, t]

            # consumed power
            for cons in self.s_load:
                node_load = self.grid_data.consumer.at[cons, "node"]
                if node_load == node:
                    dem_avg = 0
                    previous_periods = [p for p in self.s_period if p <= period]
                    for p in previous_periods:
                        dem_avg += self.grid_data.consumer.at[cons, f"demand_{p}"]
                    dem_profile_ref = self.grid_data.consumer.at[cons, "demand_ref"]
                    if self.parameters["profiles_period_suffix"]:
                        dem_profile_ref = f"{dem_profile_ref}_{period}"
                    profile = self.grid_data.profiles.at[t, dem_profile_ref]

                    flow_into_node -= (
                        +dem_avg * profile
                        + self.v_load_flex_shift[cons, period, t]
                        + self.v_load_flex_price[cons, period, t]
                    )

            expr = flow_into_node == 0

            if (type(expr) is bool) and (expr is True):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr

        self.c_powerbalance = pyo.Constraint(self.s_node, self.s_period, self.s_time, rule=rule_powerbalance)

        # Power into node vs node capacity : gen+demand+flow into node=0
        def rule_nodecapacity(model, node, period):
            # sum power into node
            # rating according to branch capacity

            previous_periods = [p for p in self.s_period if p <= period]
            connected_capacity = 0
            # capacity of connected branches
            for branch in self.s_branch:
                node_from = self.grid_data.branch.at[branch, "node_from"]
                node_to = self.grid_data.branch.at[branch, "node_to"]
                if (node_from == node) or (node_to == node):
                    # does not matter if branch is directed to or from
                    for p in previous_periods:
                        connected_capacity += self.grid_data.branch.at[branch, f"capacity_{p}"]
                        connected_capacity += self.v_branch_new_capacity[branch, p]

            # TODO: add connected generation capacity?

            # TODO add connected load?

            cap_existing = 0
            cap_new = 0
            for p in previous_periods:
                cap_existing += self.grid_data.node.at[node, f"capacity_{p}"]
                if self.grid_data.node.at[node, f"expand_{p}"] == 1:
                    cap_new += self.v_node_new_capacity[node, p]
            node_capacity = cap_existing + cap_new

            expr = connected_capacity <= node_capacity

            if (type(expr) is bool) and (expr is True):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr

        self.c_nodecapacity = pyo.Constraint(self.s_node, self.s_period, rule=rule_nodecapacity)

    def costNode(self, node, period):
        """Expression for cost of node, investment cost no discounting"""
        n_cost = 0
        var_num = self.v_new_nodes[node, period]
        var_cap = self.v_node_new_capacity[node, period] * 1e-3  # MW to GW
        is_offshore = self.grid_data.node.at[node, "offshore"]  # 1 or 0
        nodetype = self.grid_data.node.at[node, "type"]
        nodetype_costs = self.nodetypes[nodetype]
        scale = self.grid_data.node.at[node, "cost_scaling"]
        n_cost += is_offshore * (nodetype_costs["S"] * var_num + nodetype_costs["Sp"] * var_cap)
        n_cost += (1 - is_offshore) * (nodetype_costs["L"] * var_num + nodetype_costs["Lp"] * var_cap)
        return scale * n_cost

    def costBranch(self, branch, period):
        """Expression for cost of branch, investment cost no discounting"""
        b_cost = 0
        # HGS 14.dec 2023: This simplifies expression, but makes no significant difference for coeff range.
        #        if self.grid_data.branch.at[branch,f"expand_{period}"]==0:
        #            # if branch capacity is not expandable, exclue from objective since it is unnecessary
        #            return b_cost
        var_num = self.v_branch_new_cables
        var_cap = self.v_branch_new_capacity
        branchtype = self.grid_data.branch.at[branch, "type"]
        branchtype_costs = self.branchtypes[branchtype]
        distance = self.grid_data.branch.at[branch, "distance"]
        b_cost += branchtype_costs["B"] * var_num[branch, period]
        b_cost += branchtype_costs["Bd"] * distance * var_num[branch, period]
        b_cost += branchtype_costs["Bdp"] * distance * var_cap[branch, period] * 1e-3

        # endpoint costs (difference onshore/offshore)
        node1 = self.grid_data.branch.at[branch, "node_from"]
        node2 = self.grid_data.branch.at[branch, "node_to"]
        is_offshore1 = self.grid_data.node.at[node1, "offshore"]
        is_offshore2 = self.grid_data.node.at[node2, "offshore"]
        for N in [is_offshore1, is_offshore2]:
            b_cost += N * (
                branchtype_costs["CS"] * var_num[branch, period]
                + branchtype_costs["CSp"] * var_cap[branch, period] * 1e-3
            )
            b_cost += (1 - N) * (
                branchtype_costs["CL"] * var_num[branch, period]
                + branchtype_costs["CLp"] * var_cap[branch, period] * 1e-3
            )
        scale = self.grid_data.branch.at[branch, "cost_scaling"]
        return scale * b_cost

    def costGen(self, gen, period):
        """Expression for cost of generator, investment cost no discounting"""
        g_cost = 0
        var_cap = self.v_gen_new_capacity
        gentype = self.grid_data.generator.at[gen, "type"]
        gentype_cost = self.gentypes[gentype]
        scale = self.grid_data.generator.at[gen, "cost_scaling"]
        g_cost += gentype_cost["Cp"] * var_cap[gen, period] * 1e-3
        return scale * g_cost

    def npvInvestment(self, period, investment, include_om=True, subtract_residual_value=True):
        """NPV of investment cost including lifetime O&M and salvage value

        Parameters
        ----------
        period : int
            Year of investment
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
            # NPV value at time 0
            residual_factor = (delta_years / self.finance_years) * (
                1 / ((1 + self.finance_interest_rate) ** self.finance_years)
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

        pv_cost = investment * (discount_t0 + om_factor - residual_factor)
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
            fuelcost = self.grid_data.generator.at[gen, f"fuelcost_{period}"]
            cost_profile_ref = self.grid_data.generator.at[gen, "fuelcost_ref"]
            if self.parameters["profiles_period_suffix"]:
                cost_profile_ref = f"{cost_profile_ref}_{period}"
            cost_profile = self.grid_data.profiles[cost_profile_ref]
            gentype = self.grid_data.generator.at[gen, "type"]
            emission_rate = self.gentypes[gentype]["CO2"]
            opcost += sum(
                self.v_generation[gen, period, t]
                * (fuelcost * cost_profile[t] + emission_rate * self.CO2_price)
                * self.sample_factor[t]
                for t in self.s_time
            )
        for cons in self.s_load:
            # negative cost for price sensitive load:
            if self.has_load_flex_price:
                price_sense_cap = self.parameters["load_flex_price_cap"][period]
                opcost -= sum(
                    self.v_load_flex_price[cons, period, t] * price_sense_cap * self.sample_factor[t]
                    for t in self.s_time
                )

            # penalty for load shedding:
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
        opcost = opcost * 1e-6  # EUR to MEUR
        return opcost

    def costOperationSingleGen(self, gen, period):
        """Operational costs: cost of gen, load shed (NPV)"""

        fuelcost = self.grid_data.generator.at[gen, f"fuelcost_{period}"]
        cost_profile_ref = self.grid_data.generator.at[gen, "fuelcost_ref"]
        if self.parameters["profiles_period_suffix"]:
            cost_profile_ref = f"{cost_profile_ref}_{period}"
        cost_profile = self.grid_data.profiles[cost_profile_ref]
        gentype = self.grid_data.generator.at[gen, "type"]
        emission_rate = self.gentypes[gentype]["CO2"]
        opcost = sum(
            self.v_generation[gen, period, t]
            * (fuelcost * cost_profile[t] + emission_rate * self.CO2_price)
            * self.sample_factor[t]
            for t in self.s_time
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
        opcost = opcost * 1e-6  # EUR to MEUR
        return opcost

    def scenario_creator(self, scenario_name, probability):
        """
        Creates a list of non-leaf tree node objects associated with this scenario

        Returns a Pyomo Concrete Model
        """

        # first investment year is stage 1
        stage1 = self.investment_years[0]

        # Convert slices to lists as slices otherwise give error
        stage1vars = (
            _slice_to_list(self.v_branch_new_cables[:, stage1])
            + _slice_to_list(self.v_branch_new_capacity[:, stage1])
            + _slice_to_list(self.v_new_nodes[:, stage1])
            + _slice_to_list(self.v_node_new_capacity[:, stage1])
            + _slice_to_list(self.v_gen_new_capacity[:, stage1])
            # + [self.v_operating_cost[stage1], self.v_investment_cost[stage1]]
        )
        # Create the list of nodes associated with the scenario (for two stage,
        # there is only one node associated with the scenario--leaf nodes are
        # ignored).
        # TODO: Any difference between these? (does not seem to be (and should not be))
        stage1cost = self.costInvestments(stage1) + self.costOperation(stage1)
        # stage1cost = self.v_operating_cost[stage1] + self.v_investment_cost[stage1]
        root_node = scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=stage1cost,
            nonant_list=stage1vars,
            scen_model=self,
        )
        self._mpisppy_probability = probability  # probabilities[scenario_name]
        self._mpisppy_node_list = [root_node]
        # mpisppy.utils.sputils.attach_root_node(self, stage1cost, stage1vars) # alternative to scenario_tree
        return self

    def extract_all_variable_values(self):
        """Extract variable values and return as a dictionary of pandas milti-index series"""
        all_values = {}
        all_obj = self.component_objects(ctype=pyo.Var)
        for myvar in all_obj:
            # extract the variable index names in the right order
            # if myvar.index_set().subsets() is None:
            #    index_names = None
            # else:
            #    index_names = [index_set.name for index_set in myvar.index_set().subsets()]
            var_values = myvar.get_values()
            if not var_values:
                # empty dictionary, so no variables to store
                all_values[myvar.name] = None
                continue
            # This creates a pandas.Series:
            df = pd.DataFrame.from_dict(var_values, orient="index", columns=["value"])["value"]
            index_names = [index_set.name for index_set in myvar.index_set().subsets()]
            if len(index_names) > 1:
                df.index = pd.MultiIndex.from_tuples(df.index, names=index_names)
            else:
                df.index.name = index_names[0]
            # if index_names is not None:
            #    print(myvar.name, index_names)
            #    df.index = pd.MultiIndex.from_tuples(df.index, names=index_names)

            # ignore NA values
            df = df.dropna()
            if df.empty:
                all_values[myvar.name] = None
                continue

            all_values[myvar.name] = df
        return all_values

    def grid_data_result(self, all_var_values):
        """Create grid data representing optimisation results"""
        years = list(all_var_values["v_investment_cost"].index)
        grid_data = self.grid_data
        nodes = grid_data.node.copy()
        branches = grid_data.branch.copy()
        generators = grid_data.generator.copy()
        consumers = grid_data.consumer.copy()
        is_expanded = all_var_values["v_branch_new_cables"].clip(upper=1).unstack("s_period")
        new_branch_cap = is_expanded * all_var_values["v_branch_new_capacity"].unstack("s_period")
        new_node_cap = all_var_values["v_node_new_capacity"].unstack("s_period")
        for y in years:
            nodes[f"capacity_{y}"] = nodes[f"capacity_{y}"] + new_node_cap[y]
            branches[f"capacity_{y}"] = branches[f"capacity_{y}"] + new_branch_cap[y]
            # mean absolute flow:
            branches[f"flow_{y}"] = (
                (
                    all_var_values["v_branch_flow12"].unstack("s_period")
                    + all_var_values["v_branch_flow21"].unstack("s_period")
                )[y]
                .unstack("s_time")
                .mean(axis=1)
            )
            # mean directional flow:
            branches[f"flow12_{y}"] = (
                all_var_values["v_branch_flow12"].unstack("s_period")[y].unstack("s_time").mean(axis=1)
            )
            branches[f"flow21_{y}"] = (
                all_var_values["v_branch_flow21"].unstack("s_period")[y].unstack("s_time").mean(axis=1)
            )
            generators[f"output_{y}"] = (
                all_var_values["v_generation"].unstack("s_period")[y].unstack("s_time").mean(axis=1)
            )
        grid_res = powergim.grid_data.GridData(years, nodes, branches, generators, consumers)
        return grid_res

    def model_info(self):
        """Return info about model as dictionary"""
        model = self
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
