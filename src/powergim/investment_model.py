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


class SipModel:
    """
    Power Grid Investment Module - Stochastic Investment Problem
    """

    _NUMERICAL_THRESHOLD_ZERO = 1e-6
    _HOURS_PER_YEAR = 8760

    def __init__(self, M_const=1000):
        """Create Abstract Pyomo model for PowerGIM

        Parameters
        ----------
        M_const : int
            large constant
        """
        self.abstractmodel = self._createAbstractModel()
        self.M_const = M_const

    def costNode(self, model, n, stage):
        """Expression for cost of node, investment cost no discounting"""
        n_cost = 0
        var_num = model.newNodes
        N = model.nodeOffshore[n]
        n_cost += N * (model.nodetypeCost[model.nodeType[n], "S"] * var_num[n, stage])
        n_cost += (1 - N) * (model.nodetypeCost[model.nodeType[n], "L"] * var_num[n, stage])
        return model.nodeCostScale[n] * n_cost

    def costBranch(self, model, b, stage):
        """Expression for cost of branch, investment cost no discounting"""
        b_cost = 0

        var_num = model.branchNewCables
        var_cap = model.branchNewCapacity
        typ = model.branchType[b]
        b_cost += model.branchtypeCost[typ, "B"] * var_num[b, stage]
        b_cost += model.branchtypeCost[typ, "Bd"] * model.branchDistance[b] * var_num[b, stage]
        b_cost += model.branchtypeCost[typ, "Bdp"] * model.branchDistance[b] * var_cap[b, stage]

        # endpoints offshore (N=1) or onshore (N=0) ?
        N1 = model.branchOffshoreFrom[b]
        N2 = model.branchOffshoreTo[b]
        for N in [N1, N2]:
            b_cost += N * (
                model.branchtypeCost[typ, "CS"] * var_num[b, stage]
                + model.branchtypeCost[typ, "CSp"] * var_cap[b, stage]
            )
            b_cost += (1 - N) * (
                model.branchtypeCost[typ, "CL"] * var_num[b, stage]
                + model.branchtypeCost[typ, "CLp"] * var_cap[b, stage]
            )

        return model.branchCostScale[b] * b_cost

    def costGen(self, model, g, stage):
        """Expression for cost of generator, investment cost no discounting"""
        g_cost = 0
        var_cap = model.genNewCapacity
        typ = model.genType[g]
        g_cost += model.genTypeCost[typ] * var_cap[g, stage]
        return model.genCostScale[g] * g_cost

    def npvInvestment(self, model, stage, investment, includeOM=True, subtractSalvage=True):
        """NPV of investment cost including lifetime O&M and salvage value

        Parameters
        ----------
        model : object
            Pyomo model
        stage : int
            Investment or operation stage (1 or 2)
        investment :
            cost of e.g. node, branch or gen
        """
        omfactor = 0
        salvagefactor = 0
        if subtractSalvage:
            salvagefactor = (int(stage - 1) * model.stage2TimeDelta / model.financeYears) * (
                1 / ((1 + model.financeInterestrate) ** (model.financeYears - model.stage2TimeDelta * int(stage - 1)))
            )
        if includeOM:
            omfactor = model.omRate * (
                annuityfactor(model.financeInterestrate, model.financeYears)
                - annuityfactor(model.financeInterestrate, int(stage - 1) * model.stage2TimeDelta)
            )

        # discount costs that come in stage 2 (the future)
        # present value vs future value: pv = fv/(1+r)^n
        discount_t0 = 1 / ((1 + model.financeInterestrate) ** (model.stage2TimeDelta * int(stage - 1)))

        investment = investment * discount_t0
        pv_cost = investment * (1 + omfactor - salvagefactor)
        return pv_cost

    def costInvestments(self, model, stage, includeOM=True, subtractSalvage=True):
        """Investment cost, including lifetime O&M costs (NPV)"""
        investment = 0
        # add branch, node and generator investment costs:
        for b in model.BRANCH:
            investment += self.costBranch(model, b, stage)
        for n in model.NODE:
            investment += self.costNode(model, n, stage)
        for g in model.GEN:
            investment += self.costGen(model, g, stage)
        # add O&M costs and compute NPV:
        cost = self.npvInvestment(model, stage, investment, includeOM, subtractSalvage)
        return cost

    def costOperation(self, model, stage):
        """Operational costs: cost of gen, load shed (NPV)"""
        opcost = 0
        # discount_t0 = (1/((1+model.financeInterestrate)
        #    **(model.stage2TimeDelta*int(stage-1))))

        # operation cost per year:
        opcost = sum(
            model.generation[i, t, stage]
            * model.samplefactor[t]
            * (
                model.genCostAvg[i] * model.genCostProfile[i, t]
                + model.genTypeEmissionRate[model.genType[i]] * model.CO2price
            )
            for i in model.GEN
            for t in model.TIME
        )
        opcost += sum(
            model.loadShed[c, t, stage] * model.VOLL * model.samplefactor[t] for c in model.LOAD for t in model.TIME
        )

        # compute present value of future annual costs
        if stage == len(model.STAGE):
            # from year stage2TimeDelta until financeYears
            opcost = opcost * (
                annuityfactor(model.financeInterestrate, model.financeYears)
                - annuityfactor(model.financeInterestrate, int(stage - 1) * model.stage2TimeDelta)
            )
        else:
            # from year 0
            opcost = opcost * annuityfactor(model.financeInterestrate, model.stage2TimeDelta)

        # Harald: this is already discounted back to year 0 from the present
        # value calculation above
        # opcost = opcost*discount_t0

        return opcost

    def costOperationSingleGen(self, model, g, stage):
        """Operational costs: cost of gen, load shed (NPV)"""
        opcost = 0
        # discount_t0 = (1/((1+model.financeInterestrate)
        #    **(model.stage2TimeDelta*int(stage-1))))

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

    def _createAbstractModel(self):
        model = pyo.AbstractModel()
        model.name = "PowerGIM abstract model"

        # SETS ###############################################################

        model.NODE = pyo.Set()
        model.GEN = pyo.Set()
        model.BRANCH = pyo.Set()
        model.LOAD = pyo.Set()
        model.AREA = pyo.Set()
        model.TIME = pyo.Set()
        model.STAGE = pyo.Set()

        # A set for each stage i.e. a list with two sets
        model.NODE_EXPAND1 = pyo.Set()
        model.NODE_EXPAND2 = pyo.Set()
        model.BRANCH_EXPAND1 = pyo.Set()
        model.BRANCH_EXPAND2 = pyo.Set()
        model.GEN_EXPAND1 = pyo.Set()
        model.GEN_EXPAND2 = pyo.Set()

        model.BRANCHTYPE = pyo.Set()
        model.BRANCHCOSTITEM = pyo.Set(initialize=["B", "Bd", "Bdp", "CLp", "CL", "CSp", "CS"])
        model.NODETYPE = pyo.Set()
        model.NODECOSTITEM = pyo.Set(initialize=["L", "S"])
        model.LINEAR = pyo.Set(initialize=["fix", "slope"])

        model.GENTYPE = pyo.Set()

        # PARAMETERS #########################################################
        model.samplefactor = pyo.Param(model.TIME, within=pyo.NonNegativeReals)
        model.financeInterestrate = pyo.Param(within=pyo.Reals)
        model.financeYears = pyo.Param(within=pyo.Reals)
        model.omRate = pyo.Param(within=pyo.Reals)
        model.CO2price = pyo.Param(within=pyo.NonNegativeReals)
        model.VOLL = pyo.Param(within=pyo.NonNegativeReals)
        model.stage2TimeDelta = pyo.Param(within=pyo.NonNegativeReals)
        model.maxNewBranchNum = pyo.Param(within=pyo.NonNegativeReals)

        # investment costs and limits:
        model.branchtypeMaxCapacity = pyo.Param(model.BRANCHTYPE, within=pyo.Reals)
        model.branchMaxNewCapacity = pyo.Param(model.BRANCH, within=pyo.Reals)
        model.branchtypeCost = pyo.Param(model.BRANCHTYPE, model.BRANCHCOSTITEM, within=pyo.Reals)
        model.branchLossfactor = pyo.Param(model.BRANCHTYPE, model.LINEAR, within=pyo.Reals)
        model.nodetypeCost = pyo.Param(model.NODETYPE, model.NODECOSTITEM, within=pyo.Reals)
        model.genTypeCost = pyo.Param(model.GENTYPE, within=pyo.Reals)
        model.nodeCostScale = pyo.Param(model.NODE, within=pyo.Reals)
        model.branchCostScale = pyo.Param(model.BRANCH, within=pyo.Reals)
        model.genCostScale = pyo.Param(model.GEN, within=pyo.Reals)
        model.genNewCapMax = pyo.Param(model.GEN, within=pyo.Reals)

        # branches:
        model.branchExistingCapacity = pyo.Param(model.BRANCH, within=pyo.NonNegativeReals)
        model.branchExistingCapacity2 = pyo.Param(model.BRANCH, within=pyo.NonNegativeReals)
        model.branchExpand = pyo.Param(model.BRANCH, within=pyo.Binary)
        model.branchExpand2 = pyo.Param(model.BRANCH, within=pyo.Binary)
        model.branchDistance = pyo.Param(model.BRANCH, within=pyo.NonNegativeReals)
        model.branchType = pyo.Param(model.BRANCH, within=model.BRANCHTYPE)
        model.branchOffshoreFrom = pyo.Param(model.BRANCH, within=pyo.Binary)
        model.branchOffshoreTo = pyo.Param(model.BRANCH, within=pyo.Binary)

        # nodes:
        model.nodeExistingNumber = pyo.Param(model.NODE, within=pyo.NonNegativeIntegers)
        model.nodeOffshore = pyo.Param(model.NODE, within=pyo.Binary)
        model.nodeType = pyo.Param(model.NODE, within=model.NODETYPE)

        # generators
        model.genCostAvg = pyo.Param(model.GEN, within=pyo.Reals)
        model.genCostProfile = pyo.Param(model.GEN, model.TIME, within=pyo.Reals)
        model.genCapacity = pyo.Param(model.GEN, within=pyo.Reals)
        model.genCapacity2 = pyo.Param(model.GEN, within=pyo.Reals)
        model.genCapacityProfile = pyo.Param(model.GEN, model.TIME, within=pyo.Reals)
        model.genPAvg = pyo.Param(model.GEN, within=pyo.Reals)
        model.genType = pyo.Param(model.GEN, within=model.GENTYPE)
        model.genExpand = pyo.Param(model.GEN, within=pyo.Binary)
        model.genExpand2 = pyo.Param(model.GEN, within=pyo.Binary)
        model.genTypeEmissionRate = pyo.Param(model.GENTYPE, within=pyo.Reals)

        # helpers:
        model.genNode = pyo.Param(model.GEN, within=model.NODE)
        model.demNode = pyo.Param(model.LOAD, within=model.NODE)
        model.branchNodeFrom = pyo.Param(model.BRANCH, within=model.NODE)
        model.branchNodeTo = pyo.Param(model.BRANCH, within=model.NODE)
        model.nodeArea = pyo.Param(model.NODE, within=model.AREA)
        model.coeff_B = pyo.Param(model.NODE, model.NODE, within=pyo.Reals)
        model.coeff_DA = pyo.Param(model.BRANCH, model.NODE, within=pyo.Reals)

        # consumers
        # the split int an average value, and a profile is to make it easier
        # to generate scenarios (can keep profile, but adjust demandAvg)
        model.demandAvg = pyo.Param(model.LOAD, within=pyo.Reals)
        model.demandProfile = pyo.Param(model.LOAD, model.TIME, within=pyo.Reals)
        model.emissionCap = pyo.Param(model.LOAD, within=pyo.NonNegativeReals)
        model.maxShed = pyo.Param(model.LOAD, model.TIME, within=pyo.NonNegativeReals)

        # VARIABLES ##########################################################

        # investment: new branch capacity
        def branchNewCapacity_bounds(model, j, h):
            if h > 1:
                return (0, model.branchMaxNewCapacity[j] * model.branchExpand2[j])
            else:
                return (0, model.branchMaxNewCapacity[j] * model.branchExpand[j])

        model.branchNewCapacity = pyo.Var(
            model.BRANCH,
            model.STAGE,
            within=pyo.NonNegativeReals,
            bounds=branchNewCapacity_bounds,
        )

        # investment: new branch cables
        def branchNewCables_bounds(model, j, h):
            if h > 1:
                return (0, model.maxNewBranchNum * model.branchExpand2[j])
            else:
                return (0, model.maxNewBranchNum * model.branchExpand[j])

        model.branchNewCables = pyo.Var(
            model.BRANCH,
            model.STAGE,
            within=pyo.NonNegativeIntegers,
            bounds=branchNewCables_bounds,
        )

        # investment: new nodes
        model.newNodes = pyo.Var(model.NODE, model.STAGE, within=pyo.Binary)

        # investment: generation capacity
        def genNewCapacity_bounds(model, g, h):
            if h > 1:
                return (0, model.genNewCapMax[g] * model.genExpand2[g])
            else:
                return (0, model.genNewCapMax[g] * model.genExpand[g])

        model.genNewCapacity = pyo.Var(
            model.GEN, model.STAGE, within=pyo.NonNegativeReals, bounds=genNewCapacity_bounds, initialize=0
        )

        # branch power flow (also given by constraints??)
        def branchFlow_bounds(model, j, t, h):
            if h == 1:
                ub = model.branchExistingCapacity[j] + branchNewCapacity_bounds(model, j, h)[1]
            elif h == 2:
                ub = (
                    model.branchExistingCapacity[j]
                    + model.branchExistingCapacity2[j]
                    + branchNewCapacity_bounds(model, j, h - 1)[1]
                    + branchNewCapacity_bounds(model, j, h)[1]
                )
            return (0, ub)

        model.branchFlow12 = pyo.Var(
            model.BRANCH,
            model.TIME,
            model.STAGE,
            within=pyo.NonNegativeReals,
            bounds=branchFlow_bounds,
        )
        model.branchFlow21 = pyo.Var(
            model.BRANCH,
            model.TIME,
            model.STAGE,
            within=pyo.NonNegativeReals,
            bounds=branchFlow_bounds,
        )

        # generator output (bounds set by constraint)
        model.generation = pyo.Var(model.GEN, model.TIME, model.STAGE, within=pyo.NonNegativeReals)

        # load shedding
        def loadShed_bounds(model, c, t, h):
            ub = model.maxShed[c, t]
            # ub = 0
            # for c in model.LOAD:
            #    if model.demNode[c]==n:
            #        ub += model.demandAvg[c]*model.demandProfile[c,t]
            return (0, ub)

        model.loadShed = pyo.Var(
            model.LOAD,
            model.TIME,
            model.STAGE,
            domain=pyo.NonNegativeReals,
            bounds=loadShed_bounds,
        )

        # CONSTRAINTS ########################################################

        # Power flow limitations (in both directions)
        def maxflow12_rule(model, j, t, h):
            cap = model.branchExistingCapacity[j]
            if h > 1:
                cap += model.branchExistingCapacity2[j]
            for x in range(h):
                cap += model.branchNewCapacity[j, x + 1]
            expr = model.branchFlow12[j, t, h] <= cap
            return expr

        def maxflow21_rule(model, j, t, h):
            cap = model.branchExistingCapacity[j]
            if h > 1:
                cap += model.branchExistingCapacity2[j]
            for x in range(h):
                cap += model.branchNewCapacity[j, x + 1]
            expr = model.branchFlow21[j, t, h] <= cap
            return expr

        model.cMaxFlow12 = pyo.Constraint(model.BRANCH, model.TIME, model.STAGE, rule=maxflow12_rule)
        model.cMaxFlow21 = pyo.Constraint(model.BRANCH, model.TIME, model.STAGE, rule=maxflow21_rule)

        # No new branch capacity without new cables
        def maxNewCap_rule(model, j, h):
            typ = model.branchType[j]
            expr = model.branchNewCapacity[j, h] <= model.branchtypeMaxCapacity[typ] * model.branchNewCables[j, h]
            return expr

        model.cmaxNewCapacity = pyo.Constraint(model.BRANCH, model.STAGE, rule=maxNewCap_rule)

        # A node required at each branch endpoint
        def newNodes_rule(model, n, h):
            expr = 0
            numnodes = model.nodeExistingNumber[n]
            for x in range(h):
                numnodes += model.newNodes[n, x + 1]
            for j in model.BRANCH:
                if model.branchNodeFrom[j] == n or model.branchNodeTo[j] == n:
                    expr += model.branchNewCables[j, h]
            expr = expr <= self.M_const * numnodes
            if (type(expr) is bool) and (expr is True):
                expr = pyo.Constraint.Skip
            return expr

        model.cNewNodes = pyo.Constraint(model.NODE, model.STAGE, rule=newNodes_rule)

        # Generator output limitations
        # TODO: add option to set minimum output = timeseries for renewable,
        # i.e. disallov curtaliment (could be global parameter)
        def maxPgen_rule(model, g, t, h):
            cap = model.genCapacity[g]
            if h > 1:
                cap += model.genCapacity2[g]
            for x in range(h):
                if (g in model.GEN_EXPAND1) or (g in model.GEN_EXPAND2):
                    cap += model.genNewCapacity[g, x + 1]
            allowCurtailment = True
            # TODO: make this limit a parameter (global or per generator?)
            #            if model.genCostAvg[g]*model.genCostProfile[g,t]<1:
            if model.genCostAvg[g] < 1:
                # don't allow curtailment of cheap generators (renewables)
                allowCurtailment = False
            if allowCurtailment:
                expr = model.generation[g, t, h] <= (model.genCapacityProfile[g, t] * cap)
            else:
                # don't allow curtailment of generator output
                expr = model.generation[g, t, h] == (model.genCapacityProfile[g, t] * cap)

            return expr

        model.cMaxPgen = pyo.Constraint(model.GEN, model.TIME, model.STAGE, rule=maxPgen_rule)

        # Generator maximum average output (energy sum)
        # (e.g. for hydro with storage)
        def maxEnergy_rule(model, g, h):
            cap = model.genCapacity[g]
            if h > 1:
                cap += model.genCapacity2[g]
            for x in range(h):
                cap += model.genNewCapacity[g, x + 1]
            if model.genPAvg[g] > 0:
                expr = sum(model.generation[g, t, h] for t in model.TIME) <= (model.genPAvg[g] * cap * len(model.TIME))
            else:
                expr = pyo.Constraint.Skip
            return expr

        model.cMaxEnergy = pyo.Constraint(model.GEN, model.STAGE, rule=maxEnergy_rule)

        # Emissions restriction per country/load
        # TODO: deal with situation when no emission cap has been given (-1)
        def emissionCap_rule(model, a, h):
            if model.CO2price > 0:
                expr = 0
                for n in model.NODE:
                    if model.nodeArea[n] == a:
                        expr += sum(
                            model.generation[g, t, h]
                            * model.genTypeEmissionRate[model.genType[g]]
                            * model.samplefactor[t]
                            for t in model.TIME
                            for g in model.GEN
                            if model.genNode[g] == n
                        )
                expr = expr <= sum(model.emissionCap[c] for c in model.LOAD if model.nodeArea[model.demNode[c]] == a)
            else:
                expr = pyo.Constraint.Skip
            return expr

        model.cEmissionCap = pyo.Constraint(model.AREA, model.STAGE, rule=emissionCap_rule)

        # Power balance in nodes : gen+demand+flow into node=0
        def powerbalance_rule(model, n, t, h):
            expr = 0
            # flow of power into node (subtrating losses)
            for j in model.BRANCH:
                if model.branchNodeFrom[j] == n:
                    # branch out of node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += -model.branchFlow12[j, t, h]
                    expr += model.branchFlow21[j, t, h] * (
                        1 - (model.branchLossfactor[typ, "fix"] + model.branchLossfactor[typ, "slope"] * dist)
                    )
                if model.branchNodeTo[j] == n:
                    # branch into node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += model.branchFlow12[j, t, h] * (
                        1 - (model.branchLossfactor[typ, "fix"] + model.branchLossfactor[typ, "slope"] * dist)
                    )
                    expr += -model.branchFlow21[j, t, h]

            # generated power
            for g in model.GEN:
                if model.genNode[g] == n:
                    expr += model.generation[g, t, h]

            # load shedding
            for c in model.LOAD:
                if model.demNode[c] == n:
                    expr += model.loadShed[c, t, h]

            # consumed power
            for c in model.LOAD:
                if model.demNode[c] == n:
                    expr += -model.demandAvg[c] * model.demandProfile[c, t]

            expr = expr == 0

            if (type(expr) is bool) and (expr is True):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr

        model.cPowerbalance = pyo.Constraint(model.NODE, model.TIME, model.STAGE, rule=powerbalance_rule)

        # OBJECTIVE ##############################################################
        model.investmentCost = pyo.Var(model.STAGE, within=pyo.Reals)
        model.opCost = pyo.Var(model.STAGE, within=pyo.Reals)

        def investmentCost_rule(model, stage):
            """Investment cost, including lifetime O&M costs (NPV)"""
            expr = self.costInvestments(model, stage)
            return model.investmentCost[stage] == expr

        model.cInvestmentCost = pyo.Constraint(model.STAGE, rule=investmentCost_rule)

        def opCost_rule(model, stage):
            """Operational costs: cost of gen, load shed (NPV)"""
            opcost = self.costOperation(model, stage)
            return model.opCost[stage] == opcost

        model.cOperationalCosts = pyo.Constraint(model.STAGE, rule=opCost_rule)

        def total_Cost_Objective_rule(model):
            investment = pyo.summation(model.investmentCost)
            operation = pyo.summation(model.opCost)
            return investment + operation

        model.OBJ = pyo.Objective(rule=total_Cost_Objective_rule, sense=pyo.minimize)

        return model

    def _offshoreBranch(self, grid_data):
        """find out whether branch endpoints are offshore or onshore

        Returns 1 for offshore and 0 for onsore from/to endpoints
        """
        d = {"from": [], "to": []}

        d["from"] = [
            grid_data.node[grid_data.node["id"] == n]["offshore"].tolist()[0] for n in grid_data.branch["node_from"]
        ]
        d["to"] = [
            grid_data.node[grid_data.node["id"] == n]["offshore"].tolist()[0] for n in grid_data.branch["node_to"]
        ]
        return d

    def createConcreteModel(self, dict_data):
        """Create Concrete Pyomo model for PowerGIM

        Parameters
        ----------
        dict_data : dictionary
            dictionary containing the model data. This can be created with
            the createModelData(...) method

        Returns
        -------
            Concrete pyomo model
        """

        concretemodel = self.abstractmodel.create_instance(data=dict_data, name="PowerGIM Model", namespace="powergim")
        return concretemodel

    def createModelData(self, grid_data, parameter_data, maxNewBranchNum, maxNewBranchCap):
        """Create model data in dictionary format

        Parameters
        ----------
        grid_data : powergama.GridData object
            contains grid model
        parameter_data : dict
            dictionary containing parameters (read from YAML file)
        maxNewBranchNum : int
            upper limit on parallel branches to consider (e.g. 10)
        maxNewBranchCap : float (MW)
            upper limit on new capacity to consider (e.g. 10000)

        Returns
        --------
        dictionary with pyomo data (in pyomo format)
        """

        branch_distances = grid_data.compute_branch_distances()
        timerange = range(grid_data.profiles.shape[0])

        # to see how the data format is:
        # data = pyo.DataPortal(model=self.abstractmodel)
        # data.load(filename=datafile)

        di = {}
        # Sets:
        di["NODE"] = {None: grid_data.node["id"].tolist()}
        di["BRANCH"] = {None: grid_data.branch.index.tolist()}
        di["GEN"] = {None: grid_data.generator.index.tolist()}
        di["LOAD"] = {None: grid_data.consumer.index.tolist()}
        di["AREA"] = {None: list(grid_data.node["area"].unique())}
        di["TIME"] = {None: timerange}
        # di['STAGE'] = {None: grid_data.branch.expand[grid_data.branch['expand']>0].unique().tolist()}

        br_expand1 = grid_data.branch[grid_data.branch["expand"] == 1].index.tolist()
        br_expand2 = grid_data.branch[grid_data.branch["expand"] == 2].index.tolist()
        gen_expand1 = grid_data.generator[grid_data.generator["expand"] == 1].index.tolist()
        gen_expand2 = grid_data.generator[grid_data.generator["expand"] == 2].index.tolist()
        # Convert from numpy.int64 (pandas) to int in order to work with PySP
        # (pprint function error otherwise)
        br_expand1 = [int(i) for i in br_expand1]
        br_expand2 = [int(i) for i in br_expand2]
        gen_expand1 = [int(i) for i in gen_expand1]
        gen_expand2 = [int(i) for i in gen_expand2]
        # Determine which nodes should be considered upgraded in each stage,
        # depending on whether any generators or branches are connected
        node_expand1 = []
        node_expand2 = []
        for n in grid_data.node["id"][grid_data.node["existing"] == 0]:
            if (
                n in grid_data.generator["node"][grid_data.generator["expand"] == 1].tolist()
                or n in grid_data.branch["node_to"][grid_data.branch["expand"] == 1].tolist()
                or n in grid_data.branch["node_from"][grid_data.branch["expand"] == 1].tolist()
            ):
                # stage one generator  or branch expansion connected to node
                node_expand1.append(n)
            if (
                n in grid_data.generator["node"][grid_data.generator["expand"] == 2].tolist()
                or n in grid_data.branch["node_to"][grid_data.branch["expand"] == 2].tolist()
                or n in grid_data.branch["node_from"][grid_data.branch["expand"] == 2].tolist()
            ):
                # stage two generator or branch expansion connected to node
                node_expand2.append(n)
        #        node_expand1 = grid_data.node[
        #                        grid_data.node['expand1']==1].index.tolist()
        #        node_expand2 = grid_data.node[
        #                        grid_data.node['expand2']==2].index.tolist()

        di["BRANCH_EXPAND1"] = {None: br_expand1}
        di["BRANCH_EXPAND2"] = {None: br_expand2}
        di["GEN_EXPAND1"] = {None: gen_expand1}
        di["GEN_EXPAND2"] = {None: gen_expand2}
        di["NODE_EXPAND1"] = {None: node_expand1}
        di["NODE_EXPAND2"] = {None: node_expand2}

        # Parameters:
        di["maxNewBranchNum"] = {None: maxNewBranchNum}
        di["samplefactor"] = {}
        if hasattr(grid_data.profiles, "frequency"):
            di["samplefactor"] = grid_data.profiles["frequency"]
        else:
            for t in timerange:
                di["samplefactor"][t] = self._HOURS_PER_YEAR / len(timerange)
        di["nodeOffshore"] = {}
        di["nodeType"] = {}
        di["nodeExistingNumber"] = {}
        di["nodeCostScale"] = {}
        di["nodeArea"] = {}
        for k, row in grid_data.node.iterrows():
            n = grid_data.node["id"][k]
            # n=grid_data.node.index[k] #or simply =k
            di["nodeOffshore"][n] = row["offshore"]
            di["nodeType"][n] = row["type"]
            di["nodeExistingNumber"][n] = row["existing"]
            di["nodeCostScale"][n] = row["cost_scaling"]
            di["nodeArea"][n] = row["area"]

        di["branchExistingCapacity"] = {}
        di["branchExistingCapacity2"] = {}
        di["branchExpand"] = {}
        di["branchExpand2"] = {}
        di["branchDistance"] = {}
        di["branchType"] = {}
        di["branchCostScale"] = {}
        di["branchOffshoreFrom"] = {}
        di["branchOffshoreTo"] = {}
        di["branchNodeFrom"] = {}
        di["branchNodeTo"] = {}
        di["branchMaxNewCapacity"] = {}
        offsh = self._offshoreBranch(grid_data)
        for k, row in grid_data.branch.iterrows():
            di["branchExistingCapacity"][k] = row["capacity"]
            di["branchExistingCapacity2"][k] = row["capacity2"]
            if row["max_newCap"] > 0:
                di["branchMaxNewCapacity"][k] = row["max_newCap"]
            else:
                di["branchMaxNewCapacity"][k] = maxNewBranchCap
            di["branchExpand"][k] = row["expand"]
            di["branchExpand2"][k] = row["expand2"]
            if row["distance"] >= 0:
                di["branchDistance"][k] = row["distance"]
            else:
                di["branchDistance"][k] = branch_distances[k]
            di["branchType"][k] = row["type"]
            di["branchCostScale"][k] = row["cost_scaling"]
            di["branchOffshoreFrom"][k] = offsh["from"][k]
            di["branchOffshoreTo"][k] = offsh["to"][k]
            di["branchNodeFrom"][k] = row["node_from"]
            di["branchNodeTo"][k] = row["node_to"]

        di["genCapacity"] = {}
        di["genCapacity2"] = {}
        di["genCapacityProfile"] = {}
        di["genNode"] = {}
        di["genCostAvg"] = {}
        di["genCostProfile"] = {}
        di["genPAvg"] = {}
        di["genExpand"] = {}
        di["genExpand2"] = {}
        di["genNewCapMax"] = {}
        di["genType"] = {}
        di["genCostScale"] = {}
        for k, row in grid_data.generator.iterrows():
            di["genCapacity"][k] = row["pmax"]
            di["genCapacity2"][k] = row["pmax2"]
            di["genNode"][k] = row["node"]
            di["genCostAvg"][k] = row["fuelcost"]
            di["genPAvg"][k] = row["pavg"]
            di["genExpand"][k] = row["expand"]
            di["genExpand2"][k] = row["expand2"]
            di["genNewCapMax"][k] = row["p_maxNew"]
            di["genType"][k] = row["type"]
            di["genCostScale"][k] = row["cost_scaling"]
            ref = row["fuelcost_ref"]
            ref2 = row["inflow_ref"]
            for i, t in enumerate(timerange):
                di["genCostProfile"][(k, t)] = grid_data.profiles[ref][i]
                di["genCapacityProfile"][(k, t)] = grid_data.profiles[ref2][i] * row["inflow_fac"]

        di["demandAvg"] = {}
        di["demandProfile"] = {}
        di["demNode"] = {}
        di["emissionCap"] = {}
        di["maxShed"] = {}
        for k, row in grid_data.consumer.iterrows():
            di["demNode"][k] = row["node"]
            di["demandAvg"][k] = row["demand_avg"]
            di["emissionCap"][k] = row["emission_cap"]
            ref = row["demand_ref"]
            for i, t in enumerate(timerange):
                di["demandProfile"][(k, t)] = grid_data.profiles[ref][i]
                # if profile is negative, maxShed should be zero (not negative)
                di["maxShed"][(k, t)] = max(0, grid_data.profiles[ref][i]) * row["demand_avg"]

        # Parameters coming from YAML file

        di["NODETYPE"] = {None: parameter_data["nodetype"].keys()}
        di["nodetypeCost"] = {}
        for name, item in parameter_data["nodetype"].items():
            di["nodetypeCost"][(name, "L")] = float(item["L"])
            di["nodetypeCost"][(name, "S")] = float(item["S"])

        di["BRANCHTYPE"] = {None: parameter_data["branchtype"].keys()}
        di["branchtypeCost"] = {}
        di["branchtypeMaxCapacity"] = {}
        di["branchLossfactor"] = {}
        for name, item in parameter_data["branchtype"].items():
            di["branchtypeCost"][(name, "B")] = float(item["B"])
            di["branchtypeCost"][(name, "Bd")] = float(item["Bd"])
            di["branchtypeCost"][(name, "Bdp")] = float(item["Bdp"])
            di["branchtypeCost"][(name, "CL")] = float(item["CL"])
            di["branchtypeCost"][(name, "CLp")] = float(item["CLp"])
            di["branchtypeCost"][(name, "CS")] = float(item["CS"])
            di["branchtypeCost"][(name, "CSp")] = float(item["CSp"])
            di["branchtypeMaxCapacity"][name] = float(item["maxCap"])
            di["branchLossfactor"][(name, "fix")] = float(item["lossFix"])
            di["branchLossfactor"][(name, "slope")] = float(item["lossSlope"])

        di["GENTYPE"] = {None: parameter_data["gentype"].keys()}
        di["genTypeCost"] = {}
        di["genTypeEmissionRate"] = {}
        for name, item in parameter_data["gentype"].items():
            di["genTypeCost"][name] = float(item["CX"])
            di["genTypeEmissionRate"][name] = float(item["CO2"])

        # OTHER PARAMETERS:
        item = parameter_data["parameters"]
        di["financeInterestrate"] = {None: float(item["financeInterestrate"])}
        di["financeYears"] = {None: float(item["financeYears"])}
        di["omRate"] = {None: float(item["omRate"])}
        di["CO2price"] = {None: float(item["CO2price"])}
        di["VOLL"] = {None: float(item["VOLL"])}
        di["stage2TimeDelta"] = {None: float(item["stage2TimeDelta"])}
        di["STAGE"] = {None: list(range(1, int(item["stages"]) + 1))}

        return {"powergim": di}

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
            scen_name_list=None,  # Deprecated?
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

    def extract_all_variable_values(self, model):
        """Extract variable values and return as a dictionary of pandas milti-index series"""
        all_values = {}
        all_obj = model.component_objects(ctype=pyo.Var)
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
