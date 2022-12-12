import pandas as pd
import pyomo.environ as pyo

import powergim


def extract_costs(sip: powergim.SipModel, npv=True):
    """Extract investment and operating cost information from solved SIP model"""
    if npv:
        npv_func = sip.npvInvestment
    else:
        # just return the investment without any discounting etc.
        def npv_func(x, y):
            return y

    # Investment costs:
    df_branch_cost = pd.DataFrame()
    df_node_cost = pd.DataFrame()
    df_gen_cost = pd.DataFrame()
    for branch in sip.s_branch:
        for period in sip.s_period:
            investment = pyo.value(sip.costBranch(branch, period))
            # df_branch_cost.loc[branch,f"base_{period}"]=investment
            df_branch_cost.loc[branch, period] = npv_func(period, investment)
    for node in sip.s_node:
        for period in sip.s_period:
            investment = pyo.value(sip.costNode(node, period))
            df_node_cost.loc[node, period] = npv_func(period, investment)
    for gen in sip.s_gen:
        for period in sip.s_period:
            investment = pyo.value(sip.costGen(gen, period))
            df_gen_cost.loc[gen, period] = npv_func(period, investment)

    # Operating costs:
    df_gen_opcost = pd.DataFrame()
    for gen in sip.s_gen:
        for period in sip.s_period:
            df_gen_opcost.loc[gen, period] = pyo.value(sip.costOperationSingleGen(gen, period))
    # df_sum_opcost = pd.Series({period:pyo.value(sip.costOperation(period)) for period in sip.s_period})
    return {"branches": df_branch_cost, "nodes": df_node_cost, "generators": df_gen_cost, "gencost": df_gen_opcost}
