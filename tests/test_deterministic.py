from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

import powergim as pgim

TEST_DATA_ROOT_PATH = Path(__file__).parent / "test_data"
NUMERIC_THRESHOLD = 1e-3


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_deterministic():
    # Read input data
    parameter_data = pgim.file_io.read_parameters(TEST_DATA_ROOT_PATH / "parameters.yaml")
    grid_data = pgim.file_io.read_grid(
        investment_years=parameter_data["parameters"]["investment_years"],
        nodes=TEST_DATA_ROOT_PATH / "nodes.csv",
        branches=TEST_DATA_ROOT_PATH / "branches.csv",
        generators=TEST_DATA_ROOT_PATH / "generators.csv",
        consumers=TEST_DATA_ROOT_PATH / "consumers.csv",
    )
    file_timeseries_sample = TEST_DATA_ROOT_PATH / "time_series_sample.csv"
    grid_data.profiles = pgim.file_io.read_profiles(filename=file_timeseries_sample)

    # TODO: Set temporarily to reproduce previous result:
    grid_data.branch.loc[:, "max_newCap"] = 5000

    sip = pgim.SipModel(grid_data=grid_data, parameter_data=parameter_data)
    grid_data.branch["dist_computed"] = grid_data.compute_branch_distances()

    # Fixme: Works with glpk, but not with cbc
    # opt = pyo.SolverFactory("cbc", solver_io="nl")
    opt = pyo.SolverFactory("glpk")
    results = opt.solve(
        sip,
        tee=False,
        keepfiles=False,
        symbolic_solver_labels=True,
    )
    assert results["Solver"][0]["Status"] == "ok"

    # Optimal variable values
    all_var_values = sip.extract_all_variable_values()

    # Check results are as expected
    print(f"Objective = {pyo.value(sip.OBJ)}")
    # print(all_var_values.keys())
    assert pyo.value(sip.OBJ) == pytest.approx(15189.51e3)
    assert all_var_values["v_investment_cost"][2025] == pytest.approx(18.54166e3)
    assert all_var_values["v_investment_cost"][2028] == pytest.approx(27.54888e3)

    expected_branch_new_capacity = pd.read_csv(
        TEST_DATA_ROOT_PATH / "expected_branch_new_capacity.csv", index_col=["s_branch", "s_period"]
    ).squeeze("columns")
    assert ((all_var_values["v_branch_new_capacity"] - expected_branch_new_capacity).abs() < NUMERIC_THRESHOLD).all()

    expected_branch_new_cables = pd.read_csv(
        TEST_DATA_ROOT_PATH / "expected_branch_new_cables.csv", index_col=["s_branch", "s_period"]
    ).squeeze("columns")
    assert ((all_var_values["v_branch_new_cables"] - expected_branch_new_cables).abs() < NUMERIC_THRESHOLD).all()

    # Fixme: Branch flows are different for different solvers...
    # expected_branch_flow12 = pd.read_csv(
    #    TEST_DATA_ROOT_PATH / "expected_branch_flow12_glpk.csv", index_col=["s_branch", "s_period", "s_time"]
    # ).squeeze("columns")
    # pd.testing.assert_series_equal(all_var_values["v_branch_flow12"], expected_branch_flow12, atol=0.1)
    # assert ((all_var_values["v_branch_flow12"] - expected_branch_flow12).abs() < NUMERIC_THRESHOLD).all()


if __name__ == "__main__":
    test_deterministic()
