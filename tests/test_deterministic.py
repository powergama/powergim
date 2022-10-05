from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

import powergim as pgim

TEST_DATA_ROOT_PATH = Path(__file__).parent / "test_data"
NUMERIC_THRESHOLD = 1e-3


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
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

    # Prepare model
    sip = pgim.SipModel(grid_data=grid_data, parameter_data=parameter_data)
    # dict_data = sip.createModelData(
    #    grid_data, parameter_data, maxNewBranchNum=5, maxNewBranchCap=5000, maxNewGenCap=5000
    # )
    # model = sip.createConcreteModel(dict_data)
    grid_data.branch["dist_computed"] = grid_data.compute_branch_distances()

    # Solve using external solver - this may take some time.
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
    print(all_var_values.keys())

    assert all_var_values["v_investment_cost"][2025] == 18.541664000e9
    assert all_var_values["v_investment_cost"][2028] == 25.829794000e9

    expected_branchNewCapacity = pd.read_csv(
        TEST_DATA_ROOT_PATH / "expected_branchNewCapacity.csv", index_col=["BRANCH", "STAGE"]
    ).squeeze("columns")
    assert ((all_var_values["branchNewCapacity"] - expected_branchNewCapacity).abs() < NUMERIC_THRESHOLD).all()

    expected_branchNewCables = pd.read_csv(
        TEST_DATA_ROOT_PATH / "expected_branchNewCables.csv", index_col=["BRANCH", "STAGE"]
    ).squeeze("columns")
    assert ((all_var_values["branchNewCables"] - expected_branchNewCables).abs() < NUMERIC_THRESHOLD).all()

    expected_branchFlow12 = pd.read_csv(
        TEST_DATA_ROOT_PATH / "expected_branchFlow12.csv", index_col=["BRANCH", "TIME", "STAGE"]
    ).squeeze("columns")
    assert ((all_var_values["branchFlow12"] - expected_branchFlow12).abs() < NUMERIC_THRESHOLD).all()


if __name__ == "__main__":
    test_deterministic()
