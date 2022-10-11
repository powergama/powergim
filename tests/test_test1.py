import pyomo.environ as pyo
import pytest

import powergim
from powergim import testcases


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_case_N5():
    years = [0, 10, 20]
    number_nodes = 5
    number_timesteps = 2
    grid_data, parameter_data = testcases.create_case_star(years, number_nodes, number_timesteps, base_MW=2000)
    sip = powergim.SipModel(grid_data, parameter_data)
    opt = pyo.SolverFactory("glpk")
    results = opt.solve(
        sip,
        tee=False,
        keepfiles=False,
        symbolic_solver_labels=True,
    )
    assert pyo.check_optimal_termination(results)

    # print(pyo.value(sip.OBJ))
    assert pyo.value(sip.OBJ) == pytest.approx(181.7401e9)

    # Optimal variable values
    all_var_values = sip.extract_all_variable_values()
    assert all_var_values["v_new_nodes"].sum() == 5
    print(all_var_values["v_branch_new_capacity"][4])
    assert all_var_values["v_branch_new_capacity"][4][0] == pytest.approx(0)
    assert all_var_values["v_branch_new_capacity"][4][10] == pytest.approx(3000)
    assert all_var_values["v_branch_new_capacity"][4][20] == pytest.approx(3000)
    assert (all_var_values["v_load_shed"] == 0).all()


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_case_N4():
    years = [0, 10]
    number_nodes = 4
    number_timesteps = 10
    grid_data, parameter_data = testcases.create_case_star(years, number_nodes, number_timesteps, base_MW=2000)
    sip = powergim.SipModel(grid_data, parameter_data)
    opt = pyo.SolverFactory("glpk")
    results = opt.solve(
        sip,
        tee=False,
        keepfiles=False,
        symbolic_solver_labels=True,
    )
    assert pyo.check_optimal_termination(results)

    # print(pyo.value(sip.OBJ))
    assert pyo.value(sip.OBJ) == pytest.approx(113.77779e9)

    # Optimal variable values
    all_var_values = sip.extract_all_variable_values()
    assert all_var_values["v_new_nodes"].sum() == 3
    assert all_var_values["v_branch_new_capacity"].sum() == 7000
    assert (all_var_values["v_branch_flow12"][0][0] == 0).all()
    assert (all_var_values["v_branch_flow12"][0][10] == 3000).all()
    assert (all_var_values["v_load_shed"] == 0).all()


if __name__ == "__main__":
    test_case_N5()
    test_case_N4()
