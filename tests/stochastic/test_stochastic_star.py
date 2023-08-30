import os
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest
import starcase


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_stochastic_star_ef():
    main_ef, all_var_values = starcase.solve_ef("cbc", solver_io="nl")

    print(f"EF objective: {pyo.value(main_ef.EF_Obj)}")
    assert pyo.value(main_ef.EF_Obj) == pytest.approx(117.84348e9)
    assert all_var_values["scen0"]["OBJ"] == pytest.approx(118.38123e9)
    assert all_var_values["scen0"]["scen0.v_investment_cost"][0] == pytest.approx(16.46419e9)
    assert all_var_values["scen0"]["scen0.v_investment_cost"][10] == pytest.approx(6.46752e9)


# TODO: Understand why this test fails (but not glpk, or mpi ones)
# @pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
@pytest.mark.skip(reason="CBC returns error on this case")
def test_stochastic_star_ph_cbc(tmp_path):
    starcase.TMP_PATH = tmp_path
    ph, df_res = starcase.solve_ph("cbc")
    assert ph is not None
    assert isinstance(df_res, pd.DataFrame)


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_stochastic_star_ph_glpk(tmp_path):
    starcase.TMP_PATH = tmp_path
    ph, df_res = starcase.solve_ph("glpk")
    assert ph is not None
    assert isinstance(df_res, pd.DataFrame)


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_stochastic_star_benders(tmp_path):
    starcase.TMP_PATH = tmp_path
    ls, result = starcase.solve_benders("cbc")
    # variables = ls.gather_var_values_to_rank0()
    assert ls is not None
    assert result is not None


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_stochastic_star_mpi_glpk(tmp_path):
    mpiexec_arg = ""
    progname = Path(__file__).absolute().parent / "starcase.py"
    np = 4  # need a licence per progrm (only have 1 for gurobi)
    argstring = f"{tmp_path} glpk"
    runstring_ph = f"mpiexec {mpiexec_arg} -np {np} python -m mpi4py {progname} {argstring} --with-display-progress"

    print(runstring_ph)
    exit_code_ph = os.system(runstring_ph)
    print("Exit code {}".format(exit_code_ph))

    for scenario_name in [f"scen{k}" for k in range(starcase.NUM_SCENARIOS)]:
        df_scen = pd.read_csv(tmp_path / f"ph_res_ALL_{scenario_name}.csv")
        assert isinstance(df_scen, pd.DataFrame)


# @pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
@pytest.mark.skip(reason="CBC returns error on this case")
def test_stochastic_star_mpi_cbc(tmp_path):
    mpiexec_arg = ""
    progname = Path(__file__).absolute().parent / "starcase.py"
    np = 4  # need a licence per progrm (only have 1 for gurobi)
    argstring = f"{tmp_path} cbc"
    runstring_ph = f"mpiexec {mpiexec_arg} -np {np} python -m mpi4py {progname} {argstring} --with-display-progress"

    print(runstring_ph)
    exit_code_ph = os.system(runstring_ph)
    print("Exit code {}".format(exit_code_ph))

    for scenario_name in [f"scen{k}" for k in range(starcase.NUM_SCENARIOS)]:
        df_scen = pd.read_csv(tmp_path / f"ph_res_ALL_{scenario_name}.csv")
        assert isinstance(df_scen, pd.DataFrame)
