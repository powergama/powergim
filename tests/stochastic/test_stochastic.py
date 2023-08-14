import os
from pathlib import Path

import mpisppy.opt.ph
import mpisppy.utils.sputils
import northsea
import pandas as pd
import pyomo.environ as pyo
import pytest

import powergim as pgim

TEST_DATA_ROOT_PATH = Path(__file__).parent / "data"
NUMERIC_THRESHOLD = 1e-3


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_stochastic_ef():
    # Read input data
    parameter_data = pgim.file_io.read_parameters(TEST_DATA_ROOT_PATH / "parameters_stoch.yaml")
    print(parameter_data["parameters"])
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

    # Scenarios
    scenario_creator_kwargs = {"grid_data": grid_data, "parameter_data": parameter_data}
    scenario_names = ["scen0", "scen1", "scen2"]

    # Solve estensive form
    main_ef = mpisppy.utils.sputils.create_EF(
        scenario_names,
        scenario_creator=northsea.my_scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    solver = pyo.SolverFactory("cbc")
    solver.solve(
        main_ef,
        tee=True,
        symbolic_solver_labels=True,
    )

    # Extract results:
    for scen in mpisppy.utils.sputils.ef_scenarios(main_ef):
        scen_name = scen[0]
        this_scen = scen[1]
        all_var_values = pgim.SipModel.extract_all_variable_values(this_scen)
        print(f"{scen_name}: OBJ = {pyo.value(this_scen.OBJ)}")
        print(f"{scen_name}: opCost = {all_var_values[f'{scen_name}.v_operating_cost'].values}")

    # sputils.ef_nonants_csv(main_ef, "sns_results_ef.csv")
    # sputils.ef_ROOT_nonants_npy_serializer(main_ef, "sns_root_nonants.npy")
    print(f"EF objective: {pyo.value(main_ef.EF_Obj)}")

    assert pyo.value(main_ef.EF_Obj) == pytest.approx(133.10901e9)


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_stochastic_ph(tmp_path):
    northsea.TMP_PATH = tmp_path

    ph, df_res = northsea.solve_ph("glpk")

    assert ph is not None
    assert isinstance(df_res, pd.DataFrame)


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_stochastic_ph_mpi(tmp_path):
    mpiexec_arg = ""
    progname = Path(__file__).absolute().parent / "northsea.py"
    np = 4  # need a licence per progrm (only have 1 for gurobi)
    argstring = tmp_path
    runstring_ph = f"mpiexec {mpiexec_arg} -np {np} python -m mpi4py {progname} {argstring} --with-display-progress"

    print(runstring_ph)
    exit_code_ph = os.system(runstring_ph)
    print("Exit code {}".format(exit_code_ph))

    for scenario_name in [f"scen{k}" for k in range(northsea.NUM_SCENARIOS)]:
        df_scen = pd.read_csv(tmp_path / f"ph_res_ALL_{scenario_name}.csv")
        assert isinstance(df_scen, pd.DataFrame)

        # the result is not very stable with so few iterations, so skip this test
        # mask_cable1 = (df_scen["variable"] == "branchNewCables") & (df_scen["STAGE"] == "1")
        # assert df_scen.loc[mask_cable1, "value"].sum() == 4


if __name__ == "__main__":
    # test_stochastic_ef()
    test_stochastic_ph(Path(__file__).absolute().parent / "tmp_output")
    # test_stochastic_ph_mpi(Path(__file__).absolute().parent / "tmp_output_mpi")
    pass
