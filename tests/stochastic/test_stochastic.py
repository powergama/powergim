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
    grid_data = pgim.file_io.read_grid(
        nodes=TEST_DATA_ROOT_PATH / "nodes.csv",
        branches=TEST_DATA_ROOT_PATH / "branches.csv",
        generators=TEST_DATA_ROOT_PATH / "generators.csv",
        consumers=TEST_DATA_ROOT_PATH / "consumers.csv",
    )
    parameter_data = pgim.file_io.read_parameters(TEST_DATA_ROOT_PATH / "parameters.yaml")
    file_timeseries_sample = TEST_DATA_ROOT_PATH / "time_series_sample.csv"
    grid_data.profiles = pgim.file_io.read_profiles(filename=file_timeseries_sample)

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
    sip = pgim.SipModel()
    # grid_data_res = {}
    for scen in mpisppy.utils.sputils.ef_scenarios(main_ef):
        scen_name = scen[0]
        this_scen = scen[1]
        all_var_values = sip.extract_all_variable_values(this_scen)
        print(f"{scen_name}: OBJ = {pyo.value(this_scen.OBJ)}")
        print(f"{scen_name}: opCost = {all_var_values[f'{scen_name}.opCost'].values}")

    # sputils.ef_nonants_csv(main_ef, "sns_results_ef.csv")
    # sputils.ef_ROOT_nonants_npy_serializer(main_ef, "sns_root_nonants.npy")
    print(f"EF objective: {pyo.value(main_ef.EF_Obj)}")

    assert pyo.value(main_ef.EF_Obj) == pytest.approx(131457566096)
    # assert all_var_values["scen2.opCost"][1] == pytest.approx(2.0442991e10)
    # assert all_var_values["scen2.opCost"][2] == pytest.approx(5.3318421e10)


def test_stochastic_ph():
    ph = northsea.solve_ph("cbc")

    assert ph is not None

    assert 1 == 1


def test_stochastic_ph_mpi():
    mpiexec_arg = ""
    progname = "northsea.py"
    np = 4  # need a licence per progrm (only have 1 for gurobi)
    argstring = "ph"
    runstring_ph = f"mpiexec {mpiexec_arg} -np {np} python -m mpi4py {progname} {argstring} --with-display-progress"

    print(runstring_ph)
    exit_code_ph = os.system(runstring_ph)
    print("Exit code {}".format(exit_code_ph))

    for scenario_name in [f"scen{k}" for k in range(northsea.NUM_SCENARIOS)]:
        df_scen = pd.read_csv(f"ph_res_ALL_{scenario_name}.csv")

        assert isinstance(df_scen, pd.DataFrame)

    assert 1 == 1


if __name__ == "__main__":
    test_stochastic_ef()
    test_stochastic_ph()
    test_stochastic_ph_mpi()
