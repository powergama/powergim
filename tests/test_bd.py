import pytest
from powergim.dcmp_tools import run_case_bd
from datetime import datetime
from pathlib import Path
import pyomo.environ as pyo
import math

INPUT_OPTIONS = {'PATH_INPUT': Path(__file__).parents[1] / "examples"/ "inputs"/"CASE_BASELINE",
                 'PATH_OUTPUT': Path(__file__).parents[1] / "examples"/ "outputs",
                 'BRANCHES_FILE_NAME': "branches_reduced.csv",
                 'GRID_CASE': 'star',
                 'IS_STOCHASTIC': False,
                 'N_SAMPLES': 2,
                 'BENDERS_SINGLE_SUB': False,
                 'BENDERS_SINGLE_CUT': False,
                 'MH_USE_BIN_EXPANS': False,
                 'MH_USE_FIXED_CAP': False,
                 'LP_SOLVER': 'appsi_highs', # normally 'gurobi', set to 'appsi_highs' for testing
                 'DO_CALC_VSS_EVPI': True,
                 'DO_CNVRG_PLOT': False,
                 'DO_VIZ': False}

@pytest.mark.parametrize("stochastic_case",[False,True]) # tests both determinsitic and stochastic cases.
@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_run_case_bd(capsys,stochastic_case):
    
    test_input_options = INPUT_OPTIONS.copy()
    test_input_options["IS_STOCHASTIC"] = stochastic_case

    pgim_obj_val, mh_obj_val, bd_obj_val, pgim_opex, bd_opex = run_case_bd(datetime.now(),test_input_options) # Call the function

    _ = capsys.readouterr() # Suppress the output by capturing it without using it

    assert math.isclose(pgim_obj_val,mh_obj_val, rel_tol=0.01), "PGIM and MH objective values do not match"
    assert math.isclose(pgim_obj_val,bd_obj_val, rel_tol=0.01), "PGIM and BD objective values do not match"  
    assert math.isclose(pgim_opex,bd_opex, rel_tol=0.01), "PGIM and BD operational costs do not match"