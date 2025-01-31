# dcmp_tools

---

This subpackage provides additional utilities and functionalities for solving `powergim` instances using decomposition (_dcmp_).

## Contents

- `decomposition.py`: Holds the main classes for the _multi-horizon_ reformulation of a `powegim` instance and the implementation of the supported _decomposition_ methods (_Benders_ (**bd**), _Dantzig-Wolfe_ (**dw**)).
- `bd.py`: Holds the main interface functions for **bd**.
- `utils_spi.py`: Holds classes and functions that extend `powegim` functionalities.
- `evpi.py`: Holds function to calculate the _expected value of perfect infomration (evpi)_.
- `vss.py`: Holds function to calculate the _value of stochastic solution (vss)_.
- `results_plots.py`: Holds helper functions for case and results visualization.
  - _extends the use of `powegim` method `plot_map2` to be applied for multi-horizon and stochastic cases_
  - _introduces web-based, interactive maps for case and result exploration._

---

## Usage

The main interface for solving a case using Bender's decomposition (**bd**) is implemented through `run_case_bd()`, as show below. Case options are specified modifying the `INPUT_OPTIONS`.

```python
import powergim.dcmp_tools
from datetime import datetime
from pathlib import Path

INPUT_OPTIONS = {'PATH_INPUT': Path(__file__).parents[0] / "inputs"/"CASE_BASELINE",
                 'PATH_OUTPUT': Path(__file__).parents[0] / "outputs",
                 'BRANCHES_FILE_NAME': "branches_reduced.csv",
                 'GRID_CASE': 'star',
                 'IS_STOCHASTIC': True,
                 'N_SAMPLES': 2,
                 'BENDERS_SINGLE_SUB': False,
                 'BENDERS_SINGLE_CUT': False,
                 'MH_USE_BIN_EXPANS': False,
                 'MH_USE_FIXED_CAP': False,
                 'LP_SOLVER': 'gurobi',
                 'DO_CALC_VSS_EVPI': True,
                 'DO_CNVRG_PLOT': False,
                 'DO_VIZ': False}

powergim.dcmp_tools.run_case_bd(datetime.now(),INPUT_OPTIONS)
```

_<u>**Options explained**</u>_

- IS_STOCHASTIC: If 'True', solves the 'stochastic' problem and if 'False' solves the 'deterministic' one.
- N_SAMPLES: For demonstration, use 2 samples. For paper results use 300 ('baseline' case).
- BENDERS_SINGLE_SUB: If True, Benders uses the non-decomposed (single) sub-problem. If False Benders uses the decomposed subproblems.
- BENDERS_SINGLE_CUT: If True, Benders uses a single-cut algorithm. If False Benders uses the multi-cut version.
- MH_USE_BIN_EXPANS: If True, integers are modelled using their binary expansion instead (recommend to keep 'False').
- MH_USE_FIXED_CAP: If True, branches are of fixed, specified capacity. If True, branch capacity is a variable (keep 'False' for paper results).
- LP_SOLVER: Solver for LP(s): 'gurobi' or 'appsi_highs'.
- DO_CALC_VSS_EVPI: Activates the calculation of VSS and EVPI for the stochastic cases.
- DO_VIZ: Activates the visualization of results (interactive html maps and paper figures).

_<u>**Important note**</u>_:

- _The user interface for the parallel implementation of **bd** is not implemented yet.Use [this](../../../examples/validation_Benders_parallel.py) example script instead._
- _The user interface for Dantzig-Wolfe is not implemented yet. Use [this](../../../examples/validation_DantzigWolfe.py) example script instead._

---

## Examples

More detailed scripts showcasing various functionalities of `dcmp_tools` can be found [here](../../../examples/).

- `validation_Benders.py` creates a `powegim` case and solves it with all 3 methods:

  - **pgim** (default),
  - **mh** (_multi-horizon_ reformulation)
  - **bd** (using the developed _Bender's_ decompostion).

  The methods are then validated against each other to check proper execution.

- `validation_Benders_parallel.py` is similar to `validation_Benders.py` but showcases a parallel computing implmentation based on `MPI`.

- `validation_DantzigWolfe.py` is similar to `validation_Benders.py` but instead of the **bd** method, it showcases the **dw** (_Dantzig-Wolfe_) method.
  _Important!: **dw** is to be used only for **deterministic** problems._
