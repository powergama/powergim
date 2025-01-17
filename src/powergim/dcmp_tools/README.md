# dcmp_tools

---

This subpackage provides additional utilities and unctionalities for solving `powergim` instances using decomposition (_dcmp_).

## Contents

- `decomposition.py`: Holds the main classes for the _multi-horizon_ reformulation of a `powegim` instance and the implementation of the supported _decomposition_ methods (_Benders_, _Dantzig-Wolfe_).
- `utils_spi.py`: Holds classes and functions that extend `powegim` functionalities.
- `evpi.py`: Holds function to calculate the _expected value of perfect infomration (evpi)_.
- `vss.py`: Holds function to calculate the _value of stochastic solution (vss)_.
- `results_plots.py`: Holds helper functions for case and results visualization.
  - _extends the use of `powegim` method `plot_map2` to be applied for multi-horizon and stochastic cases_
  - _introduces web-based, interactive maps for case and result exploration._

---

## Usage

```python
import powergim.dcmp_tools
```

---

## Examples

You can find example scripts [here](../../../examples/).

- `validation_Benders.py` creates a `powegim` case and solves it with all 3 methods:

  - **pgim** (default),
  - **mh** (_multi-horizon_ reformulation)
  - **bd** (using the developed _Bender's_ decompostion).

  The methods are then validated against each other to check proper execution.

- `validation_Benders_parallel.py` is similar to `validation_Benders.py` but showcases a parallel computing implmentation based on `MPI`.

- `validation_DantzigWolfe.py` is similar to `validation_Benders.py` but instead of the **bd** method, it showcases the **dw** (_Dantzig-Wolfe_) method.
  _Important!: **dw** is to be used only for **deterministic** problems._
