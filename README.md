
[![GitHub license](https://img.shields.io/github/license/powergama/powergim)](https://github.com/powergama/powergim/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)](https://python.org)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/powergama/powergim)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/powergama/powergim/CI%20Build)
![GitHub branch checks state](https://img.shields.io/github/checks-status/powergama/powergim/main)

# Power Grid Investment Module (PowerGIM)

PowerGIM is a Python package for stochastic power system expansion planning that can consider both transmission and generator investments in a two-stage formulation with uncertain parameters.


## Getting started
Install latest PowerGIM release from PyPi:
```
pip install powergim
```



## User guide and examples
The online user guide  gives more information about how to
specify input data and run a simulation case.

*  [User guide](docs/powergim.md)


## Local installation
Prerequisite: 
- [Poetry](https://python-poetry.org/docs/#installation)
- [Pre-commit](https://pre-commit.com/)
- A MILP solver, e.g. the free [CBC solver](https://projects.coin-or.org/Cbc).
Clone or download the code and install it as a python package. 
- A working MPI implementation, preferably supporting MPI-3 and built with shared/dynamic libraries

### Install dependencies
1. `git clone git@github.com:powergim/powergim.git`
2. `cd powergim`
3. `poetry install --no-root`  --no-root to not install the package itself, only the dependencies.
4. `poetry shell`
5. `poetry run pytest tests`


### GitHub Actions Pipelines
These pipelines are defined:

1. Build: Building and testing on multiple OS and python versions. Triggered on any push to GitHub.

## Contribute
You are welcome to contribute to the improvement of the code.

* Use Issues to describe and track needed improvements and bug fixes
* Use branches for development and pull requests to merge into main
* Use [Pre-commit hooks](https://pre-commit.com/)

### Contact

[Harald G Svendsen](https://www.sintef.no/en/all-employees/employee/?empid=3414)  
SINTEF Energy Research
