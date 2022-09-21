

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
- A MILP solver, e.g. the free [CBC solver](https://projects.coin-or.org/Cbc)
Clone or download the code and install it as a python package. 

### Install dependencies
1. `git clone git@github.com:powergim/powergim.git`
2. `cd powergim`
3. `poetry install --no-root`  --no-root to not install the package itself, only the dependencies.
4. `poetry shell`
5. `poetry run pytest tests`


### GitHub Actions Pipelines
These pipelines are defined:

1. Build: Building and testing on multiple OS and python versions. Triggered on any push to GitHub.
3. Release: Create release based on tags starting on v*.
4. Publish: Publish the package to PyPi when a release is marked as published.

## Contribute
You are welcome to contribute to the improvement of the code.

* Use Issues to describe and track needed improvements and bug fixes
* Use branches for development and pull requests to merge into main
* Use [Pre-commit hooks](https://pre-commit.com/)

### Contact

[Harald G Svendsen](https://www.sintef.no/en/all-employees/employee/?empid=3414)  
SINTEF Energy Research
