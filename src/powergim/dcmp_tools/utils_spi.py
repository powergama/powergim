"""
Utility functions/classes definitions:
- `extract_all_opt_variable_values()`
- `TimeseriesBuilder`
- `ReferenceModelCreator`
- `ParamsExtractor`
- `CaseMapVisualizer`
---
@author: spyridonc
"""
import os
import random
import webbrowser
from itertools import product
from operator import itemgetter
from pathlib import Path

try:
    import folium
    import folium.plugins
except ImportError:
    ImportWarning("Could not import folium.")
import numpy as np
import pandas as pd
import pyomo.core.expr as EXPR
import pyomo.environ as pyo

import powergim as pgim
from powergim.testcases import create_case_star


# ---------FUNCTION TO EXTRACT ALL OPTIMIZATION VARIABLES AFTER A SOLUTION (copied & modified from `pgim` method).
def extract_all_opt_variable_values(model):
    """Extract variable values and return as a dictionary of pandas multi-index series"""
    all_values = {}
    all_obj = model.component_objects(ctype=pyo.Var)
    for myvar in all_obj:
        var_values = myvar.get_values()
        if not var_values:
            # empty dictionary, so no variables to store
            all_values[myvar.name] = None
            continue
        # This creates a pandas.Series:
        df = pd.DataFrame.from_dict(var_values, orient="index", columns=["value"])["value"]
        index_names = [index_set.name for index_set in myvar.index_set().subsets()]
        if len(index_names) > 1:
            df.index = pd.MultiIndex.from_tuples(df.index, names=index_names)
        else:
            df.index.name = index_names[0]

        # ignore NA values
        df = df.dropna()
        if df.empty:
            all_values[myvar.name] = None
            continue

        all_values[myvar.name] = df
    return all_values


# -------------------------------PGIM TIME SERIES GENERATION CLASS----------------------------------------
class TimeseriesBuilder:
    """
    Generate 1-year long timeseries `.csv` file (including both load and RES), to be used for a pgim case (possibly after sampling).
    """

    def __init__(
        self,
        PGIM_INPUT_FILES_PATH: Path,
        data_year={"res_ts": "2009", "load_ts": "2020"},
        ninja_token=None,
        entsoe_key=None,
        custom_ts_data_path=None,
    ):

        self.data_year = data_year

        self.TS_PATH = PGIM_INPUT_FILES_PATH / "time_series_year.csv"

        if custom_ts_data_path is not None:
            self.TS_DATA_PATH = custom_ts_data_path
        else:
            self.TS_DATA_PATH = (
                PGIM_INPUT_FILES_PATH.parents[2] / "timeseries_data"
            )  # standard directory structure: "timeseries_data", "inputs", "outputs" at same hierarchy level.
        os.makedirs(self.TS_DATA_PATH, exist_ok=True)

        self.pgim_ts = {
            "windoff": {
                "ts_path": self.TS_DATA_PATH / Path("wind_power_offshore_" + self.data_year["res_ts"] + ".csv")
            },
            "solar": {"ts_path": self.TS_DATA_PATH / Path("solar_power_" + self.data_year["res_ts"] + ".csv")},
            "windon": {"ts_path": self.TS_DATA_PATH / Path("wind_power_onshore_" + self.data_year["res_ts"] + ".csv")},
            "consumption": {"ts_path": self.TS_DATA_PATH / Path("load_" + self.data_year["load_ts"] + ".csv")},
        }
        self.ninja_token = ninja_token
        self.entsoe_key = entsoe_key

    def create_csv(self):
        """
        Check if the timeseries CSV for powergim exists, and if not, create it.
        """

        mask_existing_ts = [
            self.pgim_ts["windoff"]["ts_path"].is_file(),
            self.pgim_ts["solar"]["ts_path"].is_file(),
            self.pgim_ts["windon"]["ts_path"].is_file(),
            self.pgim_ts["consumption"]["ts_path"].is_file(),
        ]

        if not self.TS_PATH.is_file():
            # Create the ts sample from the exisitng timeseries.
            if all(mask_existing_ts):
                # Read the ts data from the origin csv files.
                self.pgim_ts["windoff"].update({"data": pd.read_csv(self.pgim_ts["windoff"]["ts_path"], index_col=0)})
                self.pgim_ts["solar"].update({"data": pd.read_csv(self.pgim_ts["solar"]["ts_path"], index_col=0)})
                self.pgim_ts["windon"].update({"data": pd.read_csv(self.pgim_ts["windon"]["ts_path"], index_col=0)})
                self.pgim_ts["consumption"].update(
                    {"data": pd.read_csv(self.pgim_ts["consumption"]["ts_path"], index_col=0)}
                )
                self.pgim_ts["consumption"]["data"].fillna(0)

                self.pgim_ts["windoff"]["data"].columns = "windoff_" + self.pgim_ts["windoff"]["data"].columns
                self.pgim_ts["solar"]["data"].columns = "solar_" + self.pgim_ts["solar"]["data"].columns
                self.pgim_ts["windon"]["data"].columns = "windon_" + self.pgim_ts["windon"]["data"].columns
                self.pgim_ts["consumption"]["data"].columns = "load_" + self.pgim_ts["consumption"]["data"].columns

                self.pgim_ts["windoff"]["data"].index = pd.to_datetime(self.pgim_ts["windoff"]["data"].index, utc=True)
                self.pgim_ts["solar"]["data"].index = pd.to_datetime(self.pgim_ts["solar"]["data"].index, utc=True)
                self.pgim_ts["windon"]["data"].index = pd.to_datetime(self.pgim_ts["windon"]["data"].index, utc=True)
                self.pgim_ts["consumption"]["data"].index = pd.to_datetime(self.pgim_ts["consumption"]["data"].index)

                # Reset the load index such that it matches the datetime index of windoff
                self.pgim_ts["consumption"]["data"].set_index(self.pgim_ts["windoff"]["data"].index, inplace=True)

                ts_data = pd.concat(
                    [
                        self.pgim_ts["windoff"]["data"],
                        self.pgim_ts["consumption"]["data"],
                        self.pgim_ts["solar"]["data"],
                        self.pgim_ts["windon"]["data"],
                    ],
                    axis=1,
                )  # re-indexing load because of different self.data_years

                ts_year = ts_data

                # Add "constant" profile cloumn
                ts_year["const"] = 1
                ts_year.to_csv(self.TS_PATH, index=True)

                print("\nTimeseries CSV for PowerGIM created.")

            else:
                raise Exception(
                    f"{str([b for a, b in zip(map(lambda _x:not _x ,mask_existing_ts), list(self.pgim_ts.keys())) if a])[1:-1]} timeseries missing. \
                                \n All timeseries are required to create the PowerGIM timeseries CSV."
                )

                # print(f"{str([b for a, b in zip(map(lambda _x:not _x ,mask_existing_ts), list(self.pgim_ts.keys())) if a])[1:-1]} timeseries missing. \
                #     \n All timeseries are required to create the PowerGIM timeseries CSV.\
                #     \n Generating the timeseries that are missing...")

                # self.get_timeseries() # FIXME: To fix the functionality of this method.

        else:
            print("\nTimeseries CSV for PowerGIM exists already.")


# -------------------------------PGIM REFERENCE MODEL CLASS----------------------------------------
class ReferenceModelCreator:  # TODO: In future version attribute names can imitt prefixes/suffixes 'ref' or 'pgim' because they are implicetely implied. See corresponding 'CHECKME' notes.
    """
    Create a reference pgim model for a custom grid representing either the `baseline` or the `star` case.
    """

    def __init__(
        self,
        branches_file_name="branches.csv",
        s_sample_size=5,
        sample_random_state=1,
        investment_periods=[2035, 2040, 2050],
        probabilities={"scen0": 0.1, "scen1": 0.45, "scen2": 0.45},
        grid_case="baseline",
    ):

        self.branch_input_file_name = branches_file_name
        self.s_sample_size = s_sample_size
        self.sample_random_state = sample_random_state
        self.investment_periods = investment_periods

        # Check if the sum is not equal to 1 and raise an exception
        if sum(probabilities.values()) != 1:
            raise ValueError(f"The sum of the probability values is {sum(probabilities.values())}, but it should be 1.")
        else:
            self.probabilities = probabilities

        self.grid_case = grid_case
        if self.grid_case == "star":
            self.s_nodes = 4
            self.randomize_profiles = False

    def create_reference_pgim_model(self, PATH_INPUT, PATH_OUTPUT, time_coupling=False, scenario_selection="default"):
        """Definition of the pgim reference model needed for the multi-horizon formulation.

        Args:
            PATH_INPUT (Path): Define the \\inputs\\CASE_BASELINE path.
            PATH_OUTPUT (Path): Define the \\outputs path.
            time_coupling (bool, optional): Deactivate flex load for multi-horizon and decomposition methods. Defaults to False.
            scenario_selection (str, optional): Select reference scenario. Defaults to 'default'.

        Raises:
            ValueError: Error if `scenario_selection` takes invalid value.
        """
        match self.grid_case:
            case "baseline":
                # Read input grid (data)
                self.ref_params = pgim.file_io.read_parameters(
                    PATH_INPUT / "parameters.yaml"
                )  # CHECKME: `self.ref_params` ---> `self.params`

                # ---------------MODIFY INPUT/PARAMETERS DATA
                # For MH and decompostion methods: No use of flexibility.
                if not time_coupling:
                    self.ref_params["parameters"]["load_flex_shift_frac"] = {2035: 0, 2040: 0, 2050: 0}
                    self.ref_params["parameters"]["load_flex_price_frac"] = {
                        2035: 0,
                        2040: 0,
                        2050: 0,
                    }  # no price-sensitive load

                self.ref_grid_data = pgim.file_io.read_grid(
                    investment_years=self.ref_params["parameters"]["investment_years"],
                    nodes=PATH_INPUT / "nodes.csv",
                    branches=PATH_INPUT / self.branch_input_file_name,
                    generators=PATH_INPUT / "generators.csv",
                    consumers=PATH_INPUT / "consumers.csv",
                )  # CHECKME: `self.ref_grid_data` ---> `self.grid_data`

                # ---------------MODIFY PGIM GRID DATA
                if not time_coupling:
                    self.ref_grid_data.generator[
                        "pavg"
                    ] = 0  # For MH and decompostion methods: No energy constraint on hydro.
                    self.ref_grid_data.node[
                        "cost_scaling"
                    ] = 0  # For MH and decompostion methods: No node expansion cost.

                # CHECKME: I have to manually define the scenarios. Comment/Uncomment the corresponding lines to use the hardcoded scenarios #1, #2 or #3, as examples.
                if scenario_selection == "default":
                    pass
                elif scenario_selection == "EVP":
                    # EXPECTED VALUE PROBLEM
                    demand_scale = (
                        260 * self.probabilities["scen0"]
                        + 220 * self.probabilities["scen1"]
                        + 340 * self.probabilities["scen2"]
                    ) / 260
                    # demand_scale = (260*self.probabilities['scen0']+130*self.probabilities['scen1']+520*self.probabilities['scen2'])/260
                    m_demand = self.ref_grid_data.consumer["node"].str.startswith("NO_")
                    for year in self.ref_params["parameters"]["investment_years"]:

                        # -----Hardcoded scenario #1
                        self.ref_grid_data.consumer.loc[m_demand, f"demand_{year}"] = (
                            demand_scale * self.ref_grid_data.consumer.loc[m_demand, f"demand_{year}"]
                        )

                        # -----Hardcoded scenario #2
                        # self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = \
                        # self.probabilities['scen0']*self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] + \
                        # self.probabilities['scen1']*0.5 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] + \
                        # self.probabilities['scen2']*2 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]

                        # -----Hardcoded scenario #3
                        # self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = \
                        # self.probabilities['scen0']*self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] + \
                        # self.probabilities['scen1']*2 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] + \
                        # self.probabilities['scen2']*0.5 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                elif scenario_selection == "A1":
                    # oceangrid_A1_lessdemand ---> Less NO demand (220 TWh in 2050)
                    demand_scale = 220 / 260
                    # demand_scale = 130/260
                    m_demand = self.ref_grid_data.consumer["node"].str.startswith("NO_")
                    for year in self.ref_params["parameters"]["investment_years"]:

                        # -----Hardcoded scenario #1
                        self.ref_grid_data.consumer.loc[m_demand, f"demand_{year}"] = (
                            demand_scale * self.ref_grid_data.consumer.loc[m_demand, f"demand_{year}"]
                        )

                        # -----Hardcoded scenario #2
                        # self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 0.5 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]

                        # -----Hardcoded scenario #3
                        # self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 2 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                elif scenario_selection == "A2":
                    # oceangrid_A2_moredemand ---> More NO demand (340 TWh in 2050)
                    demand_scale = 340 / 260
                    # demand_scale = 520/260
                    m_demand = self.ref_grid_data.consumer["node"].str.startswith("NO_")
                    for year in self.ref_params["parameters"]["investment_years"]:

                        # -----Hardcoded scenario #1
                        self.ref_grid_data.consumer.loc[m_demand, f"demand_{year}"] = (
                            demand_scale * self.ref_grid_data.consumer.loc[m_demand, f"demand_{year}"]
                        )

                        # -----Hardcoded scenario #2
                        # self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"] = 2 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("DE_"), f"demand_{year}"]

                        # -----Hardcoded scenario #3
                        # self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"] = 0.5 * self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"].str.startswith("UK_"), f"demand_{year}"]

                else:
                    raise ValueError(
                        f"Allowed values for 'scenario_selection': {', '.join(map(str, ['default', 'EVP','A1', 'A2']))}"
                    )

                profiles_year = pgim.file_io.read_profiles(filename=PATH_INPUT / "time_series_year.csv")
                # -------------Sampling from whole year data
                profiles_sample = profiles_year.sample(
                    n=self.s_sample_size, random_state=self.sample_random_state
                ).reset_index()
                profiles_sample.to_csv(PATH_OUTPUT / f"time_series_sample_reduced_{self.sample_random_state}.csv")
                self.ref_grid_data.profiles = profiles_sample

                self.ref_pgim_model = pgim.SipModel(
                    self.ref_grid_data, self.ref_params
                )  # CHECKME: `self.ref_pgim_model` ---> `self.model`

            case "star":
                # Read input grid (data)
                self.ref_grid_data, self.ref_params = create_case_star(
                    investment_years=self.investment_periods,
                    number_nodes=self.s_nodes,
                    number_timesteps=self.s_sample_size,
                    base_MW=200,
                )  # CHECKME: `self.ref_grid_data` ---> `self.grid_data`

                # Node expansion is not considered in the MH fomrulation.
                self.ref_grid_data.node["cost_scaling"] = 0

                # CHECKME: I have to manually define the scenarios. Comment/Uncomment the corresponding lines to use the hardcoded DEFAULT/ALTERNATIVE SCENARIOS.
                if scenario_selection == "default":
                    pass
                elif scenario_selection == "EVP":
                    # EXPECTED VALUE PROBLEM:

                    # ---------DEFAULT SCENARIO

                    # Half the wind at n1 (wind farm node).
                    _init_wind_capacity = self.ref_grid_data.generator.loc[self.ref_grid_data.generator["node"] == "n1"]

                    for iperiod in self.ref_params["parameters"]["investment_years"]:
                        self.ref_grid_data.generator.loc[
                            self.ref_grid_data.generator["node"] == "n1", ["capacity_" + str(iperiod)]
                        ] = (
                            self.probabilities["scen1"] * 0.5 * _init_wind_capacity.loc[0, "capacity_" + str(iperiod)]
                            + (self.probabilities["scen0"] + self.probabilities["scen2"])
                            * _init_wind_capacity.loc[0, "capacity_" + str(iperiod)]
                        )

                    # Twice the load at n3 (offshore load node).
                    _init_load_capacity = self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"] == "n3"]

                    for iperiod in self.ref_params["parameters"]["investment_years"]:
                        self.ref_grid_data.consumer.loc[
                            self.ref_grid_data.consumer["node"] == "n3", ["demand_" + str(iperiod)]
                        ] = (
                            self.probabilities["scen2"] * 2 * _init_load_capacity.loc[1, "demand_" + str(iperiod)]
                            + (self.probabilities["scen0"] + self.probabilities["scen1"])
                            * _init_load_capacity.loc[1, "demand_" + str(iperiod)]
                        )

                    # ---------ALTERNATIVE SCENARIO
                    # _init_load_capacity = self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer['node'] == 'n2']

                    # for iperiod in self.ref_params['parameters']['investment_years']:
                    #     self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = self.probabilities['scen0']*_init_load_capacity.loc[0,'demand_' + str(iperiod)] + self.probabilities['scen1']*0.5*_init_load_capacity.loc[0,'demand_' + str(iperiod)] + self.probabilities['scen2']*2*_init_load_capacity.loc[0,'demand_' + str(iperiod)]

                elif scenario_selection == "A1":
                    # ---------DEFAULT SCENARIO

                    # Half the wind at n1 (wind farm node).
                    _init_wind_capacity = self.ref_grid_data.generator.loc[self.ref_grid_data.generator["node"] == "n1"]

                    for iperiod in self.ref_params["parameters"]["investment_years"]:
                        self.ref_grid_data.generator.loc[
                            self.ref_grid_data.generator["node"] == "n1", ["capacity_" + str(iperiod)]
                        ] = (0.5 * _init_wind_capacity.loc[0, "capacity_" + str(iperiod)])

                    # ---------ALTERNATIVE SCENARIO
                    # Half the load at n2 (country node)
                    # _init_load_capacity = self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer['node'] == 'n2']

                    # for iperiod in self.ref_params['parameters']['investment_years']:
                    #     self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 0.5*_init_load_capacity.loc[0,'demand_' + str(iperiod)]

                elif scenario_selection == "A2":
                    # ---------DEFAULT SCENARIO

                    # Twice the load at n3 (offshore load node) for the last two periods (2040,2050).
                    _init_load_capacity = self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer["node"] == "n3"]

                    for iperiod in self.ref_params["parameters"]["investment_years"]:
                        self.ref_grid_data.consumer.loc[
                            self.ref_grid_data.consumer["node"] == "n3", ["demand_" + str(iperiod)]
                        ] = (2 * _init_load_capacity.loc[1, "demand_" + str(iperiod)])

                    # ---------ALTERNATIVE SCENARIOS
                    # Double the load at n2 (country node)
                    # _init_load_capacity = self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer['node'] == 'n2']

                    # for iperiod in self.ref_params['parameters']['investment_years']:
                    #     self.ref_grid_data.consumer.loc[self.ref_grid_data.consumer['node'] == 'n2', ['demand_' + str(iperiod)]] = 2*_init_load_capacity.loc[0,'demand_' + str(iperiod)]

                if self.randomize_profiles:
                    random.seed(0)
                    self.ref_grid_data.profiles["demand1"] = [
                        random.random() for _ in range(self.s_sample_size)  # nosec B311 - skip bandit issue
                    ]
                    random.seed(1)
                    self.ref_grid_data.profiles["inflow_gentype0"] = [
                        random.random() for _ in range(self.s_sample_size)  # nosec B311 - skip bandit issue
                    ]

                self.ref_pgim_model = pgim.SipModel(
                    self.ref_grid_data, self.ref_params
                )  # CHECKME: `self.ref_pgim_model` ---> `self.model`

            case _:
                return f"Invalid case specification. Choose from: {', '.join(map(str, ['baseline', 'star']))}."


# -------------------------------PGIM PARAMETERS EXTRACTOR CLASS----------------------------------------
class ParamsExtractor:
    """
    Extract pgim model parameters, i.e., variables coefficients, upper/lower limits and cost function coefficients.
    """

    def __init__(self, pgim_model=None):
        if pgim_model:
            self.model = pgim_model

            self.all_contraints = list(self.model.component_data_objects(pyo.Constraint))
            self.all_sets = list(self.model.component_data_objects(pyo.Set))
            self.all_vars = list(self.model.component_data_objects(pyo.Var))

            self.my_sets = {}
            for iSet in self.all_sets:
                if (
                    (iSet.name == "s_branch")
                    or (iSet.name == "s_period")
                    or (iSet.name == "s_time")
                    or (iSet.name == "s_gen")
                    or (iSet.name == "s_load")
                ):
                    self.my_sets[iSet.name] = list(iSet.ordered_data())
        else:
            raise ValueError("A valid pgim model was not provided.")

    def get_max_number_cables(self):

        _specific_contraint_name = "c_max_number_cables"

        _multi_index = pd.MultiIndex.from_tuples(
            product(self.my_sets["s_branch"], self.my_sets["s_period"]), names=["branch", "period"]
        )

        _temp_df_data = []
        _specific_contraints = [
            cnstr for cnstr in self.all_contraints if cnstr.name.startswith(_specific_contraint_name)
        ]

        for cnstr in _specific_contraints:
            cnstr_expr = cnstr.expr

            # Extract coefficients from the constraint expression
            for i, term in enumerate(cnstr_expr.args):

                if isinstance(term, np.integer) or isinstance(term, np.floating) or isinstance(term, int):
                    _temp_df_data.append(term)
                else:
                    if not term.is_variable_type():
                        _temp_df_data.append(EXPR.decompose_term(term)[1][0][0])

        _df = pd.DataFrame(_temp_df_data, columns=["parameter_value"], index=_multi_index)

        return _df

    def get_max_new_branch_capacity(self):

        _specific_contraint_name = "c_max_new_branch_capacity"

        _multi_index = pd.MultiIndex.from_tuples(
            product(self.my_sets["s_branch"], self.my_sets["s_period"]), names=["branch", "period"]
        )

        _temp_df_data = []
        _specific_contraints = [
            cnstr for cnstr in self.all_contraints if cnstr.name.startswith(_specific_contraint_name)
        ]

        for cnstr in _specific_contraints:
            cnstr_expr = cnstr.expr

            # Extract coefficients from the constraint expression
            for _, term in enumerate(cnstr_expr.args):

                if isinstance(term, np.integer) or isinstance(term, np.floating) or isinstance(term, int):
                    _temp_df_data.append(term)
                else:
                    if not term.is_variable_type():
                        _temp_df_data.append(EXPR.decompose_term(term)[1][0][0])

        _df = pd.DataFrame(_temp_df_data, columns=["parameter_value"], index=_multi_index)

        return _df

    def get_max_gen_power(self):

        _specific_contraint_name = "c_max_gen_power"

        _multi_index = pd.MultiIndex.from_tuples(
            product(self.my_sets["s_gen"], self.my_sets["s_period"], self.my_sets["s_time"]),
            names=["gen", "period", "time"],
        )

        _temp_df_data = []
        _specific_contraints = [
            cnstr for cnstr in self.all_contraints if cnstr.name.startswith(_specific_contraint_name)
        ]

        for cnstr in _specific_contraints:
            cnstr_expr = cnstr.expr

            # Extract coefficients from the constraint expression
            for _, term in enumerate(cnstr_expr.args):

                if isinstance(term, np.integer) or isinstance(term, np.floating) or isinstance(term, int):
                    _temp_df_data.append(term)
                else:
                    if not term.is_variable_type():
                        _temp_df_data.append(EXPR.decompose_term(term)[1][0][0])

        _df = pd.DataFrame(_temp_df_data, columns=["parameter_value"], index=_multi_index)

        return _df

    def get_max_load(self):

        _specific_variable_name = "v_load_shed"

        _multi_index = pd.MultiIndex.from_tuples(
            product(self.my_sets["s_load"], self.my_sets["s_period"], self.my_sets["s_time"]),
            names=["load", "period", "time"],
        )

        _temp_df_data = []

        _specific_variables = [var for var in self.all_vars if var.name.startswith(_specific_variable_name)]

        for var in _specific_variables:
            _temp_df_data.append(var.ub)

        _df = pd.DataFrame(_temp_df_data, columns=["parameter_value"], index=_multi_index)

        return _df

    def get_investment_cost_coefs(self):

        _temp_dict = {}
        _specific_contraint_name = "c_investment_cost"

        _multi_index = pd.MultiIndex.from_tuples(
            product(self.my_sets["s_branch"], self.my_sets["s_period"]), names=["branch", "period"]
        )

        _specific_contraints = [
            cnstr for cnstr in self.all_contraints if cnstr.name.startswith(_specific_contraint_name)
        ]

        counter = 0

        for cnstr in _specific_contraints:
            cnstr_expr = cnstr.expr
            # Extract coefficients from the constraint expression
            for term in EXPR.decompose_term(cnstr_expr.args[1])[1]:
                _temp_dict[counter] = {"label": EXPR.expression_to_string(term[1]), "value": term[0]}
                counter += 1

        # Create a dictionary to aggregate values for the same keys
        _aggregated_dict = {}

        # Aggregate values for the same keys
        for key, inside_info in _temp_dict.items():
            _aggregated_dict[inside_info["label"]] = (
                _aggregated_dict.get(inside_info["label"], 0) + inside_info["value"]
            )

        # Specify the pattern to filter keys
        _pattern_branch_new_cables = (
            "v_branch_new_cables"  # we care only for the cost coefficients of variables: v_branch_new_cables
        )
        _pattern_branch_new_capacity = (
            "v_branch_new_capacity"  # we care only for the cost coefficients of variables: v_branch_new_capacity
        )

        # Filter entries where the key starts with the specified pattern
        _filtered_dict_brach_new_cables = {
            key: value for key, value in _aggregated_dict.items() if key.startswith(_pattern_branch_new_cables)
        }
        _filtered_dict_brach_new_capacity = {
            key: value for key, value in _aggregated_dict.items() if key.startswith(_pattern_branch_new_capacity)
        }

        # Create the list with the order i want, INDEX ORDER MATTERS!!! - bug when automatically converting the dict values to a list, because the order is not preserved
        _my_filtered_list_new_cables = []
        for br in self.my_sets["s_branch"]:
            for pr in self.my_sets["s_period"]:
                _var_to_check = _pattern_branch_new_cables + f"[{br},{pr}]"
                if _var_to_check in _filtered_dict_brach_new_cables:
                    _my_filtered_list_new_cables.append(
                        _filtered_dict_brach_new_cables[_pattern_branch_new_cables + f"[{br},{pr}]"]
                    )
                else:
                    _my_filtered_list_new_cables.append(0)

        _my_filtered_list_new_capacity = []
        for br in self.my_sets["s_branch"]:
            for pr in self.my_sets["s_period"]:
                _var_to_check = _pattern_branch_new_capacity + f"[{br},{pr}]"
                if _var_to_check in _filtered_dict_brach_new_capacity:
                    _my_filtered_list_new_capacity.append(
                        _filtered_dict_brach_new_capacity[_pattern_branch_new_capacity + f"[{br},{pr}]"]
                    )
                else:
                    _my_filtered_list_new_capacity.append(0)

        _df_new_cables = pd.DataFrame(_my_filtered_list_new_cables, columns=["parameter_value"], index=_multi_index)
        _df_new_capacity = pd.DataFrame(_my_filtered_list_new_capacity, columns=["parameter_value"], index=_multi_index)

        return _df_new_cables, _df_new_capacity

    def get_operation_cost_coefs(self):

        _temp_dict = {}
        _specific_contraint_name = "c_operating_costs"

        # List of indices to select
        costly_gens_condition = self.model.grid_data.generator["type"].isin(
            ["gentype1", "gas", "hydro", "oil", "nonres_other", "res_other", "coal", "nuclear"]
        )

        _indices_to_select = self.model.grid_data.generator.index[costly_gens_condition].to_list()

        # Creating an itemgetter object with the indices
        get_elements = itemgetter(*_indices_to_select)

        # Selecting elements based on indices using itemgetter
        _set_nodes_with_costly_generators = get_elements(self.my_sets["s_gen"])

        if not isinstance(_set_nodes_with_costly_generators, list):
            _set_nodes_with_costly_generators = [_set_nodes_with_costly_generators]

        if isinstance(_set_nodes_with_costly_generators[0], int):

            _multi_index_gen = pd.MultiIndex.from_tuples(
                product(_set_nodes_with_costly_generators, self.my_sets["s_period"], self.my_sets["s_time"]),
                names=["gen", "period", "time"],
            )
        else:
            _multi_index_gen = pd.MultiIndex.from_tuples(
                product(
                    [list(ele) for ele in _set_nodes_with_costly_generators][0],
                    self.my_sets["s_period"],
                    self.my_sets["s_time"],
                ),
                names=["gen", "period", "time"],
            )

        _multi_index_load_shed = pd.MultiIndex.from_tuples(
            product(self.my_sets["s_load"], self.my_sets["s_period"], self.my_sets["s_time"]),
            names=["load", "period", "time"],
        )

        _specific_contraints = [
            cnstr for cnstr in self.all_contraints if cnstr.name.startswith(_specific_contraint_name)
        ]
        counter = 0

        for cnstr in _specific_contraints:
            cnstr_expr = cnstr.expr
            # Extract coefficients from the constraint expression
            for term in EXPR.decompose_term(cnstr_expr.args[1])[1]:
                _temp_dict[counter] = {"label": EXPR.expression_to_string(term[1]), "value": term[0]}
                counter += 1

        # Create a dictionary to aggregate values for the same keys
        _aggregated_dict = {}

        # Aggregate values for the same keys
        for key, inside_info in _temp_dict.items():
            _aggregated_dict[inside_info["label"]] = (
                _aggregated_dict.get(inside_info["label"], 0) + inside_info["value"]
            )

        # Specify the pattern to filter keys
        _pattern_gen = "v_generation"  # we care only for the cost coeeficients of variables: v_generation, v_load_shed
        _pattern_load_shed = (
            "v_load_shed"  # we care only for the cost coeeficients of variables: v_generation, v_load_shed
        )

        # Filter entries where the key starts with the specified pattern
        _filtered_dict = {key: value for key, value in _aggregated_dict.items() if key.startswith(_pattern_gen)}

        _my_filtered_list = []
        if isinstance(_set_nodes_with_costly_generators[0], int):
            gn_set = _set_nodes_with_costly_generators
        else:
            gn_set = [list(ele) for ele in _set_nodes_with_costly_generators][0]
        for gn in gn_set:
            for pr in self.my_sets["s_period"]:
                for tm in self.my_sets["s_time"]:
                    _var_to_check = _pattern_gen + f"[{gn},{pr},{tm}]"
                    if _var_to_check in _filtered_dict:
                        _my_filtered_list.append(_filtered_dict[_pattern_gen + f"[{gn},{pr},{tm}]"])
                    else:
                        _my_filtered_list.append(0)

        _df_gen = pd.DataFrame(_my_filtered_list, columns=["parameter_value"], index=_multi_index_gen)

        _filtered_dict = {key: value for key, value in _aggregated_dict.items() if key.startswith(_pattern_load_shed)}

        _my_filtered_list = []
        for ld in self.my_sets["s_load"]:
            for pr in self.my_sets["s_period"]:
                for tm in self.my_sets["s_time"]:
                    _var_to_check = _pattern_load_shed + f"[{ld},{pr},{tm}]"
                    if _var_to_check in _filtered_dict:
                        _my_filtered_list.append(_filtered_dict[_pattern_load_shed + f"[{ld},{pr},{tm}]"])
                    else:
                        _my_filtered_list.append(0)

        _df_load_shed = pd.DataFrame(_my_filtered_list, columns=["parameter_value"], index=_multi_index_load_shed)

        return _df_gen, _df_load_shed

    def get_branches_losses(self):

        _branch_losses = []
        _my_index = pd.Index(self.my_sets["s_branch"], name="branch")

        for branch in self.my_sets["s_branch"]:
            branchtype = self.model.grid_data.branch.at[branch, "type"]
            dist = self.model.grid_data.branch.at[branch, "distance"]
            loss_fix = self.model.branchtypes[branchtype]["loss_fix"]
            loss_slope = self.model.branchtypes[branchtype]["loss_slope"]
            _branch_losses.append(loss_fix + loss_slope * dist)

        _df = pd.DataFrame(_branch_losses, columns=["parameter_value"], index=_my_index)

        return _df


# -------------------------------PGIM CASE VISUALIZATION CLASS----------------------------------------
class CaseMapVisualizer:
    """
    Create interactive html maps for the "baseline" case input data, along with case results (optionally).
    """

    def __init__(
        self, pgim_ref, rslt=None, data_locations_windoff=None, outFileName=None, outPath=None, visualize_rslt=False
    ):
        self.all_periods = [2030] + pgim_ref.investment_periods
        self.grid_data = pgim_ref.ref_grid_data
        self.viz_rslt = visualize_rslt
        self.rslt = rslt
        if data_locations_windoff is None:
            self.data_locations_windoff = {
                "NO_SoennavindA": {"lat": 57.51624269, "lon": 7.391032075},
                "NO_SoervestF": {"lat": 56.81703138, "lon": 4.872008852},
                "NO_SoervestB": {"lat": 57.3480636, "lon": 3.360271522},
                "NO_SoervestC": {"lat": 57.03010167, "lon": 3.885691333},
                "NO_VestavindF": {"lat": 59.275943, "lon": 4.540937},
                "NO_VestavindE": {"lat": 59.11131655, "lon": 3.852830719},
                "NO_VestavindB": {"lat": 61.05563737, "lon": 3.617045878},
                "NO_VestavindA": {"lat": 61.98188847, "lon": 3.705499849},
                "NO_NordvestC": {"lat": 63.77727349, "lon": 6.647918674},
                "NO_NordvestA": {"lat": 66.23346252, "lon": 9.580682563},
                "SE_3": {"lat": 56.8550, "lon": 12.1509},
                "DK_1": {"lat": 56.3531, "lon": 6.4490},
                "DK_2": {"lat": 56.2617, "lon": 11.4587},
                "DE_LU": {"lat": 54.7817, "lon": 5.7129},
                "NL_1": {"lat": 53.0214, "lon": 3.9441},
                "NL_2": {"lat": 54.0916, "lon": 4.8340},
                "BE": {"lat": 51.6044, "lon": 2.7136},
                "GB_mid": {"lat": 55.0909, "lon": 1.8677},
                "GB_sor": {"lat": 53.0478, "lon": 2.5049},
                "GB_scot": {"lat": 57.1720, "lon": -0.9338},
            }
        else:
            self.data_locations_windoff = data_locations_windoff
        if (outFileName is None) or (outPath is None):
            raise Exception(f"An output path needs to be specified. current path {outPath}/{outFileName}")
        else:
            self.outFileName = outFileName
            self.outPath = outPath
        self.arbitarily_large_branch_capacity = 100000
        self.cable_capacity_nom = 1500  # Nominal branch capacity [MW]
        self.radius_farm_symbol_nominal = 10

    def create_html_map(self):

        myMap = folium.Map(location=[55.88, 2.19], tiles="cartodbpositron", zoom_start=7, control_scale=True)

        # Get point coordinates on the map
        folium.LatLngPopup().add_to(myMap)

        # Marine boundaries
        boundaries = folium.WmsTileLayer(
            url="http://geo.vliz.be/geoserver/MarineRegions/wms?",
            layers="MarineRegions:eez_boundaries",
            transparent=True,
            fmt="image/png",
            name="Marine boundaries (VLIZ)",
        )
        boundaries.add_to(myMap)

        pgim_periods = [str(iperiod) for iperiod in self.all_periods]

        capacity_column_names = ["capacity_" + i for i in pgim_periods]
        capacity_column_names[0] = "existing_" + pgim_periods[0]
        expand_column_names = ["expand_" + i for i in pgim_periods[1:]]

        cumulative_generation_capacity = self.grid_data.generator.loc[(self.grid_data.generator["type"] == "windoff")][
            capacity_column_names[1:]
        ].cumsum(axis=1)

        min_capacity = self.grid_data.generator[capacity_column_names[0]].min()
        max_capacity = cumulative_generation_capacity[capacity_column_names[-1]].max()

        radius_farm_symbol = np.linspace(5, 40, 300)
        capacity_levels = np.linspace(min_capacity, max_capacity, 300)

        radius_weight_farm_symbol = {pgim_periods[0]: 1, pgim_periods[1]: 1, pgim_periods[2]: 1, pgim_periods[3]: 1}
        # Define feature_groups.
        feature_group_vars_period_1 = folium.FeatureGroup("Branches (candidates-2035)")
        feature_group_vars_period_2 = folium.FeatureGroup("Branches (candidates-2040)")
        feature_group_vars_period_3 = folium.FeatureGroup("Branches (candidates-2050)")

        feature_group_ons = folium.FeatureGroup("Branches (onshore grid)")
        feature_group_internat = folium.FeatureGroup("Branches (existing international connections)")

        if self.viz_rslt is True:
            _strategic_nodes = list(self.rslt["z_new_lines"].index.get_level_values(1).unique())

        feature_group_internat_period_0 = folium.FeatureGroup("Branches (constraints-2030)")
        feature_group_internat_period_1 = folium.FeatureGroup("Branches (constraints-2035)")
        feature_group_internat_period_2 = folium.FeatureGroup("Branches (constraints-2040)")
        feature_group_internat_period_3 = folium.FeatureGroup("Branches (constraints-2050)")

        feature_group_nonvars = folium.FeatureGroup("Branches (non-variables but constraints)")
        feature_group_exist = folium.FeatureGroup(
            f"Radial WF connections existing at {pgim_periods[0]} (no further developements)"
        )
        feature_group_gens_period_0 = folium.FeatureGroup(f"Offshore wind clusters existing at {pgim_periods[0]}")
        feature_group_gens_period_1 = folium.FeatureGroup(f"Additional offshore wind capacity at {pgim_periods[1]}")
        feature_group_gens_period_2 = folium.FeatureGroup(f"Additional offshore wind capacity at {pgim_periods[2]}")
        feature_group_gens_period_3 = folium.FeatureGroup(f"Additional offshore wind capacity at {pgim_periods[3]}")
        feature_group_nodes = folium.FeatureGroup("Nodes")
        feature_group_generators = folium.FeatureGroup("Generators (mainland)")
        feature_group_windoff = folium.FeatureGroup("Offshore wind data locations")
        feature_group_rslt_period_1 = folium.FeatureGroup("Investments 2035 (1st-stage)")

        if self.viz_rslt:

            if len(_strategic_nodes) <= 3:
                feature_group_rslt_period_2 = folium.FeatureGroup("Investments 2040 (2nd-stage)")
                feature_group_rslt_period_3 = folium.FeatureGroup("Investments 2050 (2nd-stage)")
            else:
                feature_group_rslt_period_2_scen_0 = folium.FeatureGroup("Investments 2040 - scenario 0 (2nd-stage)")
                feature_group_rslt_period_2_scen_1 = folium.FeatureGroup("Investments 2040 - scenario 1 (2nd-stage)")
                feature_group_rslt_period_2_scen_2 = folium.FeatureGroup("Investments 2040 - scenario 2 (2nd-stage)")

                feature_group_rslt_period_3_scen_0 = folium.FeatureGroup("Investments 2050 - scenario 0 (2nd-stage)")
                feature_group_rslt_period_3_scen_1 = folium.FeatureGroup("Investments 2050 - scenario 1 (2nd-stage)")
                feature_group_rslt_period_3_scen_2 = folium.FeatureGroup("Investments 2050 - scenario 2 (2nd-stage)")

        for ibranch, branch_row_data in self.grid_data.branch.iterrows():

            branch_locations = [
                (
                    self.grid_data.node.at[branch_row_data["node_from"], "lat"],
                    self.grid_data.node.at[branch_row_data["node_from"], "lon"],
                ),
                (
                    self.grid_data.node.at[branch_row_data["node_to"], "lat"],
                    self.grid_data.node.at[branch_row_data["node_to"], "lon"],
                ),
            ]

            # Identify branches that are variables.
            if branch_row_data[expand_column_names].sum() > 0:

                for _, iperiod in enumerate(pgim_periods):
                    if branch_row_data[expand_column_names[0]] > 0 and iperiod == pgim_periods[1]:
                        folium.PolyLine(
                            branch_locations,
                            color="yellow",
                            popup=f"variable_{branch_row_data['node_from']}-{branch_row_data['node_to']}_branch_index_{ibranch}",
                            weight=1.5,
                        ).add_to(feature_group_vars_period_1)
                        folium.PolyLine(branch_locations, color="black", dash_array="10", weight=1).add_to(
                            feature_group_vars_period_1
                        )
                    elif branch_row_data[expand_column_names[1]] > 0 and iperiod == pgim_periods[2]:
                        folium.PolyLine(
                            branch_locations,
                            color="yellow",
                            popup=f"variable_{branch_row_data['node_from']}-{branch_row_data['node_to']}_branch_index_{ibranch}",
                            weight=1.5,
                        ).add_to(feature_group_vars_period_2)
                        folium.PolyLine(branch_locations, color="black", dash_array="10", weight=1).add_to(
                            feature_group_vars_period_2
                        )
                    elif branch_row_data[expand_column_names[2]] > 0 and iperiod == pgim_periods[3]:
                        folium.PolyLine(
                            branch_locations,
                            color="yellow",
                            popup=f"variable_{branch_row_data['node_from']}-{branch_row_data['node_to']}_branch_index_{ibranch}",
                            weight=1.5,
                        ).add_to(feature_group_vars_period_3)
                        folium.PolyLine(branch_locations, color="black", dash_array="10", weight=1).add_to(
                            feature_group_vars_period_3
                        )

            # ----------------NON-VARIABLE BRANCHES ---> CONSTRAINTS FOR THE OPTIMIZATION PROBLEM-------------------

            # Identify branches that are on land and connect landing points with main nodes (but not internationally).
            elif (branch_row_data[capacity_column_names[0]] >= self.arbitarily_large_branch_capacity) and (
                branch_row_data["node_from"] != "UK_scotland"
            ):

                folium.PolyLine(
                    branch_locations,
                    color="pink",
                    popup=f"onshore_grid_capacity_{branch_row_data[capacity_column_names[0]]}_MW",
                    weight=2,
                ).add_to(feature_group_ons)

            # Identify specific "well-known" offshore interconnections.
            elif branch_row_data["node_from"] == "NO_fagrafjell" and branch_row_data["node_to"] == "UK_L6":
                folium.PolyLine(
                    branch_locations,
                    color="green",
                    popup=f"NorthSea Link_branch_index_{ibranch}_{branch_row_data[capacity_column_names[0]]}_MW",
                    weight=2,
                ).add_to(feature_group_internat)
            elif branch_row_data["node_from"] == "NO_kvinesdal" and branch_row_data["node_to"] == "DE_L2":
                folium.PolyLine(
                    branch_locations,
                    color="green",
                    popup=f"NordLink__{ibranch}_{branch_row_data[capacity_column_names[0]]}_MW",
                    weight=2,
                ).add_to(feature_group_internat)
            elif branch_row_data["node_from"] == "NO_kvinesdal" and branch_row_data["node_to"] == "NL_L4":
                folium.PolyLine(
                    branch_locations,
                    color="green",
                    popup=f"NorNed__{ibranch}_{branch_row_data[capacity_column_names[0]]}_MW",
                    weight=2,
                ).add_to(feature_group_internat)
            else:
                if branch_row_data[capacity_column_names].sum() > 0:
                    folium.PolyLine(branch_locations, color="red", weight=1).add_to(feature_group_nonvars)
                    for _, iperiod in enumerate(pgim_periods):
                        if branch_row_data[capacity_column_names[0]] > 0 and iperiod == pgim_periods[0]:
                            folium.PolyLine(
                                branch_locations,
                                color="black",
                                popup=f"branch_{branch_row_data['node_from'][:2]}-{branch_row_data['node_to'][:2]}_branch_index_{ibranch}_total_capacity_{round(branch_row_data[capacity_column_names[0]])}_MW",
                                weight=2,
                                dash_array="10",
                            ).add_to(feature_group_internat_period_0)
                        elif branch_row_data[capacity_column_names[1]] > 0 and iperiod == pgim_periods[1]:
                            folium.PolyLine(
                                branch_locations,
                                color="black",
                                popup=f"branch_{branch_row_data['node_from'][:2]}-{branch_row_data['node_to'][:2]}_branch_index_{ibranch}_total_capacity_{round(branch_row_data[capacity_column_names[1]])}_MW",
                                weight=2,
                                dash_array="10",
                            ).add_to(feature_group_internat_period_1)
                        elif branch_row_data[capacity_column_names[2]] > 0 and iperiod == pgim_periods[2]:
                            folium.PolyLine(
                                branch_locations,
                                color="black",
                                popup=f"branch_{branch_row_data['node_from'][:2]}-{branch_row_data['node_to'][:2]}_branch_index_{ibranch}_total_capacity_{round(branch_row_data[capacity_column_names[1]]+branch_row_data[capacity_column_names[2]])}_MW",
                                weight=2,
                                dash_array="10",
                            ).add_to(feature_group_internat_period_2)
                        elif branch_row_data[capacity_column_names[3]] > 0 and iperiod == pgim_periods[3]:
                            folium.PolyLine(
                                branch_locations,
                                color="black",
                                popup=f"branch_{branch_row_data['node_from'][:2]}-{branch_row_data['node_to'][:2]}_branch_index_{ibranch}_total_capacity_{round(branch_row_data[capacity_column_names[1]]+branch_row_data[capacity_column_names[2]]+branch_row_data[capacity_column_names[3]])}_MW",
                                weight=2,
                                dash_array="10",
                            ).add_to(feature_group_internat_period_3)

            # Identify investments from results
            # NOTE: This is NOT to be used with "binary expansion".
            if self.viz_rslt is True:
                for n_strgc in _strategic_nodes:
                    x_cbl = (
                        self.rslt["z_new_lines"]
                        .xs(n_strgc, level=1)
                        .loc[self.rslt["z_new_lines"].xs(n_strgc, level=1).values > 0.01]
                    )
                    x_cpt = (
                        self.rslt["z_new_capacity"]
                        .xs(n_strgc, level=1)
                        .loc[self.rslt["z_new_capacity"].xs(n_strgc, level=1).values > 0.01]
                    )

                    if len(_strategic_nodes) <= 3:

                        if (ibranch in x_cbl.index.get_level_values("branch")) and (
                            ibranch in x_cpt.index.get_level_values("branch")
                        ):
                            cables = round(x_cbl.loc[ibranch])
                            capacity = round(x_cpt.loc[ibranch])
                            if n_strgc == 0:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_1)
                            elif n_strgc == 1:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_2)
                            else:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_3)
                    else:
                        if (ibranch in x_cbl.index.get_level_values("branch")) and (
                            ibranch in x_cpt.index.get_level_values("branch")
                        ):
                            cables = round(x_cbl.loc[ibranch])
                            capacity = round(x_cpt.loc[ibranch])
                            if n_strgc == 0:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_1)
                            elif n_strgc == 1:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_2_scen_1)
                            elif n_strgc == 4:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_3_scen_1)
                            elif n_strgc == 2:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_2_scen_0)
                            elif n_strgc == 5:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_3_scen_0)
                            elif n_strgc == 3:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_2_scen_2)
                            elif n_strgc == 6:
                                folium.PolyLine(
                                    branch_locations,
                                    color="blue",
                                    popup=f"investment_{branch_row_data['node_from']}-{branch_row_data['node_to']}_new_lines_{cables}_total_capacity_{capacity}",
                                    weight=3,
                                ).add_to(feature_group_rslt_period_3_scen_2)

        test_keys = ["UK", "DK", "DE", "NL", "NO", "SE", "BE"]
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cadetblue",
            "darkred",
            "pink",
            "darkgreen",
            "lightgreen",
            "darkblue",
            "beige",
            "lightred",
            "lightblue",
            "darkpurple",
            "gray",
            "lightgray",
            "black",
        ]

        test_values = colors[0 : len(test_keys)]

        # using dictionary comprehension to convert the list to dictionary.
        colorMap = {test_keys[i]: test_values[i] for i in range(len(test_keys))}

        df_generators = self.grid_data.generator

        for igen, gen_row_data in df_generators.iterrows():
            # First check if (lat, lon) are "empty" (both equal to -1), and if yes, get this info from nodes.csv.
            if (gen_row_data["lat"] == -1) and (gen_row_data["lon"] == -1):
                try:
                    df_generators.at[igen, "lat"] = self.grid_data.node.at[gen_row_data["node"], "lat"]
                    df_generators.at[igen, "lon"] = self.grid_data.node.at[gen_row_data["node"], "lon"]
                except Exception:
                    print(f"Node {gen_row_data['node']} exists in generators.csv but nodes.csv")

        # Scan through the generators.csv.
        for igen, gen_row_data in df_generators.iterrows():

            # Identify the offshore wind generators.
            if gen_row_data["comment"] == "offshorewind":

                # Identify farms that will be developed in the investment pgim_periods (non-zero capacity in any of [2035,2040,2050]).
                if (gen_row_data[capacity_column_names[1:]].sum() - gen_row_data[capacity_column_names[0]]) > 0:

                    for idx, iperiod in enumerate(pgim_periods):

                        df_cum_sum = (
                            gen_row_data[capacity_column_names].cumsum() - gen_row_data[capacity_column_names[0]]
                        )

                        radius_farm_symbol_interp = np.interp(
                            df_cum_sum[capacity_column_names[idx]], capacity_levels, radius_farm_symbol
                        )

                        # Identify farms that already exist in the initial period [2030].
                        if gen_row_data[capacity_column_names[0]] > 0 and iperiod == pgim_periods[0]:
                            folium.CircleMarker(
                                [gen_row_data["lat"], gen_row_data["lon"]],
                                radius=radius_farm_symbol_interp,
                                fill=True,
                                fill_opacity=0.4,
                                weight=radius_weight_farm_symbol[iperiod],
                                popup=f"WF_{gen_row_data['node']}_({round(gen_row_data[capacity_column_names[1:]].cumsum()[capacity_column_names[-1]])}_MW)",
                                color=colorMap[self.grid_data.node.at[gen_row_data["node"], "area"]],
                            ).add_to(feature_group_gens_period_0)

                        # Identify farms that do not exist in the initial period [2030], but are going to be built in the investment pgim_periods [2035,2040,2050].
                        # Identify farm capacity developements in the first investment period [2035].
                        elif (
                            gen_row_data[capacity_column_names[1]] - gen_row_data[capacity_column_names[0]]
                        ) > 0 and iperiod == pgim_periods[1]:

                            folium.CircleMarker(
                                [gen_row_data["lat"], gen_row_data["lon"]],
                                radius=radius_farm_symbol_interp,
                                fill=False,
                                weight=radius_weight_farm_symbol[iperiod],
                                popup=f"WF_{gen_row_data['node']}_({round(gen_row_data[capacity_column_names[1:]].cumsum()[capacity_column_names[-1]])}_MW)",
                                color=colorMap[self.grid_data.node.at[gen_row_data["node"], "area"]],
                            ).add_to(feature_group_gens_period_1)

                        # Identify farm capacity developements in the second investment period [2040].
                        elif gen_row_data[capacity_column_names[2]] > 0 and iperiod == pgim_periods[2]:
                            folium.CircleMarker(
                                [gen_row_data["lat"], gen_row_data["lon"]],
                                radius=radius_farm_symbol_interp,
                                fill=False,
                                weight=radius_weight_farm_symbol[iperiod],
                                popup=f"WF_{gen_row_data['node']}_({round(gen_row_data[capacity_column_names[1:]].cumsum()[capacity_column_names[-1]])}_MW)",
                                color=colorMap[self.grid_data.node.at[gen_row_data["node"], "area"]],
                            ).add_to(feature_group_gens_period_2)

                        # Identify farm capacity developements in the thrid investment period [2050].
                        elif gen_row_data[capacity_column_names[3]] > 0 and iperiod == pgim_periods[3]:
                            folium.CircleMarker(
                                [gen_row_data["lat"], gen_row_data["lon"]],
                                radius=radius_farm_symbol_interp,
                                fill=False,
                                weight=radius_weight_farm_symbol[iperiod],
                                popup=f"WF_{gen_row_data['node']}_({round(gen_row_data[capacity_column_names[1:]].cumsum()[capacity_column_names[-1]])}_MW)",
                                color=colorMap[self.grid_data.node.at[gen_row_data["node"], "area"]],
                            ).add_to(feature_group_gens_period_3)

                else:

                    folium.Marker(
                        [gen_row_data["lat"], gen_row_data["lon"]],
                        popup=f"WF_{gen_row_data['node']}_({round(gen_row_data[capacity_column_names[0]])}_MW)",
                        icon=folium.DivIcon(
                            html=f"""
                            <div><svg height ="10" width="10">
                                <rect width="10" height="10", fill="none" stroke-width="3"
                                stroke={colorMap[self.grid_data.node.at[gen_row_data['node'], 'area']]}
                                </svg></div>"""
                        ),
                    ).add_to(feature_group_exist)

                    if (self.grid_data.node.index == gen_row_data["node"]).any():

                        branch_locations = [
                            (gen_row_data["lat"], gen_row_data["lon"]),
                            (
                                self.grid_data.node.loc[gen_row_data["node"], "lat"],
                                self.grid_data.node.loc[gen_row_data["node"], "lon"],
                            ),
                        ]
                        folium.PolyLine(branch_locations, color="gray", weight=1).add_to(feature_group_exist)

                # Identify wind farms that are radially connected to a corresponding land node.
                prefixes = [_i + "_L" for _i in test_keys]
                if (
                    (gen_row_data["node"].startswith(tuple(prefixes)))
                    or (gen_row_data["node"].endswith(("east", "west", "main", "scotland")))
                    or (gen_row_data[capacity_column_names[1:]].sum() == gen_row_data[capacity_column_names[0]])
                ):

                    branch_locations = [
                        (gen_row_data["lat"], gen_row_data["lon"]),
                        (
                            self.grid_data.node.loc[gen_row_data["node"], "lat"],
                            self.grid_data.node.loc[gen_row_data["node"], "lon"],
                        ),
                    ]
                    folium.PolyLine(branch_locations, color="gray", weight=1).add_to(feature_group_exist)

                # Identify wind farms that are connected with each other.
                # This is only needed for the plotting. Those generators, share the same node (which means additional capacity for that node).
                for node_id, node_row_data in self.grid_data.node.iterrows():

                    """
                    Conditions to verify:
                        1) Identify the generators that share a common id with a node,
                        2) and are just radial connections (not further developements)
                        3) Exclude the mainland multi-generatos.
                        4) Exclude the mainland multi-generatos and the hub in Denmark (east, west, hub).
                    """

                    if (
                        gen_row_data["node"] == node_id
                        and (gen_row_data[capacity_column_names[1:]].sum() - gen_row_data[capacity_column_names[0]]) > 0
                        and (not gen_row_data["node"].startswith(tuple(prefixes)))
                        and (not gen_row_data["node"].endswith(("east", "west", "hub")))
                    ):

                        branch_locations = [
                            (gen_row_data["lat"], gen_row_data["lon"]),
                            (node_row_data["lat"], node_row_data["lon"]),
                        ]
                        folium.PolyLine(branch_locations, color="gray", weight=1).add_to(myMap)

        # Scan through the nodes.csv and find the consumers (use of "node_id" because the index of self.grid_data.node is the node "id" ).
        for node_id, node_row_data in self.grid_data.node.iterrows():

            # Identify the "consumer" nodes.
            if node_id in self.grid_data.consumer["node"].values.tolist():
                folium.Marker(
                    [node_row_data["lat"], node_row_data["lon"]],
                    radius=1000,
                    popup="load_" + node_id,
                    icon=folium.Icon(icon="glyphicon-home", prefix="glyphicon"),
                ).add_to(myMap)
            else:
                folium.CircleMarker(
                    [node_row_data["lat"], node_row_data["lon"]],
                    radius=self.radius_farm_symbol_nominal / 6,
                    fill=True,
                    fill_opacity=1,
                    fill_color="black",
                    popup=f"node_id_{node_id}",
                    color="black",
                ).add_to(feature_group_nodes)

            # Identify the "main" nodes (with several types of generators).
            if (node_id.endswith("_main")) or (node_id.startswith("NO_") and node_row_data["comment"] == "NO_split"):
                # Create a cluster for each "main" node.
                if node_id.startswith("NO_"):
                    gens_cluster = folium.plugins.MarkerCluster(name=f"generators_cluster_{node_id}").add_to(
                        feature_group_generators
                    )
                else:
                    gens_cluster = folium.plugins.MarkerCluster(
                        name=f"generators_cluster_{node_row_data['area']}"
                    ).add_to(feature_group_generators)

                # Calculate aggragated capacity form generation mix of the cluster.
                if node_id.startswith("NO_") and node_row_data["comment"] == "NO_split":
                    aggregated_node_generation_capacity = (
                        df_generators[capacity_column_names[0]]
                        .loc[(df_generators["comment"] == "mainland")]
                        .loc[df_generators["node"].str.startswith("NO_")]
                        .sum()
                    )
                    radius_scaling_farm_symbol = 10
                else:
                    aggregated_node_generation_capacity = (
                        df_generators[capacity_column_names[0]]
                        .loc[(df_generators["comment"] == "mainland")]
                        .loc[(df_generators["node"] == node_id)]
                        .sum()
                    )
                    radius_scaling_farm_symbol = 4

                # Identify different generator types, specify different colors and radius based on local energy mix.
                # Add components to the specific cluster.
                for igen, gen_row_data in (
                    df_generators.loc[df_generators["comment"] == "mainland"]
                    .loc[df_generators["node"] == node_id]
                    .iterrows()
                ):

                    if gen_row_data["type"] != "windoff":
                        match gen_row_data["type"]:
                            case "gas":
                                genTypeColor = "cadetblue"
                            case "coal":
                                genTypeColor = "gray"
                            case "hydro":
                                genTypeColor = "blue"
                            case "nonres_other":
                                genTypeColor = "purple"
                            case "oil":
                                genTypeColor = "black"
                            case "res_other":
                                genTypeColor = "lightgreen"
                            case "solar":
                                genTypeColor = "orange"
                            case "windon":
                                genTypeColor = "green"
                            case "nuclear":
                                genTypeColor = "red"
                            case "flex":
                                genTypeColor = "pink"
                            case _:
                                print(f"{gen_row_data['type']} generator type is missing")

                    genTypeRadiusprcnt = (
                        gen_row_data[capacity_column_names[0]] / aggregated_node_generation_capacity + 0.01
                    )

                    gen_marker = folium.CircleMarker(
                        [gen_row_data["lat"], gen_row_data["lon"]],
                        radius=self.radius_farm_symbol_nominal * radius_scaling_farm_symbol * genTypeRadiusprcnt,
                        popup=f"generator_{gen_row_data['desc']}_({round(gen_row_data[capacity_column_names[0]])}_MW)",
                        fill=True,
                        fill_opacity=0.8,
                        color=genTypeColor,
                    )

                    gens_cluster.add_child(gen_marker).add_to(feature_group_generators)

        # Identify reference offshore locations used to extract wind data timeseries.
        # Different color for differenct country.
        def condition(ele, code):
            cond_true = ele.startswith(code) or (ele.startswith("GB_") and code == "UK")
            return cond_true

        for izone in list(self.data_locations_windoff.keys()):

            colorIdx = [idx for idx, country_code in enumerate(test_keys) if condition(izone, country_code)]
            # marker has an "envelope" icon inside.
            folium.Marker(
                [self.data_locations_windoff[izone]["lat"], self.data_locations_windoff[izone]["lon"]],
                radius=1000,
                popup="wind_data_" + izone,
                icon=folium.Icon(
                    color=colorMap[test_keys[colorIdx[0]]], icon="glyphicon-folder-open", prefix="glyphicon"
                ),
            ).add_to(feature_group_windoff)

        # Add all the feature_groups to myMap.
        feature_group_nodes.add_to(myMap)
        feature_group_vars_period_1.add_to(myMap)
        feature_group_vars_period_2.add_to(myMap)
        feature_group_vars_period_3.add_to(myMap)
        feature_group_nonvars.add_to(myMap)
        feature_group_exist.add_to(myMap)
        feature_group_gens_period_0.add_to(myMap)
        feature_group_gens_period_1.add_to(myMap)
        feature_group_gens_period_2.add_to(myMap)
        feature_group_gens_period_3.add_to(myMap)
        feature_group_internat.add_to(myMap)
        feature_group_internat_period_0.add_to(myMap)
        feature_group_internat_period_1.add_to(myMap)
        feature_group_internat_period_2.add_to(myMap)
        feature_group_internat_period_3.add_to(myMap)
        feature_group_ons.add_to(myMap)
        feature_group_windoff.add_to(myMap)
        feature_group_generators.add_to(myMap)

        if self.viz_rslt:
            feature_group_rslt_period_1.add_to(myMap)

            if len(_strategic_nodes) <= 3:
                feature_group_rslt_period_2.add_to(myMap)
                feature_group_rslt_period_3.add_to(myMap)
            else:
                feature_group_rslt_period_2_scen_0.add_to(myMap)
                feature_group_rslt_period_3_scen_0.add_to(myMap)

                feature_group_rslt_period_2_scen_1.add_to(myMap)
                feature_group_rslt_period_3_scen_1.add_to(myMap)

                feature_group_rslt_period_2_scen_2.add_to(myMap)
                feature_group_rslt_period_3_scen_2.add_to(myMap)

        folium.LayerControl().add_to(myMap)

        draw = folium.plugins.Draw(export=True)
        draw.add_to(myMap)

        if self.outFileName is not None:
            myMap.save(self.outPath / self.outFileName)

            # Open html files in chrome using
            filename = "file:///" + str(self.outPath / self.outFileName)
            webbrowser.open_new_tab(filename)
