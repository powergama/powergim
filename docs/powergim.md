# Power Grid Investment Module - user guide ![logo](files/logo_powergim.png)

Contents:

* [Modelling documentation](#modelling-documentation)
* [Examples](#examples)
* [What it does](#what-it-does)
* [Running an optimisation](#mrunning-an-optimisation)
* [Input data](#input-data)
    * [Grid data](#grid-data)
    * [Time sample](#time-sample)
    * [Parameters](#parameters)
* [Analysis of results](#analysis-of-results)
* [More about optimisation model](#more-about-the-powergim-optimisation-model)


## Modelling documentation
A separate (draft) document provides more details about the modelling framework and
the theoretical context of the PowerGIM model.
This is available [here](files/PowerGIM.pdf).

## Examples
For a quick demonstration of how PowerGIM works, have a look at these
Jupyter notebooks:

1. Example of [deterministic](https://nbviewer.jupyter.org/urls/bitbucket.org/harald_g_svendsen/powergama/raw/master/powergama/examples/powergim_doggerbank/powergim_doggerbank.ipynb) optimisation
2. Example with added [stochastic](https://nbviewer.jupyter.org/urls/bitbucket.org/harald_g_svendsen/powergama/raw/master/powergama/examples/powergim_doggerbank/powergim_doggerbank_stochastic.ipynb) parameters

## What it does
PowerGIM is a module for grid investment analyses that is included with
the PowerGAMA python package. It is a power system *expansion planning* model
that can consider both transmission and generator investments.

PowerGIM works by finding the optimal investments amongst a set of specified
candidates such that the net present value of total system costs (capex and
opex) are minimised.

It is built on top of the [Pyomo](http://www.pyomo.org/) package.

### Two-stage optimisation
PowerGIM is formulated as a two-stage optimisation problem, and there may be
a time delay between the two investment stages. First-stage variables represent
the *here-and-now* investments that are of primary interest. Second-stage
variables include operational decisions and future investments.


### Uncertainty - stochastic programming
For many investment decisions it is important to consider uncertainties and
identify solutions that are good for all likely values of the uncertain
parameters. In the context of grid planning, relevant uncertainties are for
example future generator capacity and energy demand at different grid
locations, power prices. These things may have a huge impact on the benefit
of different investments alternatives.

Rather than just finding the optimal solution for a specific set of assumptions,
it is more relevant to find the solution that is best for the whole range of
potential realisations of the uncertain parameters.
This is *stochastic programming*.

![two stage stochastic optimisation](files/twostage_stochastic.png)

PowerGIM includes this capability via the two-stage formulation. Uncertain
parametes and their potential values are specified via a *scenario tree*.

First stage variables (here-and-now decisions) are made without knowing
which scenario is realised, while second stage variables are different for the
different scenarios.


A simple example of a two-stage stochastic optimisation problem with three
scenarios (representing uncertainty about future wind farm capacities) and
their probabilities (P) is shown below:

![scenario tree](files/twostage_scenarios.png)

## Running an optimisation

See the [examples](#examples).

The general steps when using PowerGIM in a Python script/notebook
to specify and solve optimisation problems are:

Preparations:

1. Create input data sets (csv and yaml)
    * these specify existing infrastructure and candidate investments

Script:

2. Read input data and create optimisation model
3. Create model instance
    * for stochastic problem: specify scenario tree and model instance creation
  call-back function
4. Solve model
5. Save/inspect/analyse results


## Input data

### Grid data
Grid data is imported from CSV files. There are separate files for *nodes*, *branches*, *generators* and
*consumers*. Below, \<year\> refers to an investment year, and must match the years specified in the parameter value investment_year

#### Nodes
Node data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
id   | Unique string identifier | string
area | Area/country code | string
capacity_\<year\>  | (Additional) capacity installed before investment year <year>| float   | MW
expand_\<year\>    | Consider expansion in investment year <year>   | boolean | 0,1
lat  | Latitude   | float |degrees
lon  | Longitude  | float |degrees
offshore | Whether node is offshore | boolean | 0,1
cost_scaling | Cost scaling factor |float
type | Node (cost) type |string

#### Branches
Branch data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
node_from | Node identifier | string
node to   | Node identifier | string
capacity_\<year\>  | (Additional) capacity installed before investment year <year>| float   | MW
expand_\<year\>    | Consider expansion in investment year <year>   | boolean | 0,1
distance  | Branch length (OPT) | float | km
max_newCap    | Max new capacity (OPT) | float | km
cost_scaling  | Cost scaling factor | float
type      | Branch (cost) type | string

Branches have from and to references that must match a node identifier
in the list of nodes.
* distance may be left blank. Then distance is computed as the shortest
  distance between the associated nodes (based on lat/lon coordinates)
* expand_\<year\> is 0 if no expansion should be considered (not part of
  optimisaion)
* capacity_\<year\> is already decided (present or additional future) branch 
  capacity, i.e. it does not depend on the optimisation output



#### Generators
Generator data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
node  | Node identifier |string
desc  | Description or name (OPT) |string
type  | Generator type |string
capacity_\<year\>  | (Additional) capacity installed before investment year <year>  |float |MW
pmin  | Minimum production |float |MW
expand_\<year\> | Consider expansion in investment year <year>   | boolean | 0,1
allow_curtailment | Whether generator can be curtailed | boolean | 0,1
fuelcost  | Cost of generation |float |€/MWh
fuelcost_ref  | Cost profile |string
inflow_fac  | Inflow factor |float
inflow_ref  | Inflow profile reference |string
pavg  | Average power output (OPT) |float |MW
p_maxNew  | Maximum new capacity (OPT) |float |MW
cost_scaling  | Cost scaling factor (OPT) |float

* The average power constraint (pavg) is used to represent generators
  with large storage. pavg=0 means no constraint on average output is used
  (no storage constraint).
* capacity_\<year\> is already decided (present or additional future) generator 
  capacity, i.e. it does not depend on the optimisation output


#### Consumers
Consumer (power demand) data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
node    | Node identifier  | string
demand_avg  | Average demand |float |MW
demand_ref  | Profile reference |string
emission_cap| Maximum CO2 emission allowed (OPT) |float |kg

* There may be any number of consumers per node, although zero or one is
  typical.
* demand_avg gives the average demand, which is easily computed from the
  annual demand if necessary.
* demand_ref gives the name of the demand profile (time sample) which gives
  the variation over time. Demand profiles should be normalised and have an annual
  average of 1.


### Time sample
A set of time series or samples are used to represent the variability in
renewable energy availability, power demand and generator fuel costs
(power prices).

These data are provided as a CSV file with one profile/sample per column, with
the column header being the profile string identifier, and one row per
timestamp.

the time samples are used together with base values to get demand, available
power and fuel costs at a given time as follows:

*demand(t) = demand_avg ×  demand_ref(t) fuelcost(t) = fuelcost ×  fuelcost_ref(t) pmax(t) =  (pmax+pmax2) × inflow_fac ×  inflow_ref(t)*


### Parameters
Investment costs and other parameters are provided in an YAML file with the
following structure:
```YAML
nodetype:
    ac: { L: 1, S: 50e6}
    hvdc: { L: 1, S: 1 }
branchtype:
    <branchtype1>:
        B: 5000e3
        Bdp: 1.15e3
        Bd: 656e3
        CL: 1562e3
        CLp: 0
        CS: 4813e3
        CSp: 0
        max_cap: 400
        loss_fix: 0
        loss_slope: 5e-5
    <branchtype2>:
        B: 5000e3
        Bdp: 0.47e3
        Bd: 680e3
        CL: 0
        CLp: 0
        CS: 0
        CSp: 0
        max_cap: 2000
        loss_fix: 0
        loss_slope: 3e-5
gentype:
    <gentype1>:
       CX: 10
       CO2: 0
    <gentype2>:
       CX: 0
       CO2: 0
parameters:
    investment_years: [2025, 2028]
    finance_interest_rate: 0.05
    finance_years: 40
    operation_maintenance_rate: 0.05
    CO2_price: 0
    load_shed_penalty: 10000 # very high value of lost load (loadshedding penalty)
    profiles_period_suffix: False
```

Most of the parametes in the  ```nodetype```, ```branchtype``` and ```gentype```
blocks are [cost parameters](#cost-model)
```branchtype``` has the following additional parameters related to
[power losses](#power-losses), and the maximum allowable
power rating per cable system (maxCap)

Parameters specified in the ```parameters``` block are:

* investment_years = list of years to consider for investments. The first value is the here-and-now (stage 1) investments. Use absolute values (e.g. [2030, 2040]) or relative values (e.g. [0, 5, 10]). The years specified here must match \<year\> in column names in the grud uboyt data,
* finance_interest_rate = discount rate used in net present value calculation of
  generation costs and operation and maintenance costs
* finance_years = financial lifetime of investments - the period over which
  the total costs are computed (years) starting from first investment year
* operation_maintenance_rate = fraction specifying the annual operation and maintenance costs
  relative to the investment cost
* CO2_price = costs for CO2 emissions (EUR/kgCO2)
* load_shed_penalty = penalty cost for load shedding (demand not supplied) (EUR/MWh)
* profiles_period_suffix = True/False specifying whether to use different profiles for each operating period, with a `_<period>` suffix to the profile name

## Analysis of results

There are some different ways to analyse the optimisation results:

* save to CSV file and analyse with tool of choice
* plot on map (candidate investments / stage 1 result / stage 2 results),
using ```powergim.SipModel.extractResultingGridData(...)``` and ```powergama.plots.plotMap(...)```
* inspect variables directly using Pyomo functionality (pyomo is the python
  package used to formulate the optimisation problem)


## More about the PowerGIM optimisation model

### Cost model

##### Investment cost

Branches, Nodes and generators:

* *cost_b = B + Bbd ⋅ b ⋅ d + Bd ⋅ d + Σ(Cp ⋅ p + C)*
    * The sum is over the two branch endpoints, that may be on land or at sea.
    * d = branch distance (km)
    * p = power rating (MW)
    * B = fixed cost (EUR)
    * Bdp = cost dependence on both distance and rating (EUR/km/MW)
    * Bd = cost dependence on distance (EUR/km)
    * C = fixed endpoint cost (CL=on land, CS=at sea) (EUR)
    * Cp = endpoint cost dependence on rating (EUR/MW)
* *cost_n = N*
    * N = fixed cost (NL=on land, NS=at sea)
* *cost_g = Gp ⋅ capacity*
    * Gp = generator cost per power rating (EUR/MW)

*Present value vs future value(s)* - Present value factor (pv) for translating
future value to present value, and annuity factor (a) for translating
future cash flow to present value:

* pv(r,T0) = 1/(1+r)^T0
* a(r,T) = 1/r ⋅ [1 - 1/(1+r)^T]
    * T0 = year of investment (0 for stage 1)
    * T = number of periods (years) (financeYears)
    * r = discount rate (financeInterestrate)

Operation and maintenance (O&M) and salvage factors:

* *om_factor = omRate ⋅ [a(r,T1)-a(r,T0)]*
    * omRate = annual O&M cost as fraction of investment cost
* *salvage_factor = T0/T(1/(1+r)^(T-T0))*
    * salvage value is the remaining value after the financial lifetime
  (non-zero for investments in stage 2, since they have more life left than
    stage 1 investments)

Present value of investments including O&M costs and salvage value:

*pv_cost_inv = Σcost ⋅ pv ⋅ (1 + om_factor - salvage_factor)*

The sum is over all investments

##### Operational cost
Costs per year are:

*cost_op = sum over time sample { Pg ⋅ (fuelcost + emissionrate ⋅ CO2price)  + Pshed ⋅ VOLL } ⋅ samplefactor*

* Pg = generator output (MW)
* fuelcost = generator cost (EUR/MWh), including time profile
* emissionrate = CO2 emissions per power output (kgCO2/MWh)
* CO2price = CO2 tax
* Pshed = load shed (MW)
* VOLL = value of lost load (load shedding cost) (EUR/MWh)
* samplefactor = number of hours represented by value in time sample (hours)

Present value of generation costs:

*pv_cost_op = Σ cost_op ⋅ a*

The sum is over all generators.


##### Total costs
Total costs that is the objective function in the optimisation problem:

*cost = pv_cost_inv + pv_cost_op*


### Power losses

power out = power in (lossFix + lossSlope*d)

* lossFix = loss factor, fixed part
* lossSlope = loss factor dependence on branch distance (1/km)

### Variables
These are the optimisation problem variables

* branchNewCapacity = capacity of new branches
* branchNewCables = number of new cables (integer)
* newNodes = number of new nodes (integer)
* genNewCapacity = new generation capacity
* branchFlow12 = power flow on branch in positive direction
* branchFlow21 = power flow on branch in negative direction
* generation = generator output
* loadShed = load shedding


### Constraints

Constraints are included for

1. Branch power flow is limited by branch capacity (in both directions)
2. New branch capacity requires new branches
3. A node is required at each branch endpoint
4. Generator output is limited by capacity and energy availability
5. Generator output average over entire time sample is limited by average power
available (energy limitation for storage generators)
6. CO2 emissions are limited by emission cap
7. Power balance at each node (branch power loss is included here)
