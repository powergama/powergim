import math

import pandas as pd


class GridData(object):
    """
    Class for grid data storage and import
    """

    def __init__(self, investment_years, node, branch, generator, consumer, profiles=None):
        """ """
        self.node = node
        self.branch = branch
        self.generator = generator
        self.consumer = consumer
        self.profiles = profiles
        self.investment_years = investment_years

        # Required fields for investment analysis input data
        # value == None: Required column, value must be specified in input
        # value == -1: Optional input, value to be computed by program
        # value == <default value>: Optional input, if not specified, use the default
        self.keys_sipdata = {
            "node": {
                "id": None,
                "area": None,
                **{f"capacity_{p}": None for p in self.investment_years},
                **{f"expand_{p}": None for p in self.investment_years},
                "lat": None,
                "lon": None,
                "offshore": None,
                "type": None,
                "cost_scaling": None,
            },
            "branch": {
                "node_from": None,
                "node_to": None,
                **{f"capacity_{p}": None for p in self.investment_years},
                **{f"expand_{p}": None for p in self.investment_years},
                "max_newCap": -1,
                "distance": -1,
                "cost_scaling": None,
                "type": None,
            },
            "generator": {
                "type": None,
                "node": None,
                "lat": -1,
                "lon": -1,
                "desc": "",
                **{f"capacity_{p}": None for p in self.investment_years},
                **{f"expand_{p}": None for p in self.investment_years},
                "pmin": None,
                "allow_curtailment": None,
                "p_maxNew": -1,
                "cost_scaling": 1,
                **{f"fuelcost_{p}": None for p in self.investment_years},
                "fuelcost_ref": None,
                "pavg": 0,
                "inflow_fac": None,
                "inflow_ref": None,
            },
            "consumer": {
                "node": None,
                **{f"demand_{p}": None for p in self.investment_years},
                "demand_ref": None,
            },
        }

    def validate_grid_data(self):
        self._checkGridDataFields(self.keys_sipdata)
        self._checkGridData()
        self._addDefaultColumns(keys=self.keys_sipdata)
        self._fillEmptyCells(keys=self.keys_sipdata)

    def _fillEmptyCells(self, keys):
        """Use default data where none is given"""
        # generators:
        for col, val in keys["generator"].items():
            if val is not None:
                self.generator[col] = self.generator[col].fillna(keys["generator"][col])
        # consumers:
        for col, val in keys["consumer"].items():
            if val is not None:
                self.consumer[col] = self.consumer[col].fillna(keys["consumer"][col])

        # branches:
        for col, val in keys["branch"].items():
            if val is not None:
                self.branch[col] = self.branch[col].fillna(keys["branch"][col])

        # insert computed distances if not provided in input (and given default value -1 above)
        distances = pd.Series(index=self.branch.index, data=self.compute_branch_distances())
        self.branch["distance"].where(self.branch["distance"] != -1, distances, inplace=True)

    def _addDefaultColumns(self, keys, remove_extra_columns=False):
        """insert optional columns with default values when none
        are provided in input files"""
        for k in keys["generator"]:
            if k not in self.generator.keys():
                self.generator[k] = keys["generator"][k]
        for k in keys["consumer"]:
            if k not in self.consumer.keys():
                self.consumer[k] = keys["consumer"][k]
        for k in keys["branch"]:
            if k not in self.branch.keys():
                self.branch[k] = keys["branch"][k]

        # Discard extra columns (comments etc)
        if remove_extra_columns:
            self.node = self.node[list(keys["node"].keys())]
            self.branch = self.branch[list(keys["branch"].keys())]
            self.generator = self.generator[list(keys["generator"].keys())]
            self.consumer = self.consumer[list(keys["consumer"].keys())]

    def _checkGridDataFields(self, keys):
        """check if all required columns are present
        (ie. all columns with no default value)"""
        for k, v in keys["node"].items():
            if v is None and k not in self.node:
                raise Exception("Node input file must contain %s" % k)
        for k, v in keys["branch"].items():
            if v is None and k not in self.branch:
                raise Exception("Branch input file must contain %s" % k)
        for k, v in keys["generator"].items():
            if v is None and k not in self.generator:
                raise Exception("Generator input file must contain %s" % k)
        for k, v in keys["consumer"].items():
            if v is None and k not in self.consumer:
                raise Exception("Consumer input file must contain %s" % k)

    def _checkGridData(self):
        """Check consistency of grid data"""

        # generator nodes
        for g in self.generator["node"]:
            if g not in self.node["id"].values:
                raise Exception("Generator node does not exist: '%s'" % g)
        # consumer nodes
        for c in self.consumer["node"]:
            if c not in self.node["id"].values:
                raise Exception("Consumer node does not exist: '%s'" % c)

        # branch nodes
        for c in self.branch["node_from"]:
            if c not in self.node["id"].values:
                raise Exception("Branch from node does not exist: '%s'" % c)
        for c in self.branch["node_to"]:
            if c not in self.node["id"].values:
                raise Exception("Branch to node does not exist: '%s'" % c)

    def compute_branch_distances(self, R=6373.0):
        """computes branch distance from node coordinates, resuls in km

        Uses haversine formula

        Parameters
        ----------
        R : radius of the Earth
        """

        # approximate radius of earth in km
        n_from = self.get_branch_from_node_index()
        n_to = self.get_branch_to_node_index()
        distance = []
        # get endpoint coordinates and convert to radians
        lats1 = self.node.loc[n_from, "lat"].apply(math.radians)
        lons1 = self.node.loc[n_from, "lon"].apply(math.radians)
        lats2 = self.node.loc[n_to, "lat"].apply(math.radians)
        lons2 = self.node.loc[n_to, "lon"].apply(math.radians)
        lats1.index = self.branch.index
        lons1.index = self.branch.index
        lats2.index = self.branch.index
        lons2.index = self.branch.index

        for b in self.branch.index:
            lat1 = lats1[b]
            lon1 = lons1[b]
            lat2 = lats2[b]
            lon2 = lons2[b]

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            # atan2 better than asin: c = 2 * math.asin(math.sqrt(a))
            distance.append(R * c)
        return distance

    def get_branch_from_node_index(self):
        """get node indices for branch FROM node"""
        return [self.node[self.node["id"] == b["node_from"]].index.tolist()[0] for i, b in self.branch.iterrows()]

    def get_branch_to_node_index(self):
        """get node indices for branch TO node"""
        return [
            self.node[self.node["id"] == self.branch["node_to"][k]].index.tolist()[0]
            for k in self.branch.index.tolist()
        ]
