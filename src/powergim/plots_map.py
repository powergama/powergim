import itertools
import math

import branca
import folium
import folium.plugins
import jinja2
import pandas as pd


def plot_map(
    pg_data,
    years,
    filename=None,
    nodetype=None,
    branchtype=None,
    filter_node=None,
    filter_branch=None,
    spread_nodes_r=None,
    include_zero_capacity=False,
    add_folium_control=True,
    **kwargs,
):
    """
    Plot on a map

    Parameters
    ==========
    pg_data : grid_data object
        powergama data object
    years : list
        investment years to include
    filename : str
        name of output file (html)
    nodetype : str ('nodalprice','area') or None (default)
        how to colour nodes
    branchtype : str ('utilisation','sensitivity','flow','capacity','type') or None (default)
        how to colour branches
    filter_node : list
        max/min value used for colouring nodes (e.g. nodalprice)
    filter_branch : list
        max/min value used for colouring branches (e.g. utilisation)
    spread_nodes_r : float (degrees)
        radius (degrees) of circle on which overlapping nodes are
        spread (use eg 0.04)
    include_zero_capacity : bool
        include branches and generators even if they have zero capacity (but are expandable)
    add_folium_node : bool
        include folium layer control.
    kwargs : arguments passed on to folium.Map(...)
    """

    cmSet1 = branca.colormap.linear.Set1_03

    # Add geographic information to branches and generators/consumers
    branch = pg_data.branch.copy()
    node = pg_data.node.copy()
    generator = pg_data.generator.copy()
    consumer = pg_data.consumer.copy()

    branch["capacity"] = branch[[f"capacity_{p}" for p in years]].sum(axis=1)
    generator["capacity"] = generator[[f"capacity_{p}" for p in years]].sum(axis=1)
    node["capacity"] = node[[f"capacity_{p}" for p in years]].sum(axis=1)
    branch["expand"] = branch[[f"expand_{p}" for p in years]].sum(axis=1)
    generator["expand"] = generator[[f"expand_{p}" for p in years]].sum(axis=1)
    node["expand"] = node[[f"expand_{p}" for p in years]].sum(axis=1)
    consumer["demand_avg"] = consumer[[f"demand_{p}" for p in years]].sum(axis=1)
    if f"flow_{years[-1]}" in branch.columns:
        # Use flow from the last period:
        branch["flow"] = branch[f"flow_{years[-1]}"]
        branch["utilisation"] = branch["flow"].abs() / branch["capacity"]

    if spread_nodes_r is not None:
        # spread out nodes lying on top of each other
        coords = node[["lat", "lon"]]
        dupl_coords = pd.DataFrame()
        dupl_coords["cumcount"] = coords.groupby(["lat", "lon"]).cumcount()
        dupl_coords["count"] = coords.groupby(["lat", "lon"])["lon"].transform("count")
        for i in node.index:
            n_sum = dupl_coords.loc[i, "count"]
            if n_sum > 1:
                # there are more nodes with the same coordinates
                n = dupl_coords.loc[i, "cumcount"]
                theta = 2 * math.pi / n_sum
                node.loc[i, "lat"] += spread_nodes_r * math.cos(n * theta)
                node.loc[i, "lon"] += spread_nodes_r * math.sin(n * theta)
        # node[['lat','lon']] = coords

    # Careful! Merge may change the order
    branch["index"] = branch.index
    branch = branch.merge(node[["id", "lat", "lon"]], how="left", left_on="node_from", right_on="id")
    branch = branch.merge(node[["id", "lat", "lon"]], how="left", left_on="node_to", right_on="id")
    branch.set_index("index")
    # Use node lat,lon only if generator lat,lon is unspecified (-1):
    generator = generator.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id", suffixes=("", "_node")
    )
    mask_latlon_given = ~(generator[["lat", "lon"]] == -1).any(axis=1)
    generator["lat"] = generator["lat"].where(mask_latlon_given, generator["lat_node"])
    generator["lon"] = generator["lon"].where(mask_latlon_given, generator["lon_node"])
    consumer = consumer.merge(node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id")
    gentypes = list(pg_data.generator["type"].unique())
    areas = list(pg_data.node["area"].unique())
    node = node.reset_index().rename(columns={"index": "index_orig"})
    node = node.merge(pd.DataFrame(areas).reset_index(), how="left", left_on="area", right_on=0).rename(
        columns={"index": "area_ind"}
    )
    node = node.set_index("index_orig")
    # node.sort_index(inplace=True)
    node["area_ind"] = 0.5 + node["area_ind"] % 10

    m = folium.Map(location=[node["lat"].median(), node["lon"].median()], **kwargs)
    sw = [node["lat"].min(), node["lon"].max()]
    ne = [node["lat"].max(), node["lon"].min()]
    m.fit_bounds([sw, ne])

    callbackNode = """function (row,colour) {
               if (colour=='') {
                   colour=row[3]
               }
               var marker = L.circleMarker(new L.LatLng(row[0],row[1]),
                                           {"radius":3,
                                            "color":colour} );
                      marker.bindPopup(row[2]);
                      return marker;
            }"""
    callbackBranch = """function (row,colour) {
                if (colour=='') {
                    colour=row[3]
                }
                var polyline = L.polyline([row[0],row[1]],
                                          {"color":colour} );
                polyline.bindPopup(row[2]);
                return polyline;
            }"""
    callbackFlowArrow = """function (row,colour) {
               if (colour=='') {
                   colour=row[3]
               }
               var marker = L.circleMarker(new L.LatLng(row[0],row[1]),
                                           {"radius":row[4],
                                            "color":colour} );
                      marker.bindPopup(row[2]);
                      return marker;
            }"""

    # print("Nodes...")
    if nodetype == "nodalprice":
        value_col = "nodalprice"
        if filter_node is None:
            filter_node = [node[value_col].min(), node[value_col].max()]
        cm_node = branca.colormap.LinearColormap(["green", "yellow", "red"], vmin=filter_node[0], vmax=filter_node[1])
        cm_node.caption = "Nodal price"
        m.add_child(cm_node)
    elif nodetype == "area":
        value_col = "area_ind"
        val_max = node[value_col].max()
        cm_node = cmSet1.scale(0, val_max).to_step(10)
        # cm_node.caption = 'Area'
        # m.add_child(cm_node)
    elif nodetype == "type":
        type_val, types = node["type"].factorize(sort=True)
        node["type_num"] = type_val
        value_col = "type_num"
        val_max = node[value_col].max()
        cm_node = cmSet1.scale(0, val_max).to_step(10)
    else:
        value_col = None

    locationsN = []
    for i, n in node.iterrows():
        if (not include_zero_capacity) and (n["capacity"] == 0):
            pass
        elif include_zero_capacity and (n["capacity"] + n["expand"] == 0):
            # skip zero elements unless they are expandable
            pass
        elif not (n[["lat", "lon"]].isnull().any()):
            data = [n["lat"], n["lon"], "Node={}, area={}".format(n["id"], n["area"])]
            if value_col is not None:
                colHex = cm_node(n[value_col])
                data.append(colHex)
                colour = ""
            else:
                colour = "blue"
            if "capacity" in n:
                data[2] = f"{data[2]}, capacity={n['capacity']:g}"

            locationsN.append(data)
        else:
            print("Missing lat/lon for node index={}".format(i))
    feature_group_Nodes = folium.FeatureGroup(name="Nodes").add_to(m)
    FeatureCollection(data=locationsN, callback=callbackNode, addto=feature_group_Nodes, colour=colour).add_to(
        feature_group_Nodes
    )

    # print("Branches...")
    if branchtype == "utilisation":
        value_col = "utilisation"
        if filter_branch is None:
            filter_branch = [0, 1]
        cm_branch = branca.colormap.LinearColormap(
            ["green", "yellow", "red"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch utilisation"
        m.add_child(cm_branch)
    elif branchtype == "sensitivity":
        value_col = "sensitivity"
        if filter_branch is None:
            filter_branch = [branch[value_col].min(), branch[value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch capacity sensitivity"
        m.add_child(cm_branch)
    elif branchtype == "flow":
        value_col = "flow"
        if filter_branch is None:
            filter_branch = [branch[value_col].min(), branch[value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch flow (abs value)"
        m.add_child(cm_branch)
    elif branchtype == "capacity":
        value_col = "capacity"
        if filter_branch is None:
            filter_branch = [branch[value_col].min(), branch[value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch capacity"
        m.add_child(cm_branch)
    elif branchtype == "type":
        type_val, types = branch["type"].factorize(sort=True)
        # print(types)
        branch["type_num"] = type_val
        # print(branch[["type", "type_num"]])
        value_col = "type_num"
        val_max = branch[value_col].max()
        cm_branch = cmSet1.scale(0, val_max).to_step(10)
    else:
        value_col = None
    locationsB = []
    locationsB_flow_direction = []
    for i, n in branch.iterrows():
        if (not include_zero_capacity) and (n["capacity"] == 0):
            # skip this branch
            pass
        elif include_zero_capacity and (n["capacity"] + n["expand"] == 0):
            # skip zero elements unless they are expandable
            pass
        elif not (n[["lat_x", "lon_x", "lat_y", "lon_y"]].isnull().any()):
            data = [
                [n["lat_x"], n["lon_x"]],
                [n["lat_y"], n["lon_y"]],
                f"Branch={i} ({n['node_from']}-{n['node_to']}), type = {n['type']}, capacity={n['capacity']:g}",
            ]
            if "flow" in n:
                str_flows = "<br>".join(
                    f"flow12_{y}={n[f'flow12_{y}']:g}, flow21_{y}={n[f'flow21_{y}']:g}" for y in years
                )
                # data[2] = f"{data[2]}, flow={n['flow']:g}"
                data[2] = f"{data[2]},<br>{str_flows} <br>flow={n[f'flow']:g} <br>utilisation={n[f'utilisation']:g}"

                # flow direction marker:
                # if f"flow12_{years[-1]}" in n:
                if n[f"flow12_{years[-1]}"] + n[f"flow21_{years[-1]}"] == 0:
                    d = 0.5
                else:
                    d = min(
                        0.9, max(0.1, n[f"flow12_{years[-1]}"] / (n[f"flow12_{years[-1]}"] + n[f"flow21_{years[-1]}"]))
                    )
                (flow_lat, flow_lon) = _pointBetween((n["lat_x"], n["lon_x"]), (n["lat_y"], n["lon_y"]), weight=d)
                radius = 6  # node radius=3
                data2 = [
                    flow_lat,
                    flow_lon,
                    f"flow12_{years[-1]}: {n[f'flow12_{years[-1]}']:8.2f}<br>flow21_{years[-1]}: {n[f'flow21_{years[-1]}']:8.2f}",
                    colour,
                    radius,
                ]
                locationsB_flow_direction.append(data2)
            if value_col is not None:
                colHex = cm_branch(n[value_col])
                data.append(colHex)
                colour = ""
            else:
                colour = "black"
            locationsB.append(data)
        else:
            print("Missing lat/lon for node index={}".format(i))

    feature_group_Branches = folium.FeatureGroup(name="Branches").add_to(m)
    FeatureCollection(locationsB, callback=callbackBranch, addto=feature_group_Branches, colour=colour).add_to(
        feature_group_Branches
    )
    if f"flow_{years[-1]}" in branch.columns:
        feature_group_Branches_flowdirection = folium.FeatureGroup(name="flow direction").add_to(feature_group_Branches)
        FeatureCollection(
            locationsB_flow_direction, callback=callbackFlowArrow, addto=feature_group_Branches_flowdirection, colour=""
        ).add_to(feature_group_Branches_flowdirection)

    # print("Consumers...")
    locationsN = []
    for i, n in consumer.iterrows():
        if not (n[["lat", "lon"]].isnull().any()):
            locationsN.append(
                [
                    n["lat"],
                    n["lon"],
                    "Consumer {} at node={}, avg demand={:g} ({})".format(
                        i, n["node"], n["demand_avg"], n["demand_ref"]
                    ),
                ]
            )
        else:
            print("Missing lat/lon for node index={}".format(i))
    feature_group_Consumer = folium.FeatureGroup(name="Consumers").add_to(m)
    FeatureCollection(
        data=locationsN,
        callback=callbackNode,
        addto=feature_group_Consumer,
        colour="blue",
    ).add_to(feature_group_Consumer)

    # print("Generators...")
    # feature_group_Generator = folium.FeatureGroup(name='Generators').add_to(m)
    ngtypes = max(2, len(gentypes))
    cm_stepG = cmSet1.scale(0, ngtypes - 1).to_step(ngtypes)

    groups = generator.groupby("node")
    feature_group_Generators = folium.FeatureGroup(name="Generators").add_to(m)
    gencluster_icon_create_function = """\
    function(cluster) {
        return L.divIcon({
        html: '<b>' + cluster.getChildCount() + '</b>',
        className: 'marker-cluster marker-cluster-large',
        iconSize: new L.Point(20, 20)
        });
    }"""
    for thenode, genindices in groups.groups.items():
        locationsN = []
        locationsG = []
        options = dict(maxClusterRadius=1)  # only cluster if lat/lon are the same.
        marker_cluster = folium.plugins.MarkerCluster(
            icon_create_function=gencluster_icon_create_function, options=options
        )
        marker_cluster.add_to(feature_group_Generators)
        for genind in genindices:
            n = generator.loc[genind]
            gentype = n["type"]
            typeind = gentypes.index(gentype)
            if (not include_zero_capacity) and (n["capacity"] == 0):
                pass
            elif include_zero_capacity and (n["capacity"] + n["expand"] == 0):
                # skip zero elements unless they are expandable
                pass
            elif not (n[["lat", "lon"]].isnull().any()):
                data = [
                    n["lat"],
                    n["lon"],
                    "{}<br>Generator {}: {}, pmax={:g}".format(gentype, genind, n["desc"], n["capacity"]),
                ]
                col = cm_stepG(typeind)
                data.append(col)
                dataG = [
                    [n["lat"], n["lon"]],
                    [n["lat_node"], n["lon_node"]],
                    "{}<br>Generator {}: {}, pmax={:g}".format(gentype, genind, n["desc"], n["capacity"]),
                ]
                if f"output_{years[-1]}" in n:
                    str_output = "<br>".join(f"output_{y}={n[f'output_{y}']:g}" for y in years)
                    data[2] = f"{data[2]},<br>{str_output}"
                    dataG[2] = f"{dataG[2]},<br>{str_output}"
                locationsN.append(data)
                locationsG.append(dataG)

            else:
                print("Missing lat/lon for node index={}".format(i))

        # feature_group_GenX = folium.FeatureGroup(name=gentype).add_to(m)
        # FeatureCollection(data=locationsN,callback=callbackNode,
        #                  featuregroup=feature_group_GenX,
        #                  colour="red").add_to(feature_group_GenX)
        FeatureCollection(data=locationsN, callback=callbackNode, addto=marker_cluster, colour="").add_to(
            marker_cluster
        )
        # green line from generator to node:
        FeatureCollection(data=locationsG, callback=callbackBranch, addto=marker_cluster, colour="green").add_to(
            marker_cluster
        )

    if add_folium_control:
        folium.LayerControl().add_to(m)

    if filename:
        print("Saving map to file {}".format(filename))
        m.save(filename)

    return m


class FeatureCollection(folium.map.FeatureGroup):
    """
    Add features to a map using in-browser rendering.

    Parameters
    ----------
    data : list
        List of list of shape [[], []]. Data points should be of
        the form [[lat, lng]].
    callback : string, default None
        A string representation of a valid Javascript function
        that will be passed a lat, lon coordinate pair.
    featuregroup : folium.FeatureGroup
        Feature group
    colour : string
        colour
    name : string, default None
        The name of the Layer, as it will appear in LayerControls.
    overlay : bool, default True
        Adds the layer as an optional overlay (True) or the base layer (False).
    control : bool, default True
        Whether the Layer will be included in LayerControls.
    """

    _counts = itertools.count(0)

    def __init__(self, data, callback, addto, colour, name=None, overlay=True, control=True):
        super(FeatureCollection, self).__init__(name=name, overlay=overlay, control=control)
        self._name = "FeatureCollection"
        self._data = data
        self._addto = addto
        self._colour = colour
        self._count = next(self._counts)
        self._callback = "var callback{} = {};".format(self._count, callback)

        self._template = jinja2.Template(
            """
            {% macro script(this, kwargs) %}
            {{this._callback}}
            (function(){
                var data = {{this._data}};
                //var map = {{this._parent.get_name()}};
                var addto = {{this._addto.get_name()}};
                var colour = '{{this._colour}}';
                for (var i = 0; i < data.length; i++) {
                    var row = data[i];
                    var feature = callback"""
            + "{}".format(self._count)
            + """(row,colour);
                    feature.addTo(addto);
                };
            })();
            {% endmacro %}"""
        )


def _pointBetween(nodeA, nodeB, weight):
    """computes coords on the line between two points

     (lat,lon) = pointBetween(self,lat1,lon1,lat2,lon2,d)

    Parameters
    ----------
        nodeA: dublet (lat,lon)
            latitude/longitude of nodeA (degrees)
        nodeB: dublet (lat,lon)
            latitude/longitude of nodeB (degrees)
        weight: double
            weight=0 is node A, 0.5 is halfway between, and 1 is node B

    Returns
    -------
        (lat,lon): dublet
            coordinates of the point inbetween nodeA and nodeB (degrees)


     ref: http://www.movable-type.co.uk/scripts/latlong.html
    """
    lat1 = nodeA[0]
    lon1 = nodeA[1]
    lat2 = nodeB[0]
    lon2 = nodeB[1]
    if (lat1 == lat2) and (lon1 == lon2):
        lat = lat1
        lon = lon1
    else:
        # transform to radians
        lat1 = lat1 * math.pi / 180
        lat2 = lat2 * math.pi / 180
        lon1 = lon1 * math.pi / 180
        lon2 = lon2 * math.pi / 180

        # initial bearing
        y = math.sin(lon2 - lon1) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        bearing = math.atan2(y, x)

        # angular distance from A to B
        d_tot = math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
        d = d_tot * weight

        lat = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing))
        lon = lon1 + math.atan2(
            math.sin(bearing) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat)
        )

        # tansform to degrees
        lat = lat * 180 / math.pi
        lon = lon * 180 / math.pi
    return (lat, lon)
