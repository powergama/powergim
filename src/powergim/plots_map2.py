import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import pyproj
import shapely

from powergim.grid_data import GridData


def plot_map2(
    grid_data: GridData,
    years,
    shapefile_path: str,
    ax=None,
    include_zero_capacity=False,
    width_col=None,
    node_options=None,
    include_generators=None,
    gen_options=None,
    latlon=None,
    proj="EPSG:4326",
    **kwargs,
):
    """Plot grid using geopandas (for print)"""

    # TODO: Make it more general
    try:
        land = geopandas.read_file(f"{shapefile_path}/ne_50m_land.zip")
        lakes = geopandas.read_file(f"{shapefile_path}/ne_50m_lakes.zip")
        coastline = geopandas.read_file(f"{shapefile_path}/ne_50m_coastline.zip")
        borders = geopandas.read_file(f"{shapefile_path}/ne_50m_admin_0_boundary_lines_land.zip")
    except Exception as ex:
        print(
            ex,
            ">> Missing shape file. Download from https://www.naturalearthdata.com/downloads/",
        )
        return
    try:
        borders_sea = geopandas.read_file(f"{shapefile_path}/World_maritime_Boundaries.zip")
    except Exception as ex:
        print(
            ex,
            ">> Missing shape file. Download from https://hub.arcgis.com/datasets/nga::world-maritime-boundaries/",
        )
        return

    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    ax.set_facecolor("white")
    lakes.to_crs(proj).plot(ax=ax, color="white")
    land.to_crs(proj).plot(ax=ax, color="lightgray")
    coastline.to_crs(proj).plot(ax=ax, color="darkgray")
    borders.to_crs(proj).plot(ax=ax, color="darkgray")
    borders_sea.to_crs(proj).plot(ax=ax, color="darkgray")

    branch = grid_data.branch.copy()
    node = grid_data.node.copy()
    generator = grid_data.generator.copy()
    # consumer = grid_data.consumer.copy()
    if years is not None:
        branch["capacity"] = branch[[f"capacity_{p}" for p in years]].sum(axis=1)
        generator["capacity"] = generator[[f"capacity_{p}" for p in years]].sum(axis=1)
        node["capacity"] = node[[f"capacity_{p}" for p in years]].sum(axis=1)

    gdf_nodes = geopandas.GeoDataFrame(
        node,
        geometry=geopandas.points_from_xy(node["lon"], node["lat"]),
        crs="EPSG:4326",
    )
    branch["index"] = branch.index
    gdf_edges = branch.merge(
        gdf_nodes[["lat", "lon", "geometry"]],
        how="left",
        left_on="node_from",
        right_index=True,
    ).merge(
        gdf_nodes[["lat", "lon", "geometry"]],
        how="left",
        left_on="node_to",
        right_index=True,
    )
    # TODO: This gives shapely deprecation warning (issue 13)
    gdf_edges_geometry = gdf_edges.apply(
        lambda x: shapely.geometry.LineString([x["geometry_x"], x["geometry_y"]]),
        axis=1
        # lambda x: [[x["lon_x"],x["lat_x"]],[x["lon_y"],x["lat_y"]]], axis=1
    )
    #    gdf_edges["geometry"] = gdf_edges_geometry
    #    gdf_edges = geopandas.GeoDataFrame(gdf_edges, geometry="geometry", crs="EPSG:4326")
    gdf_edges = geopandas.GeoDataFrame(gdf_edges, geometry=gdf_edges_geometry, crs="EPSG:4326")
    gdf_edges.set_index("index")

    # Use node lat,lon only if generator lat,lon is unspecified (-1):
    generator = generator.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id", suffixes=("", "_node")
    )
    mask_latlon_given = ~(generator[["lat", "lon"]] == -1).any(axis=1)
    generator["lat"] = generator["lat"].where(mask_latlon_given, generator["lat_node"])
    generator["lon"] = generator["lon"].where(mask_latlon_given, generator["lon_node"])
    gdf_generators = geopandas.GeoDataFrame(
        generator,
        geometry=geopandas.points_from_xy(generator["lon"], generator["lat"]),
        crs="EPSG:4326",
    )

    if not include_zero_capacity:
        gdf_edges = gdf_edges[gdf_edges["capacity"] > 0]
        gdf_nodes = gdf_nodes[gdf_nodes["capacity"] > 0]
        gdf_generators = gdf_generators[gdf_generators["capacity"] > 0]

    if width_col is not None:
        kwargs["linewidth"] = (gdf_edges[width_col[0]] / width_col[1]).clip(upper=width_col[2])
    if not gdf_edges.empty:
        gdf_edges.to_crs(proj).plot(ax=ax, **kwargs)
    if node_options is None:
        node_options = {}
    if not gdf_nodes.empty:
        gdf_nodes.to_crs(proj).plot(ax=ax, **node_options)
    else:
        print("DEBUG: No new nodes to plot")

    if include_generators:
        # TODO this messes up gen_options markersize and alpha vectors
        m_gen_keep = gdf_generators["type"].isin(include_generators)
        gdf_generators = gdf_generators[m_gen_keep]
        ind_gen_keep = gdf_generators.index
        if ("alpha" in gen_options) and (type(gen_options["alpha"]) == pd.Series):
            gen_options["alpha"] = gen_options["alpha"].loc[ind_gen_keep]
        if ("markersize" in gen_options) and (type(gen_options["markersize"]) == pd.Series):
            # print("gen markersize:",gen_options["markersize"].shape)
            gen_options["markersize"] = gen_options["markersize"].loc[ind_gen_keep]

        if gen_options is None:
            gen_options = {}
        if not gdf_generators.empty:
            gdf_generators.to_crs(proj).plot(ax=ax, **gen_options)
            gdf_gen_edges = gdf_generators.copy()
            gdf_gen_edges["geometry_x"] = geopandas.points_from_xy(gdf_gen_edges["lon"], gdf_gen_edges["lat"])
            gdf_gen_edges["geometry_y"] = geopandas.points_from_xy(gdf_gen_edges["lon_node"], gdf_gen_edges["lat_node"])
            gdf_gen_edges_geometry = gdf_gen_edges.apply(
                lambda x: shapely.geometry.LineString([x["geometry_x"], x["geometry_y"]]), axis=1
            )
            gdf_gen_edges = geopandas.GeoDataFrame(gdf_gen_edges, geometry=gdf_gen_edges_geometry, crs="EPSG:4326")
            # gdf_gen_edges.set_index("index")
            gdf_gen_edges.to_crs(proj).plot(ax=ax, zorder=0, **gen_options)

    if latlon is None:
        latlon = {
            "lat": (grid_data.node["lat"].min() - 1, grid_data.node["lat"].max() + 1),
            "lon": (grid_data.node["lon"].min() - 1, grid_data.node["lon"].max() + 1),
        }
    p1 = pyproj.Proj(proj, preserve_units=False)
    minmax = p1(latlon["lon"], latlon["lat"])
    ax.set_xlim(minmax[0][0], minmax[0][1])
    ax.set_ylim(minmax[1][0], minmax[1][1])
    ax.set(xlabel=None, ylabel=None)
    return
