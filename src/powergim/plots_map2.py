import geopandas
import matplotlib.pyplot as plt
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
        print(ex, ">> Missing shape file. Download from https://www.naturalearthdata.com/downloads/")
        return
    try:
        borders_sea = geopandas.read_file(f"{shapefile_path}/World_maritime_Boundaries.zip")
    except Exception as ex:
        print(
            ex, ">> Missing shape file. Download from https://hub.arcgis.com/datasets/nga::world-maritime-boundaries/"
        )
        return

    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    ax.set_facecolor("white")
    lakes.plot(ax=ax, color="white")
    land.plot(ax=ax, color="lightgray")
    coastline.plot(ax=ax, color="darkgray")
    borders.plot(ax=ax, color="darkgray")
    borders_sea.plot(ax=ax, color="darkgray")

    branch = grid_data.branch.copy()
    node = grid_data.node.copy()
    # generator = grid_data.generator.copy()
    # consumer = grid_data.consumer.copy()
    if years is not None:
        branch["capacity"] = branch[[f"capacity_{p}" for p in years]].sum(axis=1)
        # generator["capacity"] = generator[[f"capacity_{p}" for p in years]].sum(axis=1)
        node["capacity"] = node["capacity"] + node[[f"capacity_{p}" for p in years if f"capacity_{p}" in node]].sum(
            axis=1
        )
    if not include_zero_capacity:
        branch = branch[branch["capacity"] > 0]
        node = node[node["capacity"] > 0]

    # node.plot.scatter(ax=ax, x="lon", y="lat", size=10, color="red")

    gdf_nodes = geopandas.GeoDataFrame(
        grid_data.node, geometry=geopandas.points_from_xy(grid_data.node["lon"], grid_data.node["lat"]), crs="EPSG:4326"
    )
    branch["index"] = branch.index
    gdf_edges = branch.merge(
        gdf_nodes[["lat", "lon", "geometry"]], how="left", left_on="node_from", right_index=True
    ).merge(gdf_nodes[["lat", "lon", "geometry"]], how="left", left_on="node_to", right_index=True)
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
    if width_col is not None:
        kwargs["linewidth"] = (gdf_edges[width_col[0]] / width_col[1]).clip(upper=width_col[2])
    gdf_edges.plot(ax=ax, **kwargs)
    if node_options is None:
        node_options = {}
    gdf_nodes.plot(ax=ax, **node_options)

    ax.set_xlim(grid_data.node["lon"].min() - 1, grid_data.node["lon"].max() + 1)
    ax.set_ylim(grid_data.node["lat"].min() - 1, grid_data.node["lat"].max() + 1)
    ax.set(xlabel=None, ylabel=None)
    return
