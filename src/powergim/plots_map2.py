import geopandas
import matplotlib.pyplot as plt
import shapely

from powergim.grid_data import GridData


def plot_map2(grid_data: GridData, shapefile_path: str, ax=None, **kwargs):

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

    grid_data.node.plot.scatter(ax=ax, x="lon", y="lat", size=10, color="red")

    gdf_nodes = geopandas.GeoDataFrame(
        grid_data.node, geometry=geopandas.points_from_xy(grid_data.node["lon"], grid_data.node["lat"]), crs="EPSG:4326"
    )
    gdf_edges = grid_data.branch.merge(
        gdf_nodes[["lat", "lon", "geometry"]], how="left", left_on="node_from", right_index=True
    ).merge(gdf_nodes[["lat", "lon", "geometry"]], how="left", left_on="node_to", right_index=True)
    gdf_edges_geometry = gdf_edges.apply(
        lambda x: shapely.geometry.LineString([x["geometry_x"], x["geometry_y"]]), axis=1
    )
    gdf_edges = geopandas.GeoDataFrame(gdf_edges, geometry=gdf_edges_geometry, crs="EPSG:4326")
    gdf_edges.plot(ax=ax, **kwargs)

    ax.set_xlim(grid_data.node["lon"].min() - 1, grid_data.node["lon"].max() + 1)
    ax.set_ylim(grid_data.node["lat"].min() - 1, grid_data.node["lat"].max() + 1)
    ax.set(xlabel=None, ylabel=None)
    return
