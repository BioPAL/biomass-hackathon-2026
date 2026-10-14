import geopandas as gpd
from shapely.geometry import box

from tomo_canopy.processors import GEDIProcessor


def get_gedi_points():
    # Example area in southwest Ghana. Replace this box with the study site.
    # GEDI_BOUNDS = (-2.70, 1.10, -2.50, 1.30)
    GEDI_BOUNDS = (-2.70, 5.10, -2.50, 5.30)

    roi = gpd.GeoDataFrame(
        geometry=[box(*GEDI_BOUNDS)],
        crs="EPSG:4326",
    )

    gedi_processor = GEDIProcessor()
    gedi = gedi_processor.open(
        roi,
        variables=["agbd", "rh"],
        start_time="2024-04-01",
        end_time="2024-07-25",
    )
    x = gedi["x"][:50]
    y = gedi["y"][:50]
    z = gedi["rh"][:50]
    print(len(x))
    return {
        "x": x,
        "y": y,
        "z": z,
    }
