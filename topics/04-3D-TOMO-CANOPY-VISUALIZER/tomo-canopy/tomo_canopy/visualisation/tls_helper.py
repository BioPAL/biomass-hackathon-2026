from tomo_canopy.visualisation import TomographicPlotter

from tomo_canopy.types import (
    DataLayer,
    GeometryType,
    Modality,
)
from pathlib import Path

from tomo_canopy.processors import TLSProcessor


def get_tls_vol_data(voxel_size):
    TLS_CRS = "EPSG:32630"
    TLS_VOXEL_SIZE = voxel_size  # metres
    tls_candidates = [
        Path("data/c01_r02_subsampled_1cm_trees_mesh.ply"),
        # Path("../data/c01_r02_subsampled_1cm_trees_mesh.ply"),
    ]
    TLS_PATH = next(
        (path for path in tls_candidates if path.is_file()), tls_candidates[0]
    )

    tls_processor = TLSProcessor()
    tls_mesh = tls_processor.open(TLS_PATH, crs=TLS_CRS, load=False)
    tls_points = tls_mesh.drop_vars("face_vertices").rename({"vertex": "point"})
    tls_points.attrs = {**tls_mesh.attrs, "geometry_type": "point_cloud"}

    tls_dataset = tls_processor.voxelize(
        tls_points,
        TLS_VOXEL_SIZE,
        crs=TLS_CRS,
        bounds=tls_processor.bounds(tls_mesh),
        source=str(TLS_PATH.resolve()),
    )
    return DataLayer(
        name="tls_occupancy",
        data=tls_dataset,
        modality=Modality.TLS,
        geometry=GeometryType.VOLUME,
    ).data


def get_tls_point_data():
    TLS_CRS = "EPSG:32630"
    tls_candidates = [
        Path("data/c01_r02_subsampled_1cm_trees_mesh.ply"),
    ]
    TLS_PATH = next(
        (path for path in tls_candidates if path.is_file()), tls_candidates[0]
    )

    tls_processor = TLSProcessor()
    tls_mesh = tls_processor.open(TLS_PATH, crs=TLS_CRS, load=False)

    # Discard connectivity for this visualization while retaining every vertex.
    tls_points = tls_mesh.drop_vars("face_vertices").rename({"vertex": "point"})
    tls_points.attrs = {**tls_mesh.attrs, "geometry_type": "point_cloud"}
    return DataLayer(
        name="tls_points",
        data=tls_points,
        modality=Modality.TLS,
        geometry=GeometryType.POINT_CLOUD,
    ).data
