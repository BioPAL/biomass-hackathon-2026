from dataclasses import dataclass
from tomo_canopy.types import Modality, DataLayer
import numpy as np
import pyvista as pv
import matplotlib as mpl


from tomo_canopy.visualisation.basic_plotter_app import BasicPlotterApp

MODES = ("raw_representation", "voxel_as_cubes", "voxel_as_bubbles")
LABELS = {"raw_representation": "Raw Data", "voxel_as_cubes": "Voxel (Cubes)", "voxel_as_bubbles": "Voxel (Bubbles)"}


DATA_LAYERS = ["TLS", "GEDI"]


@dataclass
class DataGetter:
    get_vol_data_fn: callable
    get_points_fn: callable


class TomographicPlotter(BasicPlotterApp):
    MAIN = "tomographic_plotter"

    def __init__(self, data_getter_dict: dict[str, DataGetter], mode="voxel_as_bubbles"):
        self.mode = mode
        self.modality = "TLS"
        self.voxel_res = 10.0
        self.data_getter_dict = data_getter_dict
        super().__init__()

    def build_ui(self):
        for _, mode in enumerate(MODES):
            self.add_checkbox(
                mode, LABELS[mode.lower()], self._make_mode_callback(mode), value=(mode == self.mode)
            )
        self.increase_btn_y()
        self.increase_btn_y()
        for data_layer in DATA_LAYERS:
            self.add_checkbox(
                data_layer,
                data_layer,
                self._make_modality_callback(data_layer),
                value=(data_layer == self.modality),
            )

        self.pl.add_slider_widget(
            self._on_vox_res_change,
            rng=[0.5, 20.0],
            value=self.voxel_res,
            title="Voxel/Points Resolution",
            pointa=(0.60, 0.94),
            pointb=(0.97, 0.94),
            style="modern",
            fmt="%.1f",
            interaction_event="end",  # rebuild on release, not on every drag step
        )

    def _make_mode_callback(self, mode):
        def callback(_state):
            self.set_mode(mode)

        return callback

    def _make_modality_callback(self, modality):
        def callback(_state):
            self.set_modality(modality)

        return callback

    def _on_vox_res_change(self, value):
        self.voxel_res = float(value)
        self.render()

    def set_mode(self, m):
        self.mode = m
        for _, mode in enumerate(MODES):
            self.buttons[mode]["value"] = mode == self.mode
        self.render()

    def set_modality(self, m):
        self.modality = m
        for _, data_layer in enumerate(DATA_LAYERS):
            self.buttons[data_layer]["value"] = data_layer == self.modality
        self.render()

    def on_render(self):
        pl = self.pl
        common = dict(name=self.MAIN, reset_camera=True)

        cmap = "viridis"

        if self.mode == "raw_representation":
            data = self.data_getter_dict[self.modality].get_points_fn()
            x, y, z = (
                data["x"],
                data["y"],
                data["z"],
            )

            if z.ndim == 2:
                new_x = []
                new_y = []
                new_z = []
                print("i" + str(x[0]) + " j" + str(x.shape))
                print("i" + str(y.shape) + " j" + str(y.shape))
                print("i" + str(z.shape) + " j" + str(z.shape))
                for i in range(z.shape[0]):
                    for j in range(z.shape[1]):
                        nx = float(x[i].x)
                        ny = float(y[i].y)
                        nz = float(z[i][j])
                        new_x.append(nx)
                        new_y.append(ny)
                        new_z.append(nz)
                x = np.array(new_x)
                y = np.array(new_y)
                z = np.array(new_z)
            points = np.column_stack((x, y, z))  # shape (N, 3)
            cloud = pv.PolyData(points)
            cloud["elevation"] = z
            pl.add_mesh_clip_plane(
                cloud,
                scalars="elevation",
                cmap=cmap,
                point_size=self.voxel_res,
                render_points_as_spheres=True,
                **common,
            )
        elif self.mode == "voxel_as_bubbles":
            ds = self.data_getter_dict[self.modality].get_vol_data_fn(self.voxel_res)
            voxel_size = float(ds.attrs.get("voxel_size", 5.0))

            counts = ds["point_count"]
            occupied = counts.values > 0

            # Centres of occupied voxels, as a flat point cloud
            xc = np.broadcast_to(counts.x.values[None, None, :], counts.shape)[occupied]
            yc = np.broadcast_to(counts.y.values[None, :, None], counts.shape)[occupied]
            zc = np.broadcast_to(counts.z.values[:, None, None], counts.shape)[occupied]
            centres = np.column_stack((xc, yc, zc))
            values = counts.values[occupied].astype(np.float32)

            cloud = pv.PolyData(centres)
            cloud["point_count"] = values
            # Scale factor per bubble: 0.5 → radius = half voxel edge, so
            # neighbours touch but don't overlap; sqrt softens density blow-up
            cloud["scale"] = 0.5 * voxel_size * np.sqrt(values / values.max())

            bubbles = cloud.glyph(
                orient=False,
                scale="scale",
                factor=2.0,  # multiplies 'scale'; tune per dataset
                geom=pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=12),
            )

            pl.add_mesh_clip_plane(
                bubbles,
                scalars="point_count",
                cmap=cmap,
                scalar_bar_args={"title": "points per voxel"},
                **common,
            )
        elif self.mode == "voxel_as_cubes":
            ds = self.data_getter_dict[self.modality].get_vol_data_fn(self.voxel_res)

            x = ds["x"].values - ds["x"].values[0]
            y = ds["y"].values - ds["y"].values[0]
            z = ds["z"].values - ds["z"].values[0]
            dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]

            grid = pv.ImageData(
                dimensions=(len(x) + 1, len(y) + 1, len(z) + 1),
                spacing=(dx, dy, dz),
                origin=(-dx / 2, -dy / 2, -dz / 2),
            )

            def to_cells(name, dtype=None):
                a = ds[name].transpose("x", "y", "z").values.ravel(order="F")
                return a.astype(dtype) if dtype else a

            grid.cell_data["point_count"] = to_cells("point_count")
            grid.cell_data["occupancy"] = to_cells("occupancy", np.uint8)

            # Keep only occupied voxels
            voxels = grid.threshold(0.5, scalars="occupancy")
            xmin, xmax, ymin, ymax, zmin, zmax = voxels.bounds

            counts = voxels.cell_data["point_count"].astype(float)

            # Normalise; the 99th percentile stops a few dense voxels from squashing everything else
            vmin, vmax = counts.min(), np.percentile(counts, 99)
            norm = np.clip((counts - vmin) / (vmax - vmin), 0, 1)

            gamma = 0.5  # <1 lifts faint voxels, >1 suppresses them
            alpha_min, alpha_max = 0.15, 0.95

            rgba = mpl.colormaps[cmap](norm)  # cmap given as a name, e.g. "viridis"
            rgba[:, 3] = alpha_min + (alpha_max - alpha_min) * norm**gamma
            voxels.cell_data["rgba"] = (rgba * 255).astype(np.uint8)

            pl.add_mesh_clip_plane(
                voxels,
                scalars="rgba",
                rgba=True,  # use the array as-is, including alpha
                normal="z",
                origin=(voxels.center[0], voxels.center[1], zmax),
                invert=False,
                show_edges=False,
                **common,
            )
            pl.enable_depth_peeling(10)
        else:  # volume
            ds = self.data_getter_dict[self.modality].get_vol_data_fn(self.voxel_res)
            grid = pv.RectilinearGrid(ds["x"].values, ds["y"].values, ds["z"].values)

            grid.point_data["point_count"] = (
                ds["point_count"].transpose("x", "y", "z").values.ravel(order="F")
            )
            grid.point_data["occupancy"] = (
                ds["occupancy"]
                .transpose("x", "y", "z")
                .values.ravel(order="F")
                .astype(np.uint8)
            )

            pl.add_volume(
                grid,
                scalars="point_count",
                cmap=cmap,
                opacity=[0, 0.35, 0.55, 0.75, 0.95],
                **common,
            )

        pl.add_text(
            f"Terrain: {LABELS[self.mode.lower()]}",
            position="upper_left",
            font_size=10,
            name="mode_title",
        )
