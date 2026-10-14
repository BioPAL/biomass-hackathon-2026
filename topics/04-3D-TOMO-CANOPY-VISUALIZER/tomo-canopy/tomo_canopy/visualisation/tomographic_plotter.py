from tomo_canopy.types import Modality, DataLayer
import numpy as np
import pyvista as pv

from tomo_canopy.visualisation.basic_plotter_app import BasicPlotterApp
from tomo_canopy.types.data_interface import DataCube

MODES = ("surface", "solid", "volume", "bubbles")
LABELS = {"surface": "Surface", "solid": "Solid block", "volume": "Volume", "bubbles": "Bubbles"}


DATA_LAYERS = ["TLS", "ALS", "P-BAND", "GEDI"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def get_demo_data(n=200, seed=42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two hills plus gentle ripples and noise, as a demo elevation grid."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    xx, yy = np.meshgrid(x, y)
    z = (
        2.0 * np.exp(-((xx - 1) ** 2 + (yy + 1) ** 2) / 4)
        + 1.5 * np.exp(-((xx + 2) ** 2 + (yy - 2) ** 2) / 2)
        + 0.3 * np.sin(xx * 1.5) * np.cos(yy * 1.5)
        + rng.normal(0, 0.02, xx.shape)
    )
    return xx * 100, yy * 100, z * 100  # scale to "meters"


# --------------------------------------------------------------------------
# Geometry builders
# --------------------------------------------------------------------------
def build_surface(data_cube: DataCube) -> pv.StructuredGrid:
    """Turn x/y/z grids into a PyVista StructuredGrid with an 'elevation' array."""
    xx, yy, z = data_cube.als.get_xyz_arrays()
    z_geom = np.where(np.isnan(z), np.nanmin(z), z)  # fill nodata holes for geometry
    grid = pv.UnstructuredGrid(xx, yy, z_geom)
    grid["elevation"] = z_geom.ravel()  # must match PyVista's point order
    return grid


def build_volume(xx, yy, z, nz=60, max_xy=150):
    """Fill the space under the terrain surface with voxels.

    Returns an ImageData grid with point arrays:
        solid      1.0 where the voxel is below the surface, else 0.0
        elevation  voxel height where solid, NaN elsewhere
    plus the (base, top) z range of the block.
    """
    x, y = xx[0, :], yy[:, 0]
    z = np.where(np.isnan(z), np.nanmin(z), z)

    # Downsample so the voxel count stays manageable
    step = max(1, max(z.shape) // max_xy)
    x, y, z = x[::step], y[::step], z[::step, ::step]

    if y[0] > y[-1]:  # GeoTIFF rows run north to south; make y ascending
        y, z = y[::-1], z[::-1, :]

    ny, nx = z.shape
    zmin, zmax = float(z.min()), float(z.max())
    base = zmin - 0.25 * (
        zmax - zmin
    )  # give the block some thickness below the lowest point
    zs = np.linspace(base, zmax, nz)

    image_cls = getattr(pv, "ImageData", None) or pv.UniformGrid
    grid = image_cls(
        dimensions=(nx, ny, nz),
        spacing=(x[1] - x[0], y[1] - y[0], zs[1] - zs[0]),
        origin=(x[0], y[0], base),
    )

    zlev = zs[:, None, None]  # shape (nz, 1, 1) broadcast against terrain (ny, nx)
    inside = (
        zlev <= z[None, :, :]
    )  # shape (nz, ny, nx); x varies fastest, as VTK expects
    grid["solid"] = inside.astype(float).ravel()
    grid["elevation"] = np.where(inside, zlev, np.nan).ravel()

    # Volume rendering needs a scalar with no NaNs. Empty voxels get a fill value
    # below the block, 1/3 of its height under the base, so the base sits at 25%
    # of the scalar range and empty space maps to zero opacity.
    fill = base - (zmax - base) / 3
    grid["density"] = np.nan_to_num(grid["elevation"], nan=fill).astype(np.float32)
    return grid, (base, zmax), fill


# --------------------------------------------------------------------------
# The interactive app
# --------------------------------------------------------------------------
class TomographicPlotter(BasicPlotterApp):
    def __init__(self, data_layer, x, y, z, mode="surface"):
        # def __init__(self, data: DataCube):
        self.mode = mode
        self.exaggeration = 1.0
        self.n_contours = 12
        self._first_render = True
        # self.cube = data
        # Build the geometry once; switching modes only rebuilds the view of it
        # self.surface = build_surface(xx, yy, z)
        # self.voxels, self.z_range, self.fill = build_volume(xx, yy, z, nz=60)
        self.x = x
        self.y = y
        self.z = z
        self.data_layer = data_layer

        super().__init__()

    @classmethod
    def from_data_layer(cls, data_layer: DataLayer, mode="surface"):
        return TomographicPlotter(
            data_layer, data_layer.data.x, data_layer.data.y, data_layer.data.z, mode
        )

    def build_ui(self):
        for _, mode in enumerate(MODES):
            self.add_checkbox(
                mode, mode, self._make_callback(mode), value=(mode == self.mode)
            )
        self.increase_btn_y()
        for data_layer in DATA_LAYERS:
            self.add_checkbox(
                data_layer,
                data_layer,
                self._make_callback(data_layer),
                value=(data_layer == self.mode),
            )

        self.pl.add_slider_widget(
            self._on_exaggeration,
            rng=[0.5, 5.0],
            value=self.exaggeration,
            title="Vertical exaggeration",
            pointa=(0.60, 0.94),
            pointb=(0.97, 0.94),
            style="modern",
            fmt="%.1f",
            interaction_event="end",  # rebuild on release, not on every drag step
        )

    def _make_callback(self, mode):
        def callback(_state):
            # Radio behaviour: clicking always selects this mode. Clicking the
            # active button would untick it, so set_mode re-syncs the buttons.
            self.set_mode(mode)

        return callback

    def _on_exaggeration(self, value):
        self.exaggeration = float(value)
        self.render()

    def set_mode(self, m):
        self.mode = m
        for _, mode in enumerate(MODES):
            self.buttons[mode]["value"] = mode == self.mode
        self.render()

    def _scaled(self, mesh):
        if self.exaggeration == 1.0:
            return mesh
        return mesh.scale([1, 1, self.exaggeration], inplace=False)

    def on_render(self):
        pl = self.pl
        keep_camera = not self._first_render
        common = dict(name=self.MAIN, reset_camera=not keep_camera)
        bar = {"title": "TODO"}
        if self.mode == "surface":
            # grid = self._scaled(build_surface(self.cube))
            x, y, z = (
                self.x,
                self.y,
                self.z,
            )

            # print(f"x: {x.shape}, y: {y.shape}, z: {z.shape}")
            # if z shape is 2D
            # if z.ndim == 2:

            points = np.column_stack((x, y, z))  # shape (N, 3)
            cloud = pv.PolyData(points)
            cloud["elevation"] = z
            pl.add_mesh(
                cloud,
                scalars="elevation",
                cmap="viridis",
                point_size=8,
                render_points_as_spheres=True,
            )

        elif self.mode == "solid":
            solid = self._scaled(self.voxels).threshold(
                0.5, scalars="solid", all_scalars=True
            )
            opts = dict(
                scalars="elevation",
                cmap="terrain",
                clim=self.z_range,
                scalar_bar_args=bar,
                **common,
            )
            pl.add_mesh_clip_plane(solid, normal="y", **opts)

        elif self.mode == "bubbles":
            ds = self.data_layer.data
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
            cloud["scale"] = 0.5 * voxel_size * np.sqrt(
                values / values.max()
            )

            bubbles = cloud.glyph(
                orient=False,
                scale="scale",
                factor=2.0,  # multiplies 'scale'; tune per dataset
                geom=pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=12),
            )

            pl.add_mesh(
                bubbles,
                scalars="point_count",
                cmap="viridis",
                scalar_bar_args=bar,
                **common,
            )

        else:  # volume
            # Build the grid from the 1D coordinates
            ds = self.data_layer.data
            grid = pv.RectilinearGrid(ds["x"].values, ds["y"].values, ds["z"].values)

            # Transpose (z, y, x) -> (x, y, z), then flatten in Fortran order
            grid.point_data["point_count"] = (
                ds["point_count"].transpose("x", "y", "z").values.ravel(order="F")
            )
            grid.point_data["occupancy"] = (
                ds["occupancy"]
                .transpose("x", "y", "z")
                .values.ravel(order="F")
                .astype(np.uint8)
            )

            # Volume rendering of point counts
            pl.add_volume(
                grid,
                scalars="point_count",
                cmap="viridis",
                opacity=[0, 0.35, 0.55, 0.75, 0.95],
                scalar_bar_args=bar,
                **common,
            )

            # volume = pv.DataSet()
            # pl.add_volume(
            #     grid,
            #     scalars="density",
            #     cmap="terrain",
            #     # clim=(self.fill, top),
            #     # A short list of opacity nodes is spread evenly across clim and
            #     # works in every renderer; long per-bin arrays rendered blank.
            #     # opacity=[0, 0.35, 0.55, 0.75, 0.95],
            #     # scalar_bar_args=bar,
            #     **common,
            # )

        pl.add_text(
            f"Terrain: {LABELS[self.mode].lower()}",
            position="upper_left",
            font_size=10,
            name="mode_title",
        )


def main():

    demo_data_cube = create_sample_data_cube()
    TomographicPlotter(demo_data_cube).run()


if __name__ == "__main__":
    main()
