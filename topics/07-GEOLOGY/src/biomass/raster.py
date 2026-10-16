"""Quick inspection of Biomass GeoTIFFs (local or remote)."""
from .auth import TokenProvider


def inspect_raster(path_or_url, auth: TokenProvider = None) -> dict:
    """Print bands, dtype, CRS, GCPs and metadata; return the rasterio meta dict.

    For an https URL the file is read remotely (needs ``auth``) without downloading it.
    """
    import rasterio as rio

    if str(path_or_url).startswith("http"):
        import fsspec
        auth = auth or TokenProvider()
        opener = fsspec.filesystem("https", headers=auth.headers()).open(path_or_url, "rb")
    else:
        opener = open(path_or_url, "rb")
    with opener as f, rio.open(f) as src:
        print(f"Size: {src.width} x {src.height} px | bands: {src.count} | dtype: {src.dtypes[0]}")
        print(f"CRS: {src.crs} | GCPs: {len(src.get_gcps()[0])}")
        for i in range(src.count):
            print(f"  band {i + 1}: {src.descriptions[i] or '(no description)'}  tags={src.tags(i + 1)}")
        print("Metadata:", src.tags())
        return src.meta
