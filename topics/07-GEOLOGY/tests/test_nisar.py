import base64
import json
import time
from types import SimpleNamespace

import pytest

from biomass import AuthError, Downloader, EarthdataToken, QueryError, SearchQuery, nisar_items_table
from biomass.nisar import NisarClient, _name_patterns, _named_assets
from conftest import FakeResponse, FakeSession

ID = "NISAR_L2_PR_GCOV_004_025_D_083_4005_DHDH_A_20251030T224554_20251030T224629_P05023_N_F_J_001"
HOST = "https://nisar.asf.earthdatacloud.nasa.gov"


def jwt(exp):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_earthdata_token_is_sent_as_bearer(tmp_path):
    f = tmp_path / "nasa_token.txt"
    f.write_text(jwt(time.time() + 3600) + "\n")
    assert EarthdataToken(f).headers() == {"Authorization": f"Bearer {f.read_text().strip()}"}


def test_earthdata_token_is_found_in_parent_folder(tmp_path, monkeypatch):
    (tmp_path / "nasa_token.txt").write_text("abc")
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")
    monkeypatch.delenv("NASA_TOKEN_FILE", raising=False)
    assert EarthdataToken().access_token() == "abc"


def test_expired_earthdata_token_raises(tmp_path):
    f = tmp_path / "nasa_token.txt"
    f.write_text(jwt(time.time() - 60))
    with pytest.raises(AuthError, match="expired"):
        EarthdataToken(f).access_token()


def test_missing_or_empty_earthdata_token_raises(tmp_path):
    with pytest.raises(AuthError):
        EarthdataToken(tmp_path / "nope.txt")
    f = tmp_path / "nasa_token.txt"
    f.write_text(" \n")
    with pytest.raises(AuthError):
        EarthdataToken(f).access_token()


def test_assets_get_short_names_and_s3_copies_are_dropped():
    hrefs = [f"{HOST}/BROWSE/C/{ID}/{ID}_NATIVE_B_HVHV.png", f"{HOST}/BROWSE/C/{ID}/{ID}_LATLON.png",
             f"s3://bucket/C/{ID}/{ID}_LATLON.png", f"{HOST}/BROWSE/C/{ID}/{ID}_LATLON_thumbnail.png",
             f"{HOST}/BROWSE/C/{ID}/{ID}_NATIVE.png", f"{HOST}/NISAR/C/{ID}/{ID}.h5", f"s3://bucket/C/{ID}/{ID}.h5",
             "https://cmr.earthdata.nasa.gov/search/concepts/G1-ASF.xml"]
    named = _named_assets(ID, {href: SimpleNamespace(href=href) for href in hrefs})
    assert {key: asset.href.rsplit("/", 1)[-1][len(ID):] for key, asset in named.items()} == {
        "browse_b_hvhv": "_NATIVE_B_HVHV.png", "browse_latlon": "_LATLON.png",
        "thumbnail": "_LATLON_thumbnail.png", "browse": "_NATIVE.png", "product": ".h5"}


def test_items_table_reads_the_product_name():
    item = SimpleNamespace(id=ID, properties={"start_datetime": "2025-10-30T22:45:54.000Z"}, assets={"product": 1})
    row = nisar_items_table([item]).iloc[0]
    assert (row["orbit"], row["track"], row["frame"], row["mode"], row["pols"]) == (
        "descending", "025", "083", "4005", "DHDH")


def test_search_rejects_product_type(tmp_path):
    f = tmp_path / "nasa_token.txt"
    f.write_text("abc")
    with pytest.raises(QueryError):
        NisarClient(EarthdataToken(f)).search(SearchQuery("NISAR_L2_GCOV_PROVISIONAL_V1_1", "GCOV"))


def test_name_patterns_for_mode_filters():
    assert _name_patterns("dh", 40) == ["*_40??_DH??_*"]
    assert _name_patterns(["DH", "QP"], None) == ["*_????_DH??_*", "*_????_QP??_*"]
    assert _name_patterns(None, 5) == ["*_05??_????_*"]


@pytest.mark.parametrize("polarisation, bandwidth", [("HH", None), (None, 80)])
def test_unknown_mode_filters_raise(polarisation, bandwidth):
    with pytest.raises(QueryError):
        _name_patterns(polarisation, bandwidth)


def test_download_without_item_folders(tmp_path, fake_auth):
    item = SimpleNamespace(id=ID, assets={"product": SimpleNamespace(href=f"{HOST}/NISAR/C/{ID}/{ID}.h5")})
    dl = Downloader(fake_auth, tmp_path, session=FakeSession([FakeResponse(body=b"h5")]), item_folders=False)
    assert dl.download([item], ["product"]) == [tmp_path / f"{ID}.h5"]


def test_read_gcov_returns_db_and_georeference(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from biomass import inspect_gcov, read_gcov

    path = tmp_path / "gcov.h5"
    with h5py.File(path, "w") as f:
        f["science/LSAR/identification/granuleId"] = ID.encode()
        for freq, spacing, n in (("A", 10.0, 8), ("B", 80.0, 4)):
            grid = f.create_group(f"science/LSAR/GCOV/grids/frequency{freq}")
            grid["listOfCovarianceTerms"] = [b"HHHH", b"HVHV", b"HHHV"]
            grid["HHHH"] = np.full((n, n), 100.0, dtype="float32")
            grid["HVHV"] = np.array([[np.nan] + [10.0] * (n - 1)] * n, dtype="float32")
            grid["HHHV"] = np.zeros((n, n), dtype="complex64")
            grid["xCoordinates"] = 500000 + spacing * (np.arange(n) + 0.5)
            grid["yCoordinates"] = 1000000 - spacing * (np.arange(n) + 0.5)
            grid["xCoordinateSpacing"], grid["yCoordinateSpacing"] = spacing, -spacing
            grid["projection"] = np.uint32(32620)

    assert inspect_gcov(path)["frequencies"]["A"] == {
        "terms": ["HHHH", "HVHV", "HHHV"], "shape": (8, 8), "epsg": 32620, "spacing_m": 10.0}
    db, names, transform, crs = read_gcov(path)  # coarsest frequency by default
    assert (names, db.shape, crs) == (["HHHH", "HVHV"], (2, 4, 4), "EPSG:32620")
    assert db[0, 0, 0] == pytest.approx(20.0) and db[1, 0, 1] == pytest.approx(10.0) and np.isnan(db[1, 0, 0])
    assert transform * (0, 0) == (500000, 1000000) and transform.a == 80
    db, _, transform, _ = read_gcov(path, frequency="A", max_size=4)
    assert db.shape == (2, 4, 4) and transform.a == 20
