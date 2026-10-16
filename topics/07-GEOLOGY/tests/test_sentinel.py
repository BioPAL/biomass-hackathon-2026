from types import SimpleNamespace

import pytest

from biomass import AuthError, CopernicusAuth, Downloader, SearchQuery, sentinel_items_table
from biomass.sentinel import _https_assets
from conftest import FakeResponse, FakeSession

NODE = "https://download.dataspace.copernicus.eu/odata/v1/Products(abc)/Nodes(X.SAFE)/Nodes(vv.tiff)/$value"


def token(access="a1", expires_in=600, refresh="r1", refresh_expires_in=3600):
    return FakeResponse(json_data={"access_token": access, "expires_in": expires_in,
                                   "refresh_token": refresh, "refresh_expires_in": refresh_expires_in})


def rejected():
    r = FakeResponse(status_code=400)
    r.ok = False
    return r


def test_copernicus_logs_in_with_password_and_caches_the_token():
    session = FakeSession([token("a1")])
    auth = CopernicusAuth("me@example.org", "secret", session=session)
    assert auth.headers() == {"Authorization": "Bearer a1"}
    assert auth.access_token() == "a1"
    assert len(session.calls) == 1
    assert session.calls[0][2]["data"] == {"client_id": "cdse-public", "grant_type": "password",
                                           "username": "me@example.org", "password": "secret"}


def test_copernicus_reads_the_credentials_file_from_a_parent_folder(tmp_path, monkeypatch):
    (tmp_path / "copernicus.txt").write_text("me@example.org\n\nsecret \n")
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")
    for var in ("COPERNICUS_USER", "COPERNICUS_PASSWORD", "COPERNICUS_CREDENTIALS_FILE"):
        monkeypatch.delenv(var, raising=False)
    session = FakeSession([token("a1")])
    CopernicusAuth(session=session).access_token()
    assert session.calls[0][2]["data"]["username"] == "me@example.org"
    assert session.calls[0][2]["data"]["password"] == "secret"


def test_copernicus_arguments_win_over_the_file_and_bad_files_raise(tmp_path, monkeypatch):
    f = tmp_path / "copernicus.txt"
    f.write_text("file@example.org\nfilepass\n")
    monkeypatch.delenv("COPERNICUS_USER", raising=False)
    monkeypatch.delenv("COPERNICUS_PASSWORD", raising=False)
    session = FakeSession([token("a1")])
    CopernicusAuth("arg@example.org", "argpass", session=session, credentials_file=f).access_token()
    assert session.calls[0][2]["data"]["username"] == "arg@example.org"
    f.write_text("only-email@example.org\n")
    with pytest.raises(AuthError):
        CopernicusAuth(credentials_file=f)
    with pytest.raises(AuthError):
        CopernicusAuth(credentials_file=tmp_path / "missing.txt")


def test_copernicus_renews_with_the_refresh_token():
    ok = token("a2")
    ok.ok = True
    session = FakeSession([token("a1"), ok])
    auth = CopernicusAuth("me@example.org", "secret", session=session)
    auth.access_token()
    auth._expires_at = 0
    assert auth.access_token() == "a2"
    assert session.calls[1][2]["data"] == {"client_id": "cdse-public", "grant_type": "refresh_token",
                                           "refresh_token": "r1"}


def test_copernicus_wrong_password_raises_and_is_not_retried(tmp_path):
    session = FakeSession([rejected()])
    dl = Downloader(CopernicusAuth("me@example.org", "wrong", session=session), tmp_path,
                    session=FakeSession([]), backoff=0)
    with pytest.raises(AuthError):
        dl.download_file(NODE, tmp_path / "vv.tiff")
    assert len(session.calls) == 1


def test_assets_use_their_https_address():
    assets = {
        "vv": SimpleNamespace(href="s3://eodata/X.SAFE/measurement/vv.tiff",
                              extra_fields={"alternate": {"https": {"href": NODE}}}),
        "thumbnail": SimpleNamespace(href="https://datahub.creodias.eu/odata/v1/Assets(1)/$value", extra_fields={}),
        "s3_only": SimpleNamespace(href="s3://eodata/X.SAFE/other", extra_fields={}),
    }
    kept = _https_assets(assets)
    assert {k: a.href for k, a in kept.items()} == {
        "vv": NODE, "thumbnail": "https://datahub.creodias.eu/odata/v1/Assets(1)/$value"}


def test_download_names_copernicus_files_from_their_local_path(tmp_path):
    asset = SimpleNamespace(href=NODE, extra_fields={"file:local_path": "X.SAFE/measurement/s1d-iw-grd-vv-001-cog.tiff"})
    item = SimpleNamespace(id="X", assets={"vv": asset})
    dl = Downloader(None, tmp_path, session=FakeSession([FakeResponse(body=b"tiff")]))
    assert dl.download([item], ["vv"]) == [tmp_path / "X" / "s1d-iw-grd-vv-001-cog.tiff"]


def test_public_file_is_downloaded_without_credentials(tmp_path):
    session = FakeSession([FakeResponse(body=b"png")])
    Downloader(None, tmp_path, session=session).download_file("https://example.org/t.png", tmp_path / "t.png")
    assert session.calls[0][2]["headers"] == {}


def test_cloud_cover_filter():
    q = SearchQuery("sentinel-2-l2a", "S2MSI2A", max_cloud_cover=10)
    assert q.cql_filter() == "product:type='S2MSI2A' AND eo:cloud_cover<=10"


def test_items_table_keeps_only_the_columns_that_apply():
    s1 = SimpleNamespace(id="S1", assets={}, properties={
        "start_datetime": "2026-07-01T23:31:06Z", "platform": "sentinel-1d", "sat:orbit_state": "ascending",
        "sat:relative_orbit": 121, "product:type": "IW_GRDH_1S", "sar:polarizations": ["VV", "VH"]})
    s2 = SimpleNamespace(id="S2", assets={}, properties={
        "datetime": "2026-07-01T15:26:49Z", "platform": "sentinel-2b", "product:type": "S2MSI2A", "eo:cloud_cover": 8.66})
    assert sentinel_items_table([s1]).iloc[0]["pols"] == "VV,VH" and "cloud" not in sentinel_items_table([s1])
    assert sentinel_items_table([s2]).iloc[0]["cloud"] == 8.66 and "pols" not in sentinel_items_table([s2])


def test_read_optical_scales_bands_to_reflectance(tmp_path):
    np = pytest.importorskip("numpy")
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from biomass import read_optical

    path = tmp_path / "T18PZS_B04_10m.tif"
    dn = np.full((8, 8), 3000, dtype="uint16")
    dn[0, :] = 0
    with rio.open(path, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint16",
                  crs="EPSG:32618", transform=from_origin(800000, 1200000, 10, 10)) as dst:
        dst.write(dn, 1)
    data, transform, crs = read_optical(path, max_size=8)
    assert data.shape == (1, 8, 8) and crs.to_epsg() == 32618 and transform.a == 10
    assert np.isnan(data[0, 0, 0]) and data[0, 1, 0] == pytest.approx(0.2)
    data, transform, _ = read_optical(path, max_size=4)
    assert data.shape == (1, 4, 4) and transform.a == 20
