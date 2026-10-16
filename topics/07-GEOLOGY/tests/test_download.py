from types import SimpleNamespace

import pytest

from biomass import Downloader, DownloadError
from conftest import FakeResponse, FakeSession

URL = "https://example.org/data/file.tiff"


def make(tmp_path, responses, auth, **kw):
    session = FakeSession(responses)
    return Downloader(auth, tmp_path, session=session, backoff=0, **kw), session


def test_downloads_file(tmp_path, fake_auth):
    dl, _ = make(tmp_path, [FakeResponse(body=b"hello world")], fake_auth)
    dest = dl.download_file(URL, tmp_path / "a" / "file.tiff")
    assert dest.read_bytes() == b"hello world"
    assert not (tmp_path / "a" / "file.tiff.part").exists()


def test_existing_file_is_skipped(tmp_path, fake_auth):
    dest = tmp_path / "file.tiff"
    dest.write_bytes(b"done")
    dl, session = make(tmp_path, [], fake_auth)
    dl.download_file(URL, dest)
    assert session.calls == []


def test_partial_file_is_resumed_with_range(tmp_path, fake_auth):
    dest = tmp_path / "file.tiff"
    (tmp_path / "file.tiff.part").write_bytes(b"hello ")
    dl, session = make(tmp_path, [FakeResponse(206, b"world")], fake_auth)
    dl.download_file(URL, dest)
    assert session.calls[0][2]["headers"]["Range"] == "bytes=6-"
    assert dest.read_bytes() == b"hello world"


def test_resume_restarts_if_server_ignores_range(tmp_path, fake_auth):
    dest = tmp_path / "file.tiff"
    (tmp_path / "file.tiff.part").write_bytes(b"garbage")
    dl, _ = make(tmp_path, [FakeResponse(200, b"hello world")], fake_auth)
    dl.download_file(URL, dest)
    assert dest.read_bytes() == b"hello world"


def test_retries_then_succeeds(tmp_path, fake_auth):
    dl, session = make(tmp_path, [ConnectionError("boom"), FakeResponse(body=b"ok")], fake_auth)
    dl.download_file(URL, tmp_path / "f.bin")
    assert len(session.calls) == 2


def test_raises_after_all_retries(tmp_path, fake_auth):
    dl, _ = make(tmp_path, [ConnectionError("x")] * 3, fake_auth, retries=3)
    with pytest.raises(DownloadError):
        dl.download_file(URL, tmp_path / "f.bin")


def test_download_items_uses_item_folder_and_skips_missing_assets(tmp_path, fake_auth):
    item = SimpleNamespace(id="ITEM1", assets={"quicklook_jpg": SimpleNamespace(href=URL.replace("tiff", "jpg"))})
    dl, _ = make(tmp_path, [FakeResponse(body=b"img")], fake_auth)
    files = dl.download([item], ["quicklook_jpg", "missing"])
    assert files == [tmp_path / "ITEM1" / "file.jpg"]
    assert files[0].read_bytes() == b"img"
