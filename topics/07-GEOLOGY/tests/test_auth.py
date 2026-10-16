import time

import pytest

from biomass import AuthError, TokenProvider
from conftest import FakeResponse, FakeSession


def ok(token="abc", expires_in=300):
    return FakeResponse(json_data={"access_token": token, "expires_in": expires_in})


def test_token_is_cached_until_expiry(tmp_path):
    f = tmp_path / "token.txt"
    f.write_text("offline\n")
    session = FakeSession([ok("t1")])
    auth = TokenProvider(f, session=session)
    assert auth.access_token() == "t1"
    assert auth.access_token() == "t1"
    assert len(session.calls) == 1


def test_token_is_refreshed_when_expired(tmp_path):
    f = tmp_path / "token.txt"
    f.write_text("offline")
    session = FakeSession([ok("t1", expires_in=1), ok("t2")])
    auth = TokenProvider(f, session=session)
    assert auth.access_token() == "t1"
    auth._expires_at = time.time() - 1
    assert auth.access_token() == "t2"


def test_request_uses_refresh_token_grant(tmp_path):
    f = tmp_path / "token.txt"
    f.write_text("line1\nline2")
    session = FakeSession([ok()])
    TokenProvider(f, session=session).access_token()
    data = session.calls[0][2]["data"]
    assert data["grant_type"] == "refresh_token"
    assert data["refresh_token"] == "line1line2"


def test_headers_have_bearer(tmp_path):
    f = tmp_path / "token.txt"
    f.write_text("x")
    auth = TokenProvider(f, session=FakeSession([ok("zzz")]))
    assert auth.headers() == {"Authorization": "Bearer zzz"}


def test_missing_token_file_raises(tmp_path):
    with pytest.raises(AuthError):
        TokenProvider(tmp_path / "nope.txt")


def test_rejected_token_raises_auth_error(tmp_path):
    f = tmp_path / "token.txt"
    f.write_text("expired")
    auth = TokenProvider(f, session=FakeSession([FakeResponse(status_code=400)]))
    with pytest.raises(AuthError):
        auth.access_token()


def test_empty_token_file_raises(tmp_path):
    f = tmp_path / "token.txt"
    f.write_text("  \n")
    with pytest.raises(AuthError):
        TokenProvider(f, session=FakeSession([])).access_token()
