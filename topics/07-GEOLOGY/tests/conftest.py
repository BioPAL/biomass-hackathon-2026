import pytest


class FakeResponse:
    def __init__(self, status_code=200, body=b"", json_data=None, headers=None):
        self.status_code = status_code
        self._body = body
        self._json = json_data
        self.headers = headers or {"content-length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Returns queued responses and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, method, url, **kw):
        self.calls.append((method, url, kw))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def get(self, url, **kw):
        return self._next("GET", url, **kw)

    def post(self, url, **kw):
        return self._next("POST", url, **kw)


class FakeAuth:
    def headers(self):
        return {"Authorization": "Bearer test"}


@pytest.fixture
def fake_auth():
    return FakeAuth()
