from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.guards import SlidingWindowLimiter, install_guards


def _app(monkeypatch, **env):
    for name in ("NEXVIA_DISABLE_ORTHANC", "NEXVIA_RATE_LIMIT_PER_MINUTE",
                 "NEXVIA_QA_LIMIT_PER_HOUR", "NEXVIA_TRUST_PROXY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    app = FastAPI()

    @app.post("/predict")
    def predict():
        return {"ok": True}

    @app.post("/qa")
    def qa():
        return {"ok": True}

    @app.post("/orthanc/sync")
    def sync():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    install_guards(app)
    return TestClient(app)


def test_limiter_blocks_after_limit_and_recovers():
    now = [0.0]
    limiter = SlidingWindowLimiter(2, 60.0, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 61.0
    assert limiter.allow("a")


def test_limiter_with_zero_limit_is_disabled():
    limiter = SlidingWindowLimiter(0, 60.0)
    assert all(limiter.allow("a") for _ in range(100))


def test_guards_are_off_by_default(monkeypatch):
    client = _app(monkeypatch)
    assert client.post("/orthanc/sync").status_code == 200
    assert all(client.post("/predict").status_code == 200 for _ in range(50))


def test_orthanc_routes_return_404_when_disabled(monkeypatch):
    client = _app(monkeypatch, NEXVIA_DISABLE_ORTHANC="true")
    response = client.post("/orthanc/sync")
    assert response.status_code == 404
    assert client.post("/predict").status_code == 200


def test_post_rate_limit_applies_per_client_and_spares_get(monkeypatch):
    client = _app(monkeypatch, NEXVIA_RATE_LIMIT_PER_MINUTE="3")
    assert [client.post("/predict").status_code for _ in range(4)] == [200, 200, 200, 429]
    assert client.post("/predict").headers["retry-after"] == "60"
    assert client.get("/health").status_code == 200


def test_qa_has_its_own_hourly_limit(monkeypatch):
    client = _app(monkeypatch, NEXVIA_QA_LIMIT_PER_HOUR="2")
    assert [client.post("/qa").status_code for _ in range(3)] == [200, 200, 429]
    assert client.post("/predict").status_code == 200


def test_proxy_header_is_used_only_when_trusted(monkeypatch):
    client = _app(monkeypatch, NEXVIA_RATE_LIMIT_PER_MINUTE="1", NEXVIA_TRUST_PROXY="true")
    assert client.post("/predict", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert client.post("/predict", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
    assert client.post("/predict", headers={"x-forwarded-for": "9.9.9.9, 1.1.1.1"}).status_code == 429
    untrusted = _app(monkeypatch, NEXVIA_RATE_LIMIT_PER_MINUTE="1")
    assert untrusted.post("/predict", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert untrusted.post("/predict", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 429


def test_orthanc_enabled_flag_follows_environment(monkeypatch):
    from api.guards import orthanc_enabled

    monkeypatch.delenv("NEXVIA_DISABLE_ORTHANC", raising=False)
    assert orthanc_enabled() is True
    monkeypatch.setenv("NEXVIA_DISABLE_ORTHANC", "true")
    assert orthanc_enabled() is False
