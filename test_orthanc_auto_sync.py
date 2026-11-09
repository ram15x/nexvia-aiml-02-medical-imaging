"""The automatic Orthanc pull runs the sync on an interval and stops on shutdown."""

import time

import api.app as app_module


def test_auto_sync_loop_calls_sync_until_stopped(monkeypatch):
    import threading

    calls = []

    def fake_sync(limit=16):
        calls.append(limit)
        return {"ok": True}

    monkeypatch.setattr(app_module, "sync_orthanc_studies", fake_sync)
    stop = threading.Event()
    worker = threading.Thread(target=app_module._auto_sync_loop, args=(stop, 0.02), daemon=True)
    worker.start()
    deadline = time.time() + 2
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(calls) >= 3
    assert set(calls) == {16}


def test_lifespan_starts_worker_only_when_enabled(monkeypatch):
    from fastapi.testclient import TestClient

    started = []
    monkeypatch.setattr(app_module, "_auto_sync_loop", lambda stop, interval: started.append(interval))

    monkeypatch.setattr(app_module, "ORTHANC_AUTO_SYNC_SECONDS", 0)
    with TestClient(app_module.app):
        time.sleep(0.05)
    assert started == []

    monkeypatch.setattr(app_module, "ORTHANC_AUTO_SYNC_SECONDS", 7)
    with TestClient(app_module.app):
        time.sleep(0.2)
    assert started == [7]
