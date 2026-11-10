"""The sync cursor must recover when Orthanc's database is reset (its change counter restarts at 0)."""
import json

from fastapi.testclient import TestClient

import api.app as api_module
import api.orthanc as orthanc_module


def test_sync_restarts_from_zero_when_orthanc_was_reset(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"last_seq": 23, "pending_studies": ["old-study-id-1"]}))
    monkeypatch.setattr(api_module, "ORTHANC_SYNC_STATE_PATH", state)
    monkeypatch.setattr(api_module, "ORTHANC_RESULTS_PATH", tmp_path / "results.jsonl")
    asked = []
    monkeypatch.setattr(orthanc_module, "get_orthanc_last_sequence", lambda: 4)

    def changes(since, limit):
        asked.append(since)
        return {"Changes": [{"Seq": 3, "ChangeType": "StableStudy", "ID": "new-study-id-1"}], "Last": 4, "Done": True}

    monkeypatch.setattr(orthanc_module, "get_orthanc_changes", changes)
    monkeypatch.setattr(orthanc_module, "list_stable_study_instances", lambda study_id: [])
    body = TestClient(api_module.app).post("/orthanc/sync").json()
    assert asked == [0]
    assert body["stable_studies_queued"] == 1 and body["last_sequence"] == 4
    assert "old-study-id-1" not in json.dumps(json.loads(state.read_text()))


def test_sync_keeps_cursor_when_orthanc_is_ahead(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"last_seq": 23, "pending_studies": []}))
    monkeypatch.setattr(api_module, "ORTHANC_SYNC_STATE_PATH", state)
    monkeypatch.setattr(api_module, "ORTHANC_RESULTS_PATH", tmp_path / "results.jsonl")
    asked = []
    monkeypatch.setattr(orthanc_module, "get_orthanc_last_sequence", lambda: 40)
    monkeypatch.setattr(orthanc_module, "get_orthanc_changes", lambda since, limit: asked.append(since) or {"Changes": [], "Last": 40, "Done": True})
    TestClient(api_module.app).post("/orthanc/sync")
    assert asked == [23]
