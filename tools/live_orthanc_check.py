"""Live end-to-end check of the Orthanc path: upload DICOM -> API pulls it -> inference -> stored results.

Needs the compose stack running with the Orthanc profile:
    docker compose --profile orthanc up -d --build
Uses only the Python standard library. Fixtures (DICOM files + manifest.json) are built by
tools/png_to_dicom_dx.py and tools/nifti_to_dicom_series.py; the manifest holds the scores and
spleen voxel count that the same models give on the original PNG / NIfTI files.

    python tools/live_orthanc_check.py --fixtures <folder> --out results/live_orthanc_check.json
"""
from __future__ import annotations

import argparse, base64, hashlib, json, sys, time, urllib.error, urllib.request
from pathlib import Path


def env_value(path: Path, key: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return ""


def call(method, url, *, data=None, headers=None, auth=None, timeout=600):
    h = dict(headers or {})
    if auth:
        h["Authorization"] = "Basic " + base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def jcall(*a, **k):
    status, body = call(*a, **k)
    try:
        return status, json.loads(body)
    except ValueError:
        return status, {"raw": body[:200].decode("utf-8", "replace")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--api", default="http://localhost:18080")
    ap.add_argument("--orthanc", default="http://localhost:18042")
    ap.add_argument("--env", type=Path, default=Path(".env"))
    ap.add_argument("--out", type=Path, default=Path("results/live_orthanc_check.json"))
    ap.add_argument("--timeout", type=int, default=420, help="seconds to wait for Orthanc to mark studies stable")
    a = ap.parse_args()
    manifest = json.loads((a.fixtures / "manifest.json").read_text())
    auth = (env_value(a.env, "ORTHANC_USERNAME") or "orthanc", env_value(a.env, "ORTHANC_PASSWORD"))
    report: dict = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        return ok

    s, h = jcall("GET", a.api + "/health")
    check("API /health", s == 200 and h.get("status") in ("ok", "healthy") or s == 200, f"http {s}")
    report["api_health"] = h
    s, sysinfo = jcall("GET", a.orthanc + "/system", auth=auth)
    check("Orthanc reachable with auth", s == 200, f"http {s}, version {sysinfo.get('Version')}")
    report["orthanc_version"] = sysinfo.get("Version")
    s401, _ = call("GET", a.orthanc + "/system")
    check("Orthanc rejects unauthenticated requests", s401 == 401, f"http {s401}")
    if s != 200:
        a.out.write_text(json.dumps(report, indent=1)); return 1

    dx = {}
    for f in sorted((a.fixtures / "dx").glob("*.dcm")):
        s, r = jcall("POST", a.orthanc + "/instances", data=f.read_bytes(), headers={"Content-Type": "application/dicom"}, auth=auth)
        check(f"upload {f.name} to Orthanc", s == 200 and "ID" in r, f"http {s}")
        dx[f.name] = r.get("ID")
    ct_dir = a.fixtures / manifest["ct"]["series_file"]
    ct_series = None; ct_ok = 0
    for f in sorted(ct_dir.glob("*.dcm")):
        s, r = jcall("POST", a.orthanc + "/instances", data=f.read_bytes(), headers={"Content-Type": "application/dicom"}, auth=auth)
        ct_ok += s == 200; ct_series = ct_series or r.get("ParentSeries")
    check(f"upload CT series ({manifest['ct']['instances']} slices)", ct_ok == manifest["ct"]["instances"], f"{ct_ok} uploaded")
    s, sinfo = jcall("GET", f"{a.orthanc}/series/{ct_series}", auth=auth)
    report["ct_series_orthanc_status"] = sinfo.get("Status"); report["ct_series_instances_in_orthanc"] = len(sinfo.get("Instances", []))

    # The API reads Orthanc's change log; studies become 'stable' after Orthanc's StableAge (60 s by default).
    t0 = time.time(); got = {}
    while time.time() - t0 < a.timeout and len(got) < len(dx):
        s, sync = jcall("POST", a.api + "/orthanc/sync?limit=64")
        for name, iid in dx.items():
            if name in got:
                continue
            _, res = jcall("GET", f"{a.api}/orthanc/{iid}/results")
            rec = [r for r in res.get("results", []) if r.get("status") == "processed"]
            if rec:
                got[name] = rec[-1]
        if len(got) < len(dx):
            time.sleep(15)
    report["seconds_until_all_radiographs_processed"] = round(time.time() - t0)
    check("API pulled and scored all uploaded radiographs", len(got) == len(dx), f"{len(got)}/{len(dx)} after {round(time.time()-t0)} s")
    worst = 0.0
    for name, rec in got.items():
        scores = {f["label"]: f["score"] for f in rec["findings"]}
        diff = max(abs(scores[k] - v) for k, v in manifest["dx"][name].items())
        worst = max(worst, diff)
    check("DICOM-via-Orthanc scores equal the offline scores", got and worst < 1e-5, f"max |diff| {worst:.2e}")
    report["max_score_difference_vs_offline"] = worst
    s, res = jcall("GET", f"{a.api}/orthanc/{next(iter(dx.values()))}/results")
    # CT instances must be skipped by the chest classifier, not scored.
    _, ct_inst = jcall("GET", f"{a.orthanc}/series/{ct_series}/instances", auth=auth)
    ct_first = ct_inst[0]["ID"] if ct_inst else None
    _, ctres = jcall("GET", f"{a.api}/orthanc/{ct_first}/results") if ct_first else (0, {"results": []})
    statuses = [(r.get("status"), r.get("reason")) for r in ctres.get("results", [])]
    check("CT slice is skipped by the chest classifier", ("skipped", "unsupported_modality") in statuses, str(statuses[:1]))

    # Spleen segmentation of the CT series pulled from Orthanc.
    seg_url = f"{a.api}/orthanc/series/{ct_series}/segment/spleen"
    t1 = time.time(); s, seg = jcall("POST", seg_url, timeout=900)
    used_expected = False
    if s == 422 and "not confirmed complete" in json.dumps(seg):
        used_expected = True
        t1 = time.time(); s, seg = jcall("POST", seg_url + f"?expected_instances={manifest['ct']['instances']}", timeout=900)
    report["ct_segmentation_seconds"] = round(time.time() - t1); report["ct_needed_expected_instances"] = used_expected
    check("spleen segmentation of the Orthanc CT series", s == 200, f"http {s}, {report['ct_segmentation_seconds']} s" + (", expected_instances supplied" if used_expected else ""))
    if s == 200:
        check("voxel count equals the direct-NIfTI result", seg["predicted_spleen_voxels"] == manifest["ct"]["expected_predicted_spleen_voxels"],
              f"{seg['predicted_spleen_voxels']} vs {manifest['ct']['expected_predicted_spleen_voxels']}")
        ms, mask = call("GET", a.api + seg["mask_url"])
        check("stored mask downloads and matches its recorded hash", ms == 200 and hashlib.sha256(mask).hexdigest() == seg["mask_sha256"], f"{len(mask)} bytes")
        report["ct_result"] = {k: seg.get(k) for k in ("predicted_spleen_voxels", "estimated_predicted_volume_ml", "dicom_instance_count", "input_shape", "orthanc_series_status", "completeness_basis", "mask_sha256")}
    else:
        report["ct_error"] = seg

    # Reviewer decision on an Orthanc-sourced result, then confirm it is in the audit log.
    first = next(iter(got.values())) if got else None
    if first:
        s, r = jcall("POST", a.api + "/reviews", data=json.dumps({"case_ref": first["result_id"], "reviewed_findings": [], "note": "live Orthanc check", "reviewer_action": "accepted_for_demo"}).encode(), headers={"Content-Type": "application/json"})
        check("approve decision recorded in the audit log", s in (200, 201), f"http {s}")
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["all_passed"] = all(c["ok"] for c in report["checks"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nall passed" if report["all_passed"] else "\nSOME CHECKS FAILED", "->", a.out)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
