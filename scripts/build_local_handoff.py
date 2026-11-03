"""Refresh the local AIML-02 handoff with current aggregate evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "deliverables" / "AIML02_Project1_Local_Handoff_2026-11-01_v9.zip"
OUTPUT = ROOT / "deliverables" / "AIML02_Project1_Local_Handoff_2026-11-02_v14.zip"

ADD = {
    "deliverables/AIML02_Project1_Presentation_2026-11-02_v15.pptx",
    "reports/classifier_improvement_decision_2026-11-01.md",
    "reports/ensemble_8k10k_5k_2026-11-01/README.md",
    "reports/ensemble_8k10k_5k_2026-11-01/metrics.json",
    "reports/ensemble_8k10k_5k_2026-11-01/per_label_metrics.csv",
    "reports/ensemble_8k10k_5k_2026-11-01/per_label_metrics.png",
    "reports/ensemble_8k10k_99_patient_check_2026-11-01/README.md",
    "reports/ensemble_8k10k_99_patient_check_2026-11-01/metrics.json",
    "reports/ensemble_8k10k_99_patient_check_2026-11-01/per_label_metrics.csv",
    "reports/external_validation_plan_chexpert_2026-10-31.md",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/README.md",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/metrics.json",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/per_label_metrics.csv",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/weight_soup_metrics.json",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/weight_soup_per_label_metrics.csv",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/weight_soup_per_label_average_precision.png",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/weight_soup_99_patient_check_2026-11-01/metrics_99.json",
    "reports/frozen_candidates_2250_internal_holdout_2026-11-01/weight_soup_99_patient_check_2026-11-01/per_label_metrics_99.csv",
    "reports/handoff_runtime_smoke_2026-11-01.md",
    "reports/masked_negative_supervision_screen_2026-11-01.md",
    "reports/no_finding_patient_filter_screen_2026-11-01.md",
    "reports/subgroup_performance_2026-11-01/README.md",
    "reports/subgroup_performance_2026-11-01/summary.json",
    "reports/subgroup_performance_2026-11-01/subgroup_summary.csv",
    "reports/subgroup_performance_2026-11-01/subgroup_differences.csv",
    "reports/subgroup_performance_2026-11-01/subgroup_per_label.csv",
    "reports/subgroup_performance_2026-11-01/subgroup_metrics.svg",
    "scripts/audit_subgroup_performance.py",
    "scripts/prepare_no_finding_patient_filter_screen.py",
    "scripts/evaluate_label_mask_policy_validation.py",
    "scripts/evaluate_frozen_candidates_internal_holdout.py",
    "train_classifier.py",
    "test_classifier.py",
    "test_subgroup_audit.py",
}

DROP = {
    "deliverables/AIML02_Project1_Presentation_2026-10-31_v11.pptx",
    "deliverables/AIML02_Project1_Presentation_2026-11-01_v12.pptx",
    "deliverables/AIML02_Project1_Presentation_2026-11-02_v13.pptx",
    "deliverables/AIML02_Project1_Presentation_2026-11-02_v14.pptx",
}
PRIVATE_PARTS = {".git", ".venv", ".venv-dvc", "__pycache__", "node_modules"}
RUNTIME_PREFIXES = ("api/", "web/", "artifacts/")
RUNTIME_FILES = {
    "docker-compose.yml",
    "requirements-api.lock.txt",
    "requirements-torch-cpu.lock.txt",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not (set(path.parts) & PRIVATE_PARTS)
        and name not in {".env", "FILE_MANIFEST.json"}
        and not name.startswith(("data/NexviaProjectData/", "data/external/"))
    )


def is_runtime_file(name: str) -> bool:
    return name in RUNTIME_FILES or name.startswith(RUNTIME_PREFIXES)


def main() -> None:
    if not BASE.is_file():
        raise FileNotFoundError(f"Base runtime-verified package is missing: {BASE}")

    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(BASE) as archive:
        manifest = json.loads(archive.read("FILE_MANIFEST.json"))
        for item in manifest:
            name = item["path"]
            data = archive.read(name)
            if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
                raise ValueError(f"Base archive manifest mismatch: {name}")
            if name not in DROP:
                contents[name] = data

    for name in list(contents):
        source = ROOT.joinpath(*PurePosixPath(name).parts)
        if source.is_file():
            updated = source.read_bytes()
            if is_runtime_file(name) and updated != contents[name]:
                raise ValueError(
                    f"Runtime file changed since v6 verification; refresh the runtime "
                    f"evidence before packaging: {name}"
                )
            contents[name] = updated
        else:
            contents.pop(name)

    for name in ADD:
        source = ROOT.joinpath(*PurePosixPath(name).parts)
        if not source.is_file():
            raise FileNotFoundError(f"Required handoff file is missing: {source}")
        contents[name] = source.read_bytes()

    source_readme = ROOT / "HANDOFF_README.md"
    contents["HANDOFF_README.md"] = source_readme.read_bytes()
    if not all(safe_path(name) for name in contents):
        raise ValueError("Handoff includes a private or unsafe path")

    file_manifest = [
        {"path": name, "bytes": len(data), "sha256": sha256(data)}
        for name, data in sorted(contents.items())
    ]
    contents["FILE_MANIFEST.json"] = (
        json.dumps(file_manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=8) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr(name, data)

    with zipfile.ZipFile(OUTPUT) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC check failed: {bad_member}")
        saved_manifest = json.loads(archive.read("FILE_MANIFEST.json"))
        if len(saved_manifest) != len(archive.namelist()) - 1:
            raise ValueError("Handoff manifest count does not match archive contents")
        for item in saved_manifest:
            data = archive.read(item["path"])
            if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
                raise ValueError(f"Output archive manifest mismatch: {item['path']}")

    print(f"Created {OUTPUT}")
    print(f"Files: {len(file_manifest)}; archive bytes: {OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
