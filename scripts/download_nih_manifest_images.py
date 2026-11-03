"""Download and resize images listed in one project manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from skimage.transform import resize


BASE = "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/"


def fetch(row: dict[str, str], output_dir: Path) -> dict[str, object]:
    name = row["Image Index"]
    destination = output_dir / name
    if destination.exists() and destination.stat().st_size:
        return {"Image Index": name, "status": "already_present", "bytes": destination.stat().st_size}

    error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(BASE + name, timeout=(15, 90))
            response.raise_for_status()
            payload = response.content
            source_hash = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as source:
                gray = np.asarray(source.convert("L"), dtype=np.uint8)
            if gray.ndim != 2 or min(gray.shape) < 224:
                raise ValueError(f"Unexpected source image shape {gray.shape}")
            height, width = gray.shape
            side = min(height, width)
            y0, x0 = height // 2 - side // 2, width // 2 - side // 2
            crop = gray[y0:y0 + side, x0:x0 + side][None, :, :]
            resized = resize(crop, (1, 224, 224), mode="constant", preserve_range=True)[0]
            image = Image.fromarray(np.clip(np.rint(resized), 0, 255).astype(np.uint8), mode="L")
            temporary = destination.with_suffix(destination.suffix + ".part")
            image.save(temporary, format="PNG", optimize=False)
            os.replace(temporary, destination)
            return {
                "Image Index": name,
                "status": "downloaded",
                "source_sha256": source_hash,
                "source_bytes": len(payload),
                "width": width,
                "height": height,
                "output_bytes": destination.stat().st_size,
            }
        except Exception as exc:  # retry transient network and decode errors
            error = exc
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"{name}: {type(error).__name__}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["Image Index"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Manifest contains duplicate image filenames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in args.output_dir.glob("*.png")}
    if existing and not args.resume:
        raise RuntimeError(f"Output already contains {len(existing)} images; use --resume")
    pending = [row for row in rows if row["Image Index"] not in existing]
    print(json.dumps({"manifest_images": len(rows), "present": len(existing), "to_fetch": len(pending),
                      "source": BASE, "output": str(args.output_dir)}), flush=True)

    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_rows = {pool.submit(fetch, row, args.output_dir): row["Image Index"] for row in pending}
        for completed, future in enumerate(as_completed(future_rows), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"Image Index": future_rows[future], "error": str(exc)})
            if completed % 500 == 0 or completed == len(future_rows):
                print(f"completed={completed}/{len(future_rows)} errors={len(errors)}", flush=True)

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Image Index", "status", "source_sha256", "source_bytes", "width", "height", "output_bytes"]
    with args.ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(item)
    error_path = args.ledger.with_name("download_errors.json")
    error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    missing = [name for name in names if not (args.output_dir / name).is_file()]
    if errors or missing:
        raise SystemExit(f"Download incomplete: {len(errors)} errors and {len(missing)} missing; rerun with --resume")
    summary = {
        "images": len(names),
        "all_files_present": True,
        "total_output_bytes": sum((args.output_dir / name).stat().st_size for name in names),
        "preprocessing": "raw image decoded, converted to grayscale, center-cropped, resized to 224x224 with scikit-image semantics, saved as PNG",
        "source": BASE,
        "raw_mirror_authentication": "The object mirror is author-hosted. Source-object hashes are recorded but are not independently authenticated against NIH for every image.",
    }
    args.ledger.with_name("download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
