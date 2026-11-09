"""Does keeping only the largest connected component help? Choose on validation, then report test once.

Run: python tools/spleen_postprocess_study.py --root <folder with imagesTr/labelsTr> --split results/spleen/case_split.json \
     --checkpoint artifacts/best_spleen_unet_20261109.pt --output results/spleen/postprocess_study.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_segmentation import build_transform, make_model, seed_everything


def dice(pred, ref):
    d = int(pred.sum()) + int(ref.sum())
    return 1.0 if d == 0 else 2.0 * int(np.logical_and(pred, ref).sum()) / d


def largest_component(mask):
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    seed_everything(42)
    split = json.loads(a.split.read_text())
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    patch = tuple(ck["patch_size"])
    model = make_model(ck["architecture"]); model.load_state_dict(ck["state_dict"], strict=True); model.eval()
    out = {}
    for part in ("validation", "test"):
        rows = []
        cases = [{"image": str(a.root / "imagesTr" / c), "label": str(a.root / "labelsTr" / c), "case_id": c} for c in split[part]]
        loader = DataLoader(Dataset(cases, transform=build_transform(False, patch)), batch_size=1)
        with torch.inference_mode():
            for b in loader:
                p = torch.softmax(sliding_window_inference(b["image"], patch, 1, model, overlap=0.25), dim=1)[0, 1].numpy()
                ref = (b["label"][0, 0] == 1).numpy()
                raw = p >= 0.5
                rows.append({"case_id": b["case_id"][0], "raw": dice(raw, ref), "largest_component": dice(largest_component(raw), ref),
                             "components": int(ndimage.label(raw)[1])})
        out[part] = {"cases": rows, "mean_raw": float(np.mean([r["raw"] for r in rows])),
                     "mean_largest_component": float(np.mean([r["largest_component"] for r in rows]))}
        print(part, round(out[part]["mean_raw"], 4), round(out[part]["mean_largest_component"], 4), flush=True)
    a.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
