"""Create a dependency-free SVG of per-label metrics from the frozen NIH sample."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports" / "classifier_unseen_test_sample_500_2026-10-30.json"
OUTPUT = ROOT / "reports" / "classifier_unseen_test_sample_per_label_2026-10-30.svg"


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result.get("split") != "official held-out test subset":
        raise ValueError("Expected an official-test subset result")
    if result.get("complete_test_split_evaluated") is not False:
        raise ValueError("Refusing to plot this sample as the complete official test set")
    if result.get("test_images") != 500 or result.get("test_patients") != 58:
        raise ValueError("Unexpected test sample; check the frozen result JSON")

    count = int(result["test_images"])
    rows = []
    for label, metrics in result["test_metrics"]["per_label"].items():
        if metrics.get("auroc") is None or metrics.get("auprc") is None:
            continue
        positives = int(metrics["positive_count"])
        rows.append((label.replace("_", " "), positives, float(metrics["auroc"]),
                     float(metrics["auprc"]), positives / count))
    rows.sort(key=lambda row: row[2])

    width = 1180
    row_height = 34
    top = 108
    height = top + len(rows) * row_height + 72
    left_x, left_width = 278, 390
    right_x, right_width, right_max = 805, 270, 0.5
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#24364b}.title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#52677f}.head{font-size:15px;font-weight:700}.lab{font-size:13px}.tick{font-size:11px;fill:#66788c}.foot{font-size:12px;fill:#52677f}</style>',
        '<text x="34" y="38" class="title">Frozen DenseNet-121 test sample</text>',
        f'<text x="34" y="62" class="sub">{count} NIH official-test images · {result["test_patients"]} patients · small subset, not the complete official test set</text>',
        f'<text x="{left_x}" y="92" class="head">AUROC — ranking</text>',
        f'<text x="{right_x}" y="92" class="head">AUPRC — retrieval</text>',
    ]
    for step in range(6):
        value = step / 5
        x = left_x + value * left_width
        pieces.append(f'<line x1="{x:.1f}" y1="101" x2="{x:.1f}" y2="{top + len(rows)*row_height - 3}" stroke="#dce3eb"/>')
        pieces.append(f'<text x="{x:.1f}" y="{top + len(rows)*row_height + 18}" class="tick" text-anchor="middle">{value:.1f}</text>')
    for step in range(6):
        value = step / 10
        x = right_x + value / right_max * right_width
        pieces.append(f'<line x1="{x:.1f}" y1="101" x2="{x:.1f}" y2="{top + len(rows)*row_height - 3}" stroke="#dce3eb"/>')
        pieces.append(f'<text x="{x:.1f}" y="{top + len(rows)*row_height + 18}" class="tick" text-anchor="middle">{value:.1f}</text>')

    for index, (label, positives, auc, ap, prevalence) in enumerate(rows):
        y = top + index * row_height
        pieces.append(f'<text x="260" y="{y+17}" class="lab" text-anchor="end">{html.escape(label)}  (n+={positives})</text>')
        pieces.append(f'<rect x="{left_x}" y="{y+4}" width="{max(1, auc*left_width):.1f}" height="18" rx="3" fill="#39799b"/>')
        pieces.append(f'<text x="{left_x + auc*left_width + 6:.1f}" y="{y+18}" class="tick">{auc:.2f}</text>')
        ap_w = ap / right_max * right_width
        pieces.append(f'<rect x="{right_x}" y="{y+4}" width="{max(1, ap_w):.1f}" height="18" rx="3" fill="#cf8b38"/>')
        px = right_x + prevalence / right_max * right_width
        pieces.append(f'<line x1="{px:.1f}" y1="{y+2}" x2="{px:.1f}" y2="{y+24}" stroke="#263647" stroke-width="2"/>')
        pieces.append(f'<text x="{right_x + ap_w + 6:.1f}" y="{y+18}" class="tick">{ap:.2f}</text>')

    footer_y = height - 22
    pieces.append(f'<text x="34" y="{footer_y}" class="foot">Dark tick marks positive prevalence (baseline for average precision). Fixed 0.5 threshold applies only to precision/recall/F1.</text>')
    pieces.append('</svg>')
    OUTPUT.write_text("\n".join(pieces), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
