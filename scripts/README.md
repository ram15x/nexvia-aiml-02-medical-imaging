# Development scripts

This directory holds one-off scripts from model screening, evaluation, plots, data preparation, and historical experiments. They need datasets and run artifacts that are not included in the compact submission. They are not part of the demo's inference path.

For the current source workflow, use the root-level entry points: `data_audit.py`, `train_classifier.py`, `train_segmentation.py`, `evaluate_classifier.py`, `evaluate_segmentation.py`, and `segment_spleen.py`. The API and browser demo live in `api/` and `web/`.

Historical scripts are retained for provenance. Running an old experiment requires checking its report, manifest, and split provenance first; the development panels were reused and are not an independent test set.
