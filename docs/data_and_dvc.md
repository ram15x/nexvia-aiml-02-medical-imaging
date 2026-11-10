# Data and DVC

## Sources

| Dataset | Use | Format | Where it lives |
|---|---|---|---|
| NIH ChestX-ray14, official test split sample (999 images, 549 patients) | Classifier evaluation | PNG | `data/nih_test_sample_1000/`. Images tracked by DVC. Manifest and protocol in git. |
| MSD Task09 Spleen | Spleen model training and validation | NIfTI | Not in this repository. Obtain from the MSD release. |
| Public de-identified CT series (TCIA) | Orthanc interoperability | DICOM | Not in this repository. |

`data/nih_test_sample_1000/PROTOCOL.md` records how the sample was drawn: uniform random, seed 42, from the official test archive, with one truncated file removed.

## Versioning

- **Data:** `data/nih_test_sample_1000/images.dvc` pins the 999 images by content hash. The images themselves are stored in the DVC remote, not git.
- **Pipeline:** `dvc.yaml` defines the stages. `dvc.lock` pins every input and output hash.
- **Checkpoints:** the two served checkpoints are committed to git in `artifacts/`, so the application runs from a clone without DVC.
- **Demo examples:** `demo/*.dvc` pin the two demo inputs.

## Remote

The DVC remote is named `storage`. In this project it is a folder, `../dvc-storage` relative to the repository. The remote folder is not part of the GitHub repository. Without it, `dvc pull` cannot fetch the images; download the NIH ChestX-ray14 images listed in `data/nih_test_sample_1000/selected_1000.txt` from the NIH release instead, or copy that folder from the developer's machine or from the submission folder, then run:

```bash
dvc pull
dvc repro
```

## Stages

| Stage | Command | Output |
|---|---|---|
| `build_eval_manifest` | `tools/build_eval_manifest.py` | `results/nih_sample/eval_manifest.csv` |
| `evaluate_nih_sample` | `evaluate_bootstrap.py` with the served checkpoint | `per_image_predictions.csv`, `per_label_intervals.json` |
| `subgroup_bias` | `tools/subgroup_bias.py` | `subgroup_metrics.csv` |
Not a DVC stage: run `python tools/make_results_doc.py` to regenerate `docs/results.md`, so the page stays tracked by git.

Evaluation is deterministic (seed 42). Re-running `dvc repro` reproduces the same numbers on the same inputs.

## Known gaps

- The classifier training stage that produced the served checkpoint is not in `dvc.yaml`. Its training data (full NIH train split, validation panel) is not in this repository.
- The spleen training and holdout stages are not in `dvc.yaml`, because the MSD split used for the served model is not recorded here.
