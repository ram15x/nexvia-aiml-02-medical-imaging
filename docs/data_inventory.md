# Data inventory and reproducibility — Medical Imaging AI Review Platform

This inventory describes files present in the project folder and the saved audit records. The full original datasets are not included.

## Selected data sources and roles

| Source | Project role | Recorded use | Current local/package status |
|---|---|---|---|
| NIH ChestX-ray14 | 14-label chest X-ray classification | Report-mined findings. Classifier development used training cohorts recorded as 8k and 10k patients and reused 5,000- and 2,138-patient development panels. An earlier reduced subset used 4,000 train, 1,779 validation, and 1,492 held-out images; these are separate historical cohorts and must not be conflated. | Full image data and training manifests are absent from this project copy. One small chest PNG example remains only in the private working folder under `demo/sample_inputs/`; it is excluded from the public release. |
| Medical Segmentation Decathlon Task09 Spleen | 3D spleen CT segmentation | Audit records 41 labeled volumes split into 29 training, 6 validation, and 6 project-internal holdout volumes. The selected checkpoint used the six validation cases for epoch selection. The holdout has prior project evaluation history; five cases appeared in training in separate nested-CV folds. | Full NIfTI volumes and case manifests are absent from this project copy. One small NIfTI example remains only in the private working folder under `demo/sample_inputs/`; it is excluded from the public release. |

The Task09 official challenge test set contains 20 unlabeled volumes in the source dataset; the reports state those images were not evaluated. The development results are not external or clinical validation.

## Exact files present

- Demo example files exist in the private working folder only; they are not part of the submission bundle or public release.
- The compact package and public release include the deployed classifier and segmentation checkpoints, but exclude demo inputs and full training datasets.


## DVC and local versus GitHub

- The project root has no `data/` directory. `dvc.yaml` refers to missing local paths under `data/NexviaProjectData/nih/` and a machine-local Python environment.
- In the private working copy, `.dvc/config` contains a site-cache setting but no DVC remote; a small local `.dvc/cache` remains. Neither the cache nor DVC config is included in the public package. They do not supply the missing image/volume datasets or training manifests.
- `.gitignore` excludes `.env`, bulk data, DVC cache, and most `.pt` files. Keep credentials out of GitHub. The `.env` file is local configuration and is not part of `submission_final`.
- The public GitHub repository is `https://github.com/ram15x/nexvia-aiml-02-medical-imaging`. It contains a scrubbed source snapshot, the two deployed model files, one current presentation deck, and selected aggregate audit reports. Raw datasets, sample inputs, per-patient outputs, environment files, historical checkpoints, and archived experiments are excluded.

## Reproduction boundary

The packaged checkpoints support inference on user-supplied inputs; example inputs are not included. Training and full data-audit reproduction require restoring the dataset files and manifests at the paths expected by the scripts, plus the recorded environment. Dataset terms and company sharing expectations must be checked before distributing raw data or derived patient-level files.

## Evidence

- `README.md` — historical NIH inventory and cleanup notes.
- `reports/project_closeout.md` — model cohorts, segmentation splits, checkpoint selection, and evaluation caveats.
- `reports/final_checks.md` — six-case validation and internal-holdout scores/history.
- `dvc.yaml` and `.dvc/config` — pipeline inputs and absence of a configured remote.

