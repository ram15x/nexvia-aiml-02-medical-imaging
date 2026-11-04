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

- The release root has no data/ directory. dvc.yaml refers to missing paths under data/NexviaProjectData/nih/ and a machine-local Windows virtual environment. Its dvc.lock records one historical DenseNet-121 subset screen: 3 epochs, seed 42, 7,271 image files (2,883,330,459 bytes; MD5 directory hash 0dcaa0f03adcdbfabff6bb71ae5d9045.dir) and a 250,145-byte manifest (MD5 2a4419aae47585a6005d333da618dc4c). These are recorded hashes/sizes in the lock, not files verified in this release.
- In the private working copy, .dvc/config contains a site-cache setting but no DVC remote; a small local .dvc/cache remains. Neither is included in the public package. The release lacks the image/volume data and manifests required by the recorded stage, and DVC is not installed in the current verification environment. The lock output checkpoint is recorded as 28,486,934 bytes, while the packaged classifier checkpoint is 28,496,134 bytes. The lock MD5 for api/modeling.py (af0db3ac8fd85d7c26834d3bff8ea38c) differs from the release file (d9e22d3f646e3081d2d1ffba0b8e5bb0); train_classifier.py (d7eb6f22d84698926da1b6d926dbdd89) differs from (b6244808b7abfa5b670fd3e5338fd881). requirements-training.txt matches at 6354f1ced67503cc35e5f6e577d61207. The 3D segmentation workflow is outside this DVC stage.
- `.gitignore` excludes `.env`, bulk data, DVC cache, and most `.pt` files. Keep credentials out of GitHub. The `.env` file is local configuration and is not part of `submission_final`.
- The public GitHub repository is `https://github.com/ram15x/nexvia-aiml-02-medical-imaging`. It contains a scrubbed source snapshot, the two deployed model files, one current presentation deck, and selected aggregate audit reports. Raw datasets, sample inputs, per-patient outputs, environment files, historical checkpoints, and archived experiments are excluded.

## Reproduction boundary

The packaged checkpoints support inference on user-supplied inputs; example inputs are not included. Training and full data-audit reproduction require restoring the dataset files and manifests at the paths expected by the scripts, fixing or replacing its machine-specific command, and recreating the recorded environment. The current DVC lock does not reproduce the shipped checkpoints. Dataset terms and company sharing expectations must be checked before distributing raw data or derived patient-level files.

## Evidence

- `README.md` — historical NIH inventory and cleanup notes.
- `reports/project_closeout.md` — model cohorts, segmentation splits, checkpoint selection, and evaluation caveats.
- `reports/final_checks.md` — six-case validation and internal-holdout scores/history.
- `dvc.yaml` and `.dvc/config` — pipeline inputs and absence of a configured remote.

