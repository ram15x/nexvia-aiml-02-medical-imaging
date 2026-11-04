# Data and versioning guide

The platform uses two public research datasets for separate tasks. The application checkpoints and code are organized independently from the source image collections.

## Datasets and roles

| Dataset | Workflow | Data organization |
|---|---|---|
| NIH ChestX-ray14 | Multi-label chest X-ray classification | Fourteen findings are derived from radiology reports. Model development used patient-level training cohorts and development panels so a patient's images stay grouped during comparisons. |
| Medical Segmentation Decathlon Task09 Spleen | 3D spleen CT segmentation | The recorded labeled development cohort contains 41 volumes: 29 training, 6 validation, and 6 internal holdout. The official challenge test set is unlabeled. |

The classifier panels contain 5,000 and 2,138 patients and support 13 and 9 labels, respectively, at the ≥30-positive reporting threshold. They were reused during model comparison. The selected segmentation checkpoint was chosen using the six validation cases. These are development cohorts and should be described with those roles.

## DVC and package organization

DVC pointer files identify the retained chest X-ray and spleen CT demonstration examples. The package also contains the two deployed model checkpoints under `artifacts/`. Dataset examples, training cohorts, and application code are handled as distinct assets; source dataset terms should be followed when accessing or sharing the original collections.

The root `dvc.yaml`, `dvc.lock`, and `params.yaml` record a historical NIH subset-screen stage and its parameters. The deployed checkpoints represent the model workflows summarized in the README and model card. DVC metadata is used for data lineage and version references, while model inference uses the packaged checkpoints.

## Data handling

- Keep `.env`, local DVC cache, and patient-level material out of public source control.
- Keep dataset download and access aligned with NIH and MSD terms.
- Use patient-level identifiers for splitting chest radiographs and case-level identifiers for CT volumes.
- Use the included demonstration examples for the application walkthrough; do not upload identifiable patient data.

See [the model card](model_card.md) for evaluation cohorts and [the architecture guide](architecture.md) for data flow.