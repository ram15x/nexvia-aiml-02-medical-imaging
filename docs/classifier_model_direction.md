# Classifier model direction

Reviewed October 27, 2026.

## Current state

The working chest X-ray classifier is a pretrained DenseNet-121 with a 14-label sigmoid head and square 224×224 input. Training uses a patient-disjoint shard-001 manifest and a square-root positive-class weighting in the binary cross-entropy loss. The selected epoch is 6 by validation macro AUROC (0.7480). A one-time held-out evaluation on the available shard subset (967 images from 62 patients) gave macro AUROC 0.6778 and micro F1 0.2456 at 0.5. This is weak, incomplete evidence; it is not an accuracy guarantee.

The app already has a Grad-CAM endpoint and UI. Grad-CAM is a coarse attribution map, not a faithful medical explanation. The project also already has a separate MONAI 3D U-Net for spleen CT segmentation. Its internal held-out mean Dice is 0.3272 over six cases, which is also weak and variable.

## Candidate direction

Keep DenseNet-121 as the reproducible baseline. Two candidates were screened on the available shard: a frozen DenseNet-121 + Swin-Tiny feature-fusion model, and standalone ImageNet-pretrained Swin-Tiny. The standalone Swin and a fresh DenseNet run used the same patient-disjoint 3,466-train / 566-validation rows, seed, preprocessing, loss, metric code, three epochs, batch size 4, and optimizer settings. DenseNet reached validation macro AUROC 0.7385 and micro F1 0.2151 at the fixed 0.5 threshold; Swin-Tiny reached macro AUROC 0.6553 and micro F1 0.0000 at its selected epoch. The three-epoch screen is preliminary, but it provides no reason to replace the existing DenseNet checkpoint. Details: `../reports/matched_swin_comparison_2026-10-28.md`.

The frozen fusion candidate scored validation macro AUROC 0.6966 against 0.7480 for the existing DenseNet run, under a different optimization protocol; it was not promoted. The available 967-image test subset has already been used once and must not be used for more model selection. The complete official test split is not local.

## Why not BERT or RAD-DINO for this evaluation

BERT is a text encoder. The current NIH rows contain image-level finding labels, not paired radiology reports, so adding BERT would add no valid image evidence. A vision-language model would only be justified if paired reports were acquired with proper access, provenance, patient-level splitting, and a report-text evaluation plan.

Microsoft's RAD-DINO is a plausible radiology-specialized vision transformer and can expose both image and patch embeddings. Its published model card says the released checkpoint was trained on 112,120 NIH-CXR images, the same dataset as this project. That creates possible pretraining overlap with this project's train, validation, and test images. It is therefore not a clean candidate for claiming independent NIH test performance unless image-level overlap is audited and addressed. The model card also limits it to research use and warns about population bias. We should not select it just because it is medical-specific.

## Explainability and accuracy plan

- Keep Grad-CAM for the CNN baseline and the CNN branch of the hybrid; label its output as coarse attribution. The heatmap is not a faithful explanation of the fused model as a whole.
- For a transformer, use attention/patch visualization as a qualitative aid; do not call attention a causal explanation. Compare maps against known annotations only if a suitable localization dataset is added.
- SHAP can be added later for selected predictions, but it will not make the classifier more accurate and may be expensive across 14 labels. It is secondary to clean data, patient-level splits, per-label AUROC/AUPRC, calibration, and error review.
- Report sensitivity and precision at thresholds chosen on validation only. Do not describe AUROC as accuracy or optimize from the already-used test subset.

## Sources

- Microsoft RAD-DINO model card (architecture, intended use, biases, training-data sizes, and checkpoint provenance): https://huggingface.co/microsoft/rad-dino
- Swin Transformer paper: https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html
- Current measured classifier results: ../reports/classifier_shard001_results.md
- Current segmentation results: ../reports/msd_segmentation_results.md
# Proposed next classifier experiment: DenseNet label-graph attention

Status: implemented for controlled validation; **not yet trained or promoted**.

## Why this experiment

The shard pilot's DenseNet-121 remains the strongest measured model so far. The late-fusion DenseNet-121 + Swin-Tiny candidate scored lower on the matched shard validation screen, so simply adding a second pretrained backbone is not supported by our evidence. The next candidate changes how the current visual encoder makes its 14 decisions instead of adding a large second encoder.

The candidate uses DenseNet-121 spatial feature maps, one learned query per finding, cross-attention from each query to image regions, and an optional pathology co-occurrence graph. Graph edges are estimated only from the current training split using smoothed positive pointwise mutual information. The first direct attention-head screen scored well below DenseNet, so a second variant preserves the global DenseNet head and adds the attention logits as a learned per-label residual correction. The current ablations are: (1) existing DenseNet head, (2) direct label-query head with and without the graph, and (3) global-head residual attention with and without the training-only graph.

Label co-occurrence modeling has prior work in chest X-ray classification, including CheXGCN. The project contribution is therefore the concrete small-footprint implementation and controlled ablation, not a claim that graph-based classification is novel. CheXNet is a useful DenseNet reference, but its reported results use a different evaluation setup and must not be treated as a target or directly comparable benchmark.

## Evaluation gate

- Use the existing patient-disjoint train/validation manifest. The shard-001 967-image held-out subset has already been inspected once, so it stays out of all subsequent model choices.
- First run a short, fixed-seed engineering comparison on shard 001 with identical preprocessing, patient split, epoch count, and optimizer budget for the baseline and candidate.
- If the candidate is viable, rerun the comparison on the full training and validation cohorts after reacquiring the remaining NIH images. Use at least three seeds for the final comparison and report per-label AUROC/AUPRC, macro summaries, calibration, parameter count, throughput, and bootstrap confidence intervals.
- Select checkpoints and thresholds using validation only. Keep the official test set sealed for one final comparison after the architecture and training recipe are frozen.
- Record failure cases and uncertainty. Do not claim diagnostic accuracy or clinical readiness from report-mined labels.

## Preliminary shard-001 screen

The attention-only head was screened on the fixed 3,466-train / 566-validation manifest for three epochs with seed 42, batch size 4, ImageNet initialization, AdamW at 0.0001, and no test rows. Best validation macro AUROC was 0.663 without the graph and 0.680 with it; micro-F1 at fixed 0.5 was 0.089 and 0.039 respectively at their selected AUROC epochs. Matched DenseNet under the same budget scored 0.739 macro AUROC, 0.140 macro AUPRC, and 0.215 micro-F1 at 0.5. The direct attention-only head is therefore not promoted.

The graph-off global-head residual reached 0.750 macro AUROC at epoch 1, but its epoch-1 macro AUPRC (0.134) and micro-F1 at 0.5 (0.162) were below the matched DenseNet values. Later validation epochs had higher fixed-threshold F1 and AUPRC but lower AUROC. The graph-on residual had almost identical selected-epoch AUROC (0.7502 vs 0.7501), lower macro AUPRC (0.127 vs 0.134), and higher micro-F1 at 0.5 (0.182 vs 0.162). This is a mixed signal and does not justify promotion. Neither residual model has used the held-out test subset. See `reports/label_attention_ablation_2026-10-28.md` for full per-epoch metrics and artifact hashes.

## Data and attribution

The NIH dataset's Google-hosted alternative is explicitly Requester Pays, so it must not be used without checking billing first. The current plan uses the NIH Clinical Center Box distribution and stores archives, metadata, manifests, and checkpoints on D:. Cite the dataset's CVPR 2017 paper and acknowledge the NIH Clinical Center as provider in the final package.

## Research references

- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning* (2017): https://arxiv.org/abs/1711.05225
- Chen et al., *Label Co-Occurrence Learning With Graph Convolutional Networks for Multi-Label Chest X-Ray Image Classification* (2020): https://doi.org/10.1109/JBHI.2020.2967084
- Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases* (CVPR 2017): https://doi.org/10.1109/CVPR.2017.369
- NIH dataset access and attribution instructions: https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
