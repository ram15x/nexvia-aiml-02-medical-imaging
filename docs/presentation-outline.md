# Ten-minute presentation outline

Use `deliverables/AIML02_Project1_Presentation_2026-11-02_v15.pptx`. The speaker-note timings below add to exactly 10 minutes (600 seconds). Rehearse with the demo ready; the actual talk has not yet been timed.

1. **Project purpose — 20 sec.** Introduce AIML-02 as an educational review prototype with separate chest X-ray classification and spleen CT segmentation. State that it is not for diagnosis or care.
2. **Question and scope — 45 sec.** Explain the two outcomes and that they use separate models, datasets, and pipelines.
3. **Data and evaluation boundary — 60 sec.** The current app uses 8k DenseNet-121, selected on macro AP using an additional patient-disjoint 5,000-patient NIH development cohort; the 10k candidate scored higher AUROC. This is same-source development evidence, not external validation. The earlier 500-image/58-patient official-test result belongs to the former 10,761-image checkpoint and is retired for model selection. The learning curve used one seed per size and a previously inspected 1,000-patient panel. MSD Spleen has 41 labeled cases.
4. **System architecture — 60 sec.** Walk through the React/FastAPI review UI, DenseNet-121 chest classifier, and separate MONAI 3D U-Net spleen segmenter. Orthanc sync handles stable single-frame CR/DX. A separate manual CT-series route exists. The Nov 2 synthetic upload reached stable status, but `Status: Unknown` blocked conversion before inference; this route remains unverified end to end.
5. **Classifier learning curve and loss screen — 100 sec.** Show the nested 800–10,000-patient results. AUROC/AP/F1@0.5 at 800 were 0.635/0.081/0.177; at 10,000 they were 0.746/0.139/0.218. Gains flattened past roughly 6,000. This is one seed per size on a reused development panel. The matched 2,500-patient Asymmetric Loss screen did not establish a reliable improvement; weighted BCE remains the reference.
6. **Current classifier comparison — 55 sec.** At fixed 0.5, the previous demo scored macro AUROC/AP and micro F1 of 0.745/0.130/0.226; 8k scored 0.779/0.143/0.248; 10k scored 0.786/0.140/0.243. 8k was selected on macro AP; 10k had higher AUROC. All missed 34 pneumonia positives. Do not call these accuracy or independent clinical results.
7. **Spleen CT segmentation — 90 sec.** Describe the 3D preprocessing, MONAI U-Net, geometry-preserving mask output, and separate inference route. The selected flip-averaged mean Dice was 0.3412 on the same six internal cases; it remains weak and is not independent validation. The 20 official challenge cases have no labels.
8. **Local review and Orthanc flow — 65 sec.** Demonstrate uncalibrated chest scores, coarse Grad-CAM, and demo-only review logging. Show that CT uses a separate NIfTI or manual Orthanc Series ID route. Software checks passed, including 44 Python tests, the Vite production build, and Compose configuration; these do not establish model quality. A Nov 2 synthetic CT upload reached stability, but `Status: Unknown` stopped conversion; no live CT inference ran.
9. **Limits and remaining work — 65 sec.** State that no untouched final test exists for the current classifier and no external validation was run. The 8k/10k comparison is a documented metric tradeoff, and both model paths remain weak. Before a public release, clean local Git history and decide how to distribute the chest checkpoint. The talk still needs timing, the reflection must be yours, and AIML-03 plus the one-time combined Drive submission remain open.
10. **Your reflection — 40 sec.** Describe only work you personally did, one decision changed by evidence, what failed, and what you would change. Use `docs/reflection-prompts.md`; write this in your own words.

## Precise language

- Say macro AUROC, macro average precision, micro F1 at fixed threshold 0.5, and mean case Dice. Do not call these accuracy.
- Keep results tied to their checkpoint and cohort; the former model's official-test subset does not validate the current checkpoint.
- Identify the Orthanc DICOM shown in the demo as synthetic.
- Keep the educational-use disclaimer visible. Never imply diagnosis, triage, or treatment suitability.
