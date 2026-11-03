# Ten-minute presentation guide

The slides are in `deliverables/Medical_Imaging_AI_Review_Platform.pptx`. Use these notes to explain the work in your own words; the listed times add to ten minutes. Rehearse with the local demo ready. Do not claim personal work or results you cannot explain.

1. **Purpose — 20 sec.** Introduce the portal assignment and its educational, non-diagnostic scope.
2. **Tasks — 45 sec.** Explain that chest X-ray classification and spleen CT segmentation are separate models and workflows.
3. **Data and evaluation — 60 sec.** Describe report-mined NIH labels, the 5,000- and 2,138-patient reused panels, and why neither is an independent test. Mention the small MSD cohort.
4. **Architecture — 60 sec.** Walk through React, FastAPI, DenseNet-121, Grad-CAM, MONAI U-Net, NIfTI masks, and the limited Orthanc DX demo path. State that the live Orthanc CT-series route is unverified.
5. **Classifier comparison — 90 sec.** Present the API weight average versus 6c32 metrics on the 5,000-patient and 2,138-patient panels. State the label counts and that both panels were reused.
6. **Interpretation — 60 sec.** Explain AUROC versus AP, why prevalence matters for AP, and why these results do not establish clinical usefulness.
7. **Segmentation — 90 sec.** Describe the 29-volume training set, six validation cases, selected checkpoint, mean Dice 0.8327, and why checkpoint selection makes that validation result optimistic. Describe prior use of the internal holdout cases.
8. **Demo and engineering — 75 sec.** Show upload, scores/overlay, local review event, mask, and the synthetic Orthanc DX sync/inference/result path. Do not imply the CT Orthanc route passed.
9. **Limitations and next work — 80 sec.** Discuss data availability, reused panels, limited cohorts, unverified CT Orthanc conversion, CPU Docker segmentation, and the absence of clinical validation.
10. **Personal reflection — 40 sec.** Explain what you personally did, what you found difficult, and what you would change next. Write this yourself; do not use a prewritten reflection.

## Terms to use carefully

- Say **macro AUROC**, **macro average precision**, and **mean case Dice**. Do not call these accuracy.
- Tie every score to its exact checkpoint and cohort.
- Call the Orthanc study synthetic.
- Say the classifier panels were reused; they are not independent test data.
- Keep the education-only notice visible.
