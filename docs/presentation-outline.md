# Presentation guide

The 10-slide deck is `deliverables/Medical_Imaging_AI_Review_Platform.pptx`. This outline follows the project story from the task and data through architecture, measured outputs, and the intended use. Adjust timing to fit a ten-minute walkthrough.

1. **Project overview — 20 sec.** Introduce the Medical Imaging AI Review Platform and its two independent workflows.
2. **Scope — 45 sec.** Explain chest X-ray finding classification and 3D spleen CT segmentation.
3. **Data and evaluation — 60 sec.** Describe NIH report-mined labels, the two classifier development panels, and the Task09 spleen cohorts.
4. **Architecture — 60 sec.** Trace React uploads through FastAPI to DenseNet-121, Grad-CAM, MONAI 3D U-Net, and the optional Orthanc paths.
5. **Classifier design — 60 sec.** Explain the deployed 8k/10k DenseNet weight average and the 6c32 comparison model.
6. **Classifier results — 90 sec.** Present macro AUROC and AP, label support, and paired patient bootstrap intervals for the 5,000-patient panel.
7. **Second panel — 60 sec.** Present the corresponding 2,138-patient results and note that the panels were reused during development.
8. **Segmentation — 90 sec.** Describe the 29-volume training cohort, six validation cases, selected checkpoint, and Dice results.
9. **Application walkthrough — 75 sec.** Show image upload, classifier scores, Grad-CAM overlay, review action, NIfTI segmentation mask, and the Orthanc workflow.
10. **Reflection — 40 sec.** Share your own contribution, one evidence-led decision, a lesson from the work, and the next improvement you would pursue.

## Language for results

- Use **macro AUROC**, **macro average precision (AP)**, and **mean case Dice**; these are not accuracy percentages.
- Tie each result to its model and cohort.
- State that classifier panels were reused and that labels are report-mined.
- Describe Grad-CAM as a coarse attribution overlay.
- Keep the educational, non-diagnostic use statement visible.