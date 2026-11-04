# Presentation guide — Medical Imaging AI Review Platform

The 10-slide deck is `Medical_Imaging_AI_Review_Platform.pptx`. It presents the project from task scope to data, architecture, measured outputs, application use, and personal reflection.

## Slide sequence

1. Project overview
2. Two independent imaging workflows
3. Data and evaluation design
4. End-to-end system architecture
5. Deployed classifier and reference model
6. Classifier results on the 5,000-patient panel
7. Classifier results on the 2,138-patient panel
8. Spleen CT segmentation results
9. Application outputs and intended use
10. Personal reflection

For technical details, see the root [README](../README.md), [architecture guide](../docs/architecture.md), and [model card](../docs/model_card.md). Metrics use macro AUROC/AP and mean case Dice with the stated cohorts. Classifier panels were reused during development, and segmentation checkpoint selection used the six validation cases. Keep the project’s educational, non-diagnostic scope clear.