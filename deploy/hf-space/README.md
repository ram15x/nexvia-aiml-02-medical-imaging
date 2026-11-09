---
title: Medical Imaging AI Review
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Educational chest X-ray and spleen CT demo
---

# Medical Imaging AI Review

Educational demo only. Not for diagnosis or treatment. Do not upload patient data.

This hosted demo runs the same API and interface as the local Docker stack: chest X-ray scores, Grad-CAM and SHAP views, the review log, and the spleen CT segmentation. The question-answering panel needs an `OPENROUTER_API_KEY` Space secret; without it that panel says it is not configured. Orthanc is not connected in this Space.

Model scores are uncalibrated and experimental. The spleen model's held-out accuracy is not measured. See the [project repository](https://github.com/ram15x/nexvia-aiml-02-medical-imaging) for the evaluation, limits and deployment notes.
