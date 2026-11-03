# Medical Imaging AI Review Platform — presentation and evidence

The 10-slide deck is `Medical_Imaging_AI_Review_Platform.pptx`. It presents the portal scope, architecture, classifier comparisons, segmentation results, limitations, and prompts for the student’s own reflection.

Metric sources and caveats are linked from the project README and `docs/model_card.md`. Classifier panels were reused; neither is an independent test. The 15.5k manifest link is **NOT VERIFIED**. Segmentation checkpoint selection used the six validation cases; the internal holdout had prior project evaluation history.

Synthetic Orthanc DX upload, sync, classifier inference, and JSONL result storage passed. Orthanc CT-series segmentation remains unverified. Docker is configured for CPU-only PyTorch. The 56-test suite passed in the local Docker test image, not the pinned Windows/CUDA environment.

The deck is a ten-minute presentation guide, not a substitute for practice. I still need to record the demo and write my own reflection accurately.
