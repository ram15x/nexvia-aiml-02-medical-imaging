# Medical Imaging AI Review Platform — presentation and evidence

The 10-slide deck is `Medical_Imaging_AI_Review_Platform.pptx`. It presents the portal scope, architecture, classifier comparisons, segmentation results, limitations, and prompts for the student’s own reflection.

Metric sources and caveats are linked from the project README and `docs/model_card.md`. Classifier panels were reused; neither is an independent test. The 15.5k manifest link is **NOT VERIFIED**. Segmentation checkpoint selection used the six validation cases; the internal holdout had prior project evaluation history.

Synthetic Orthanc DX upload, sync, classifier inference, and JSONL result storage passed. A synthetic CT and one public de-identified CT series passed the CT segmentation workflow; the real-series check used an independently supplied matching count and had no reference mask. Broad CT compatibility and accuracy remain unverified. Docker uses CPU-only PyTorch. The latest full staged suite passed 69 tests without warnings in a disposable CPU container, not the pinned Windows/CUDA environment.

Before recording, correct or explain these older deck statements: slide 3 calls the 8k comparator deployed; it is the earlier reference, and the API uses the 8k/10k weight soup. Slides 4 and 9 say the Orthanc CT path is unverified; it has since passed the limited checks described above. Slide 9 also states an older test result and collection failure; the latest suite result is 69 passed. The deck file itself has not been updated for these later checks. See `../reports/requirements_and_verification.md` for evidence and limits.

The deck is a ten-minute presentation guide, not a substitute for practice. Ram will record and present it after Project 2, with a short walkthrough of the local demo; he will write his own reflection. No recording is included yet.
