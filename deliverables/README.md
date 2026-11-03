# AIML-02 Project 1 deliverables

## Presentation

`AIML02_Project1_Presentation_2026-11-03_v21.pptx` updates the segmentation holdout wording, classifier comparator wording, and Nov 3 demo status. v20 remains in the private working archive and is excluded from the compact submission and public GitHub release. The student's reflection slide remains a prompt and should be completed in the student's own words.

## Evidence and interpretation

The detailed values, source paths, commands, caveats, and failures are in [Project 1 closeout](../reports/project1_closeout_2026-11-03.md) and [final inference checks](../reports/project1_final_checks_2026-11-03.md). Classifier panels were reused; neither is an independent test set. The 15.5k manifest link is **NOT VERIFIED**. Segmentation checkpoint selection used the six validation cases, while the six internal holdout cases had been scored previously. This is an educational, non-diagnostic prototype.

## Demo boundary

Live HTTP, browser upload, fresh Windows installation, and Docker API/web checks passed Nov 3. Orthanc remains **NOT VERIFIED**; Docker segmentation is configured for CPU and runtime device was not logged. See the [demo audit](../reports/Nexvia_AIML02_Demo_Segmentation_Audit_2026-11-03.md) and final-day closeout.

## Environment snapshot

[`requirements-windows-tested.txt`](../requirements-windows-tested.txt) is a freeze of the existing Windows `aiml02-venv310` environment. A later fresh Windows install was verified on Nov 3.



