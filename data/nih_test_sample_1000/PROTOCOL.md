# NIH ChestX-ray14 — 1,000-image official test sample

- **Source:** NIH Clinical Center CXR8 release (https://nihcc.app.box.com/v/ChestXray-NIHCC), archive `images/images_012.tar.gz`
- **Archive size:** 2,914,187,733 bytes
- **Archive SHA-256:** 7316ce5f4d5e0154730e592cc2b45b48a2bee8457f6339d36c2ca55ed6e60b26
- **Labels and patient metadata:** `Data_Entry_2017_v2020.csv` (NIH)
- **Split reference:** `test_list.txt` (NIH official test split, 25,596 images)
- **Selection:** the 3,592 official-test images contained in this archive were sampled uniformly at random with seed 42, n = 1,000, covering 549 patients.
- **Files:** `images/` (999 PNGs, all fully decoded), `manifest.csv`, `selected_1000.txt` (drawn list of 1,000; one exclusion below).
- **Exclusion:** `00029820_000.png` was drawn but is truncated inside the NIH archive (fails full PNG decode, both on extraction and in a fresh copy). It was removed. The archive was deleted, so no replacement was drawn; the evaluation is therefore 999 images.

This is a sample of the official test split from a single archive. It is not a full official-test evaluation. Labels are report-mined and noisy. Images are for educational evaluation only.

Attribution: Wang X. et al., "ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization," CVPR 2017. Please cite the NIH Clinical Center when using these images.
