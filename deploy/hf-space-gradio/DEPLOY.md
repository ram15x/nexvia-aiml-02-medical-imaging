# Hosted Gradio Space

The public demo (https://huggingface.co/spaces/Ram15x/medical-imaging-ai-review) runs on Hugging Face ZeroGPU.

A Space needs these files at its root:

| In the Space | Source in this repository |
|---|---|
| `app.py`, `requirements.txt`, `README.md` | `deploy/hf-space-gradio/` |
| `api/` (all `.py` files, `api/__init__.py`) | `api/` |
| `segment_spleen.py`, `train_segmentation.py` | repository root |
| `artifacts/best_densenet121.pt`, `artifacts/best_spleen_unet_20261109.pt` | `artifacts/` |

`app.py` imports the scoring and segmentation helpers from `api/app.py`, so the hosted demo and the local API run the same model code. Orthanc is not used on the Space.
