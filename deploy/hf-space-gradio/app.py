import base64, io, os, tempfile
from pathlib import Path
import gradio as gr
from PIL import Image
import spaces
from api.app import MAX_NIFTI_UPLOAD_BYTES, MAX_UPLOAD_BYTES, _decode_uploaded_image, _process_spleen_volume, _score_image

def classify(path):
    if not path:
        raise gr.Error("Select a chest X-ray image.")
    file = Path(path)
    if file.stat().st_size > MAX_UPLOAD_BYTES:
        raise gr.Error("Image exceeds the 12 MiB limit.")
    kind = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(file.suffix.lower())
    if kind is None:
        raise gr.Error("Upload a PNG or JPEG image.")
    try:
        image = _decode_uploaded_image(file.read_bytes(), kind)
        prediction = _score_image(image)
    except Exception as exc:
        raise gr.Error(f"Image could not be processed ({type(exc).__name__}).") from exc
    return [[item["label"], item["score"]] for item in prediction["findings"]]

@spaces.GPU(duration=60)
def segment(path):
    if not path:
        raise gr.Error("Select a 3D NIfTI CT volume.")
    source = Path(path)
    if not source.name.lower().endswith((".nii", ".nii.gz")):
        raise gr.Error("Upload a .nii or .nii.gz file.")
    if source.stat().st_size > MAX_NIFTI_UPLOAD_BYTES:
        raise gr.Error("The upload exceeds the 128 MiB limit.")
    try:
        with tempfile.TemporaryDirectory(prefix="medical-demo-") as tmp:
            result = _process_spleen_volume(source, Path(tmp) / "output")
            previews = []
            for plane in ("sagittal", "coronal", "axial"):
                image = Image.open(io.BytesIO(base64.b64decode(result["preview_planes"][plane]["png_base64"])))
                previews.append(image.copy())
            info = f"Predicted volume: {result['estimated_predicted_volume_ml']:.1f} mL. Accuracy for this input is unknown."
            return *previews, info
    except Exception as exc:
        raise gr.Error(f"Volume could not be processed ({type(exc).__name__}).") from exc

with gr.Blocks(title="Medical Imaging AI Review Platform") as demo:
    gr.Markdown("# Medical Imaging AI Review Platform\n**Educational demo only. Not for diagnosis or treatment. Do not upload patient data.**")
    gr.Markdown("Two separate models analyze chest X-rays and CT volumes. Outputs are experimental and are not calibrated clinical probabilities.")
    with gr.Tab("Chest X-ray"):
        chest = gr.File(label="PNG or JPEG", file_types=[".png", ".jpg", ".jpeg"], type="filepath")
        chest_button = gr.Button("Analyze X-ray", variant="primary")
        findings = gr.Dataframe(headers=["Finding", "Score"], datatype=["str", "number"], interactive=False)
        chest_button.click(classify, inputs=chest, outputs=findings)
    with gr.Tab("Spleen CT"):
        volume = gr.File(label="3D NIfTI (.nii or .nii.gz)", file_types=[".nii", ".nii.gz"], type="filepath")
        ct_button = gr.Button("Segment spleen", variant="primary")
        gr.Markdown("A 3D model generates mask overlays; this does not establish accuracy for the uploaded case.")
        with gr.Row():
            sagittal = gr.Image(label="Sagittal", type="pil")
            coronal = gr.Image(label="Coronal", type="pil")
            axial = gr.Image(label="Axial", type="pil")
        info = gr.Textbox(label="Output", interactive=False)
        ct_button.click(segment, inputs=volume, outputs=[sagittal, coronal, axial, info], time_limit=60)
    gr.Markdown("This hosted copy runs the same model code as the local React/FastAPI app, but shows only the chest X-ray scores and the spleen segmentation. Grad-CAM, SHAP, the review log, the Q&A panel and Orthanc are in the local app. The chest X-ray model scores 0.797 macro AUROC on 999 NIH images; the spleen model scores a mean Dice of 0.877 on 6 held-out CT cases (0.715 before the largest-blob clean-up, which was chosen on validation cases and applied after a first test run). Both are small samples. [Project repository](https://github.com/ram15x/nexvia-aiml-02-medical-imaging)")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), max_file_size="128mb")