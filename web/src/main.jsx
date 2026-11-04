import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = (() => {
  const url = new URL(import.meta.env.VITE_API_URL || 'http://localhost:8000', window.location.origin);
  const localHosts = new Set(['localhost', '127.0.0.1']);
  if (localHosts.has(url.hostname) && localHosts.has(window.location.hostname)) {
    url.hostname = window.location.hostname;
  }
  return url.href.replace(/\/$/, '');
})();
const FROZEN_EVALUATION = {
  checkpointSha256: '5634f21659d2cb8bf57bb90ff45797bd998beef15e1231e7f916f953f610db87',
  images: 500,
  patients: 58,
  macroAUROC: 0.7116412172524108,
  macroAuprc: 0.17552140259240692,
  microF1: 0.27929568913175473,
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const [reviewState, setReviewState] = useState('');
  const [selectedLabel, setSelectedLabel] = useState('');
  const [gradcamUrl, setGradcamUrl] = useState('');
  const [gradcamBusy, setGradcamBusy] = useState(false);
  const [syncBusy, setSyncBusy] = useState(false);
  const [syncMessage, setSyncMessage] = useState('');
  const [ctFile, setCtFile] = useState(null);
  const [orthancSeriesId, setOrthancSeriesId] = useState('');
  const [orthancExpectedCount, setOrthancExpectedCount] = useState('');
  const [ctResult, setCtResult] = useState(null);
  const [ctBusy, setCtBusy] = useState(false);
  const [ctError, setCtError] = useState('');
  const [ctPlane, setCtPlane] = useState('axial');

  async function segmentCt(event) {
    event.preventDefault();
    if (!ctFile) return;
    setCtBusy(true); setCtError(''); setCtResult(null);
    try {
      const body = new FormData(); body.append('file', ctFile);
      const response = await fetch(`${API}/segment/spleen`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The segmentation service could not process this volume.');
      setCtResult(data);
      setCtPlane('axial');
    } catch (err) { setCtError(err.message || 'Could not reach the local API.'); }
    finally { setCtBusy(false); }
  }

  async function segmentOrthancCt(event) {
    event.preventDefault();
    const seriesId = orthancSeriesId.trim();
    if (!seriesId) return;
    setCtBusy(true); setCtError(''); setCtResult(null);
    try {
      const count = orthancExpectedCount.trim();
      const query = count ? `?expected_instances=${encodeURIComponent(count)}` : '';
      const response = await fetch(`${API}/orthanc/series/${encodeURIComponent(seriesId)}/segment/spleen${query}`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Orthanc CT series could not be processed.');
      setCtResult(data);
      setCtPlane('axial');
    } catch (err) { setCtError(err.message || 'Could not reach the Orthanc CT service.'); }
    finally { setCtBusy(false); }
  }

  function downloadSpleenMask() {
    if (!ctResult) return;
    const binary = atob(ctResult.output_mask_nii_gz_base64);
    const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/gzip' }));
    const link = document.createElement('a');
    link.href = url; link.download = ctResult.output_mask_filename;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function chooseFile(event) {
    const next = event.target.files?.[0];
    setResult(null); setReviewState(''); setError(''); setFile(next || null); setSelectedLabel('');
    if (gradcamUrl) URL.revokeObjectURL(gradcamUrl);
    setGradcamUrl('');
    if (preview) URL.revokeObjectURL(preview);
    setPreview(next ? URL.createObjectURL(next) : '');
  }

  async function analyze(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true); setError(''); setResult(null); setGradcamUrl('');
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch(`${API}/predict`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The model service did not accept this image.');
      setResult(data);
      setSelectedLabel([...data.findings].sort((a, b) => b.score - a.score)[0]?.label || '');
    } catch (err) { setError(err.message || 'Could not reach the local API.'); }
    finally { setBusy(false); }
  }

  async function explain(event) {
    event.preventDefault();
    if (!file || !selectedLabel) return;
    setGradcamBusy(true); setError('');
    try {
      const body = new FormData(); body.append('file', file); body.append('label', selectedLabel);
      const response = await fetch(`${API}/explain`, { method: 'POST', body });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'The attribution view could not be created.');
      }
      if (gradcamUrl) URL.revokeObjectURL(gradcamUrl);
      setGradcamUrl(URL.createObjectURL(await response.blob()));
    } catch (err) { setError(err.message || 'Could not reach the local API.'); }
    finally { setGradcamBusy(false); }
  }

  async function saveReview(action) {
    if (!result) return;
    setReviewState('Saving…'); setError('');
    try {
      const response = await fetch(`${API}/reviews`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_ref: result.case_ref, reviewed_findings: [], note, reviewer_action: action }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Review record was not saved.');
      setReviewState(`Review recorded (${data.event_id.slice(0, 8)}).`);
    } catch (err) { setError(err.message || 'Could not save the review.'); setReviewState(''); }
  }

  async function syncOrthanc() {
    setSyncBusy(true); setError(''); setSyncMessage('');
    try {
      const response = await fetch(`${API}/orthanc/sync?limit=16`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Orthanc sync failed.');
      const retry = data.temporary_error ? 'A temporary connection problem stopped part of the batch; try again. ' : '';
      setSyncMessage(`${retry}Checked ${data.candidates_seen} instance(s): ${data.processed} processed, ${data.skipped} skipped. ${data.more_may_remain ? 'Run again to continue.' : ''}`);
    } catch (err) { setError(err.message || 'Could not reach the Orthanc service.'); }
    finally { setSyncBusy(false); }
  }

  return <main className="shell">
    <header className="topbar"><div className="brand-mark">N</div><div><p className="eyebrow">NEXVIA · AIML-02</p><h1>Chest imaging review</h1></div><span className="demo-tag">LOCAL DEMO</span></header>
    <aside className="notice"><strong>Coursework prototype.</strong> Model scores are experimental and are not a diagnosis or a substitute for a clinician's review. Do not upload real patient data to this demo.</aside>
    <section className="sync-panel"><div><p className="eyebrow">OPTIONAL LOCAL ORTHANC DEMO</p><strong>Check for new chest radiographs</strong><span>Reads Orthanc’s stable-study change log and processes at most 16 instances per click. Only single-frame CR/DX radiographs are sent to the chest X-ray classifier.</span></div><button className="secondary" onClick={syncOrthanc} disabled={syncBusy}>{syncBusy ? 'Checking…' : 'Sync new radiographs'}</button>{syncMessage && <p className="saved">{syncMessage}</p>}<form className="orthanc-ct-form" onSubmit={segmentOrthancCt}><label htmlFor="orthanc-series-id">Experimental spleen segmentation from one stable Orthanc CT series</label><input id="orthanc-series-id" value={orthancSeriesId} onChange={e => setOrthancSeriesId(e.target.value)} placeholder="Orthanc series ID" autoComplete="off" /><label htmlFor="orthanc-expected-count">Expected slice count from the data source (optional)</label><input id="orthanc-expected-count" type="number" min="2" max="512" step="1" value={orthancExpectedCount} onChange={e => setOrthancExpectedCount(e.target.value)} placeholder="Only if independently known" /><button className="secondary" disabled={!orthancSeriesId.trim() || ctBusy}>{ctBusy ? 'Running 3D model…' : 'Segment Orthanc CT series'}</button><small>For a series marked Unknown, provide the expected count only if the source independently supplies it. A matching count is not proof that every slice is correct or complete. Single-frame CT slices only; geometry and size are checked. No patient names or demographics are shown. The series reference and count basis are recorded; mask accuracy is unknown for this input.</small></form></section>
    <section className="workspace">
      <form className="card upload" onSubmit={analyze}>
        <div className="section-head"><div><p className="eyebrow">01 / IMAGE</p><h2>Select a chest image</h2></div><span className="format">PNG or JPEG</span></div>
        <label className="dropzone" htmlFor="image-file">
          {preview ? <img className="preview" src={preview} alt="Selected image preview" /> : <span className="placeholder"><span className="upload-icon">↑</span><strong>Choose an image</strong><span>Use a de-identified educational sample only.</span></span>}
          <input id="image-file" type="file" accept="image/png,image/jpeg" onChange={chooseFile} />
        </label>
        {file && <p className="file-name">{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</p>}
        <button className="primary" disabled={!file || busy}>{busy ? 'Running model…' : 'Run educational model'}</button>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      <section className="card results">
        <div className="section-head"><div><p className="eyebrow">02 / MODEL OUTPUT</p><h2>Finding scores</h2></div>{result && <span className="case-id">Case {result.case_ref.slice(0, 8)}</span>}</div>
        {!result ? <div className="empty"><div className="empty-icon">◎</div><strong>Scores will appear here</strong><span>Run the model to see its uncalibrated output.</span></div> : <>
          <p className="score-note">Uncalibrated model scores from 0 to 1. They are not diagnostic probabilities. {result.model}</p>
          {result.model_metadata?.promotion_evaluation && <p className="score-note">Current checkpoint development cohort ({result.model_metadata.promotion_evaluation.patients.toLocaleString()} patient-disjoint NIH patients): macro AUROC {result.model_metadata.promotion_evaluation.macro_auroc.toFixed(3)}, macro AP {result.model_metadata.promotion_evaluation.macro_average_precision.toFixed(3)}, micro-F1 {result.model_metadata.promotion_evaluation.micro_f1_at_0_5.toFixed(3)} at threshold 0.5. This same-source cohort was used to select the checkpoint; these are not independent test estimates, external validation, or clinical performance.</p>}
          {result.model_metadata?.checkpoint_sha256 === FROZEN_EVALUATION.checkpointSha256 && <p className="score-note">Frozen test sample: {FROZEN_EVALUATION.images} images / {FROZEN_EVALUATION.patients} patients. Macro AUROC {FROZEN_EVALUATION.macroAUROC.toFixed(3)}, AUPRC {FROZEN_EVALUATION.macroAuprc.toFixed(3)}, micro-F1 {FROZEN_EVALUATION.microF1.toFixed(3)} at 0.5. This is a small NIH test subset, not the full test set. See the model card for the limitations.</p>}
          <div className="scores">{result.findings.map(item => <div className="score-row" key={item.label}><span>{item.label.replaceAll('_', ' ')}</span><div className="track"><div className="fill" style={{ width: `${Math.max(1, item.score * 100)}%` }} /></div><code>{item.score.toFixed(3)}</code></div>)}</div>
          <form className="explain-box" onSubmit={explain}>
            <label htmlFor="explain-label">Coarse attribution view</label>
            <div className="explain-controls"><select id="explain-label" value={selectedLabel} onChange={e => setSelectedLabel(e.target.value)}>{result.findings.map(item => <option key={item.label} value={item.label}>{item.label.replaceAll('_', ' ')}</option>)}</select><button className="secondary" disabled={gradcamBusy || !selectedLabel}>{gradcamBusy ? 'Creating…' : 'Show Grad-CAM'}</button></div>
            {gradcamUrl && <figure className="explanation"><img src={gradcamUrl} alt={`Grad-CAM overlay for ${selectedLabel}`} /><figcaption>Coarse feature attribution only. It does not establish why the model scored the image and is not a clinical explanation.</figcaption></figure>}
          </form>
          <div className="review-box"><label htmlFor="review-note">Demo reviewer note <span>(no personal or patient details)</span></label><textarea id="review-note" maxLength="500" value={note} onChange={e => setNote(e.target.value)} placeholder="Optional annotation for this local demo…" />
            <div className="review-actions"><button type="button" className="secondary" onClick={() => saveReview('needs_review')}>Needs review</button><button type="button" className="secondary" onClick={() => saveReview('rejected')}>Reject output</button><button type="button" className="primary compact" onClick={() => saveReview('accepted_for_demo')}>Approve (demo only)</button></div>
            {reviewState && <p className="saved">{reviewState}</p>}
          </div>
        </>}
      </section>
    </section>
    <section className="card ct-panel">
      <div className="section-head"><div><p className="eyebrow">03 / EXPERIMENTAL CT</p><h2>Spleen segmentation</h2></div><span className="format">Orthanc DICOM series or NIfTI</span></div>
      <p className="ct-limit">Separate 3D U-Net workflow. NIfTI uploads and stable single-frame CT series from Orthanc are processed in a temporary folder and removed after the request. Maximum 128 MiB transfer and 45 million voxels. Geometry is preserved during DICOM-to-NIfTI conversion; arbitrary anatomy and acquisition conditions remain outside the training evidence.</p>
      <form className="ct-controls" onSubmit={segmentCt}>
        <input type="file" accept=".nii,.nii.gz,application/gzip" onChange={e => { setCtFile(e.target.files?.[0] || null); setCtResult(null); setCtError(''); }} />
        <button className="secondary" disabled={!ctFile || ctBusy}>{ctBusy ? 'Running 3D model…' : 'Segment spleen'}</button>
      </form>
      {ctFile && <p className="file-name">{ctFile.name} · {(ctFile.size / 1024 / 1024).toFixed(1)} MB</p>}
      {ctError && <p className="error" role="alert">{ctError}</p>}
      {ctResult && <div className="ct-result">
        <div className="plane-viewer">
          <div className="plane-tabs" role="tablist" aria-label="CT view plane">
            {['axial', 'coronal', 'sagittal'].map(plane => <button key={plane} type="button" role="tab" aria-selected={ctPlane === plane} className={ctPlane === plane ? 'plane-tab active' : 'plane-tab'} onClick={() => setCtPlane(plane)}>{plane}</button>)}
          </div>
          <img src={`data:image/png;base64,${ctResult.preview_planes[ctPlane].png_base64}`} alt={`${ctPlane} CT slice with predicted spleen mask overlaid in red`} />
          <small>RAS orientation · mask-centered slice {ctResult.preview_planes[ctPlane].slice_index}</small>
        </div>
        <div><strong>3D spleen model output</strong><p>Predicted mask: {ctResult.predicted_spleen_voxels.toLocaleString()} voxels · estimated volume {ctResult.estimated_predicted_volume_ml} mL. Output shape and affine match the input: {ctResult.output_geometry_matches_input ? 'yes' : 'no'}.</p>{ctResult.dicom_instance_count && <small>Orthanc source series · {ctResult.dicom_instance_count} DICOM slices · result recorded as {ctResult.result_id.slice(0, 8)}. {ctResult.completeness_basis === 'operator_supplied_count_match' ? 'Count supplied by operator; completeness is not independently proven.' : ''}</small>}<small>Inference: {ctResult.inference_augmentation}</small><p><button type="button" className="secondary mask-download" onClick={downloadSpleenMask}>Download binary NIfTI mask</button></p><p className="error">This input has no reference mask, so its accuracy is unknown. See the model card for the evaluation and its limits.</p><small>{ctResult.disclaimer}</small></div>
      </div>}
    </section>
    <footer><span>DenseNet-121 · NIH ChestX-ray14 label set</span><span>Scores are not calibrated probabilities · educational use only</span></footer>
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
