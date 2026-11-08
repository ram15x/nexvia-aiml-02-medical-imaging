from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'reports' / 'weight_soup_5k_development_recheck_2026-11-02.json'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, make_model, seed_everything
from evaluate_frozen_candidates_internal_holdout import score_checkpoint, summarize

nih = PROJECT / 'data' / 'NexviaProjectData' / 'nih'
cohort_dir = nih / 'candidate_holdout_additional_5k_20261031'
manifest = cohort_dir / 'manifest_additional_5000_validation.csv'
image_root = nih / 'images_224_additional_5k_20261030'
prior = pd.read_csv(cohort_dir / 'predictions_additional_5k.csv')
soup_path = PROJECT / 'reports' / 'frozen_candidates_2250_internal_holdout_2026-11-01' / 'densenet121_weight_soup_8k10k_bn_recalibrated.pt'
train8 = pd.read_csv(nih / 'learning_curve_8000_train_1000_val_20261031' / 'manifest_8000_train_1000_validation.csv')
train10 = pd.read_csv(nih / 'learning_curve_10000_train_1000_val_20261031' / 'manifest_10000_train_1000_validation.csv')
hold = pd.read_csv(manifest)
ids = set(hold['Patient ID'].astype(str))
train_ids = set(train8.loc[train8.split == 'train', 'Patient ID'].astype(str)) | set(train10.loc[train10.split == 'train', 'Patient ID'].astype(str))
if len(hold) != 5000 or hold['Patient ID'].nunique() != 5000 or ids & train_ids:
    raise ValueError('Cohort shape or training-patient separation check failed')
if not prior['Image Index'].astype(str).equals(hold['Image Index'].astype(str)):
    raise ValueError('Existing predictions are not aligned to the frozen cohort')

dataset = NIHChestXrayDataset(manifest, image_root, 'validation', None, 20261102, image_size=224)
if len(dataset) != 5000 or dataset.missing_images:
    raise ValueError(f'Expected 5000 available images, got {len(dataset)}, missing={dataset.missing_images}')
y = dataset.targets.astype(np.int32)
for j, label in enumerate(NIH_LABELS):
    col = f'target_{label}'
    if col not in prior or not np.array_equal(prior[col].to_numpy(dtype=np.int32), y[:, j]):
        raise ValueError(f'Prior prediction targets do not match frozen cohort: {label}')

def prior_score(name):
    return prior[[f'score_{name}_{label}' for label in NIH_LABELS]].to_numpy(dtype=np.float32)

scores = {
    'active_demo': prior_score('demo_current'),
    'imagenet_8k': prior_score('train8000'),
    'imagenet_10k': prior_score('train10000'),
}
scores['probability_average_8k_10k'] = (scores['imagenet_8k'] + scores['imagenet_10k']) / 2

seed_everything(20261102)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(soup_path, map_location='cpu', weights_only=False)
if checkpoint.get('labels') != list(NIH_LABELS):
    raise ValueError('Weight-soup label schema mismatch')
model = make_model(pretrained=False, architecture=checkpoint['architecture'], label_adjacency=checkpoint.get('label_adjacency'))
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.to(device).eval()
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
chunks=[]
with torch.inference_mode():
    for images, _ in loader:
        chunks.append(torch.sigmoid(model(images.to(device, non_blocking=device.type == 'cuda'))).cpu().numpy())
scores['weight_soup_8k_10k'] = np.concatenate(chunks).astype(np.float32)
if scores['weight_soup_8k_10k'].shape != y.shape:
    raise ValueError('Candidate predictions have unexpected shape')

summaries = {name: summarize(y, pred) for name, pred in scores.items()}
patient = hold['Patient ID'].astype(str).to_numpy()
unique = np.unique(patient)
groups = {pid: np.flatnonzero(patient == pid) for pid in unique}
rng = np.random.default_rng(20261102)
pairs = [('weight_soup_8k_10k','active_demo'), ('weight_soup_8k_10k','probability_average_8k_10k')]
deltas = {f'{a}_minus_{b}': {m: [] for m in ('macro_auprc','macro_auroc','micro_f1_at_0_5')} for a,b in pairs}
def compact_metrics(yy, pp):
    aps, aucs = [], []
    for j in range(yy.shape[1]):
        target, prob = yy[:, j], pp[:, j]
        if target.sum() > 0:
            aps.append(float(average_precision_score(target, prob)))
        if np.unique(target).size == 2:
            aucs.append(float(roc_auc_score(target, prob)))
    return {'macro_auprc': float(np.mean(aps)) if aps else None,
            'macro_auroc': float(np.mean(aucs)) if aucs else None,
            'micro_f1_at_0_5': float(f1_score(yy.ravel(), pp.ravel() >= 0.5, zero_division=0))}
for _ in range(500):
    chosen = rng.choice(unique, size=len(unique), replace=True)
    idx = np.concatenate([groups[pid] for pid in chosen])
    boot = {name: compact_metrics(y[idx], scores[name][idx])
            for name in ('weight_soup_8k_10k','active_demo','probability_average_8k_10k')}
    for a,b in pairs:
        key = f'{a}_minus_{b}'
        for metric in deltas[key]:
            va, vb = boot[a][metric], boot[b][metric]
            if va is not None and vb is not None and np.isfinite(va-vb):
                deltas[key][metric].append(float(va-vb))
intervals = {pair:{metric:{'median':float(np.median(vals)),'percentile_95_ci':[float(v) for v in np.quantile(vals,[.025,.975])],'valid_replicates':len(vals)} for metric,vals in fields.items()} for pair,fields in deltas.items()}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
result = {
 'dataset':'NIH ChestX-ray14-derived 5,000-patient development cohort',
 'patients':5000, 'images':5000, 'device':str(device), 'threshold':0.5,
 'cohort_manifest_sha256':sha(manifest), 'candidate_checkpoint_sha256':sha(soup_path),
 'patient_overlap_with_8k_or_10k_training':0, 'official_test_used':False,
 'existing_baselines_reused_from_prior_5k_evaluation':True,
 'results':{name:{k:v for k,v in row.items() if k!='labels'} for name,row in summaries.items()},
 'paired_patient_bootstrap_500_replicates':intervals,
 'positive_support_by_label':{label:int(y[:,i].sum()) for i,label in enumerate(NIH_LABELS)},
 'caveat':'This 5k cohort was previously used to evaluate the active, 8k, and 10k checkpoints. The weight-soup candidate was not in that prior comparison, but this is a reused development cohort, not an untouched or external validation set. NIH labels are report-mined; results do not establish clinical performance.',
}
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'candidate':summaries['weight_soup_8k_10k']|{'labels':'omitted'},'baselines':{k:{x:v for x,v in z.items() if x!='labels'} for k,z in summaries.items() if k!='weight_soup_8k_10k'},'paired_patient_bootstrap':intervals,'caveat':result['caveat']},indent=2))
