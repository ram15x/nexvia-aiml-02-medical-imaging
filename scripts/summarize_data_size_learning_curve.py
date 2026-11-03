from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_1000_image_learning_curve import metrics
BASE=ROOT/'data'/'NexviaProjectData'/'nih'
stages=[
    (1500, 'learning_curve_1500_train_1000_val_20261031', 'baseline_800', 'candidate_1500', 'score_train1500_'),
    (2500, 'learning_curve_2500_train_1000_val_20261031', 'baseline_1500', 'candidate_2500', 'score_train2500_'),
    (4000, 'learning_curve_4000_train_1000_val_20261031', 'baseline_2500', 'candidate_4000', 'score_train4000_'),
    (6000, 'learning_curve_6000_train_1000_val_20261031', 'baseline_4000', 'candidate_6000', 'score_train6000_'),
    (8000, 'learning_curve_8000_train_1000_val_20261031', 'baseline_6000', 'candidate_8000', 'score_train8000_'),
    (10000,'learning_curve_10000_train_1000_val_20261031','baseline_8000', 'candidate_10000','score_train10000_'),
]
first=BASE/stages[0][1]/'validation_predictions.csv'
frame=pd.read_csv(first)
ids=frame['Image Index'].astype(str).tolist()
labels=[c.removeprefix('target_') for c in frame.columns if c.startswith('target_')]
y=frame[[f'target_{x}' for x in labels]].to_numpy(dtype=np.int8)
scores={800:frame[[f'score_train800_{x}' for x in labels]].to_numpy(dtype=np.float32)}
summary800=json.loads((BASE/stages[0][1]/'metrics_summary.json').read_text(encoding='utf-8'))
metrics_by_size={800:metrics(y,scores[800])}
runtimes={800:float(summary800['baseline_800']['metrics'].get('elapsed_seconds',0))}
# The original 800-image run summary provides its measured fit time.
runtimes[800]=float(json.loads((BASE/'learning_curve_1000_train400_vs800_20261031'/'densenet121_train800'/'run_summary.json').read_text(encoding='utf-8'))['elapsed_seconds'])
run_summaries={}
for size,dirname,basekey,candkey,prefix in stages:
    folder=BASE/dirname
    pred=pd.read_csv(folder/'validation_predictions.csv')
    if pred['Image Index'].astype(str).tolist()!=ids:
        raise ValueError(f'Validation ordering changed in {dirname}')
    yp=pred[[f'target_{x}' for x in labels]].to_numpy(dtype=np.int8)
    if not np.array_equal(y,yp):
        raise ValueError(f'Validation labels changed in {dirname}')
    # Check the prior rung is the exact same model score when carried forward.
    if size==1500:
        previous=800
    elif size==2500: previous=1500
    elif size==4000: previous=2500
    elif size==6000: previous=4000
    elif size==8000: previous=6000
    else: previous=8000
    base_scores=pred[[f'score_train{previous}_{x}' for x in labels]].to_numpy(dtype=np.float32)
    if previous in scores and not np.allclose(base_scores,scores[previous],atol=1e-6,rtol=1e-5):
        raise ValueError(f'Prior model score mismatch at {size} stage (baseline {previous})')
    cand_scores=pred[[prefix+x for x in labels]].to_numpy(dtype=np.float32)
    scores[size]=cand_scores
    metrics_by_size[size]=metrics(y,cand_scores)
    js=json.loads((folder/'metrics_summary.json').read_text(encoding='utf-8'))
    run_summaries[size]=js
    runtimes[size]=float(js['training']['candidate_elapsed_seconds'])

# Adjacent paired comparisons from the stored stage summaries.
adjacent=[]
previous=800
for size,dirname,basekey,candkey,_ in stages:
    js=run_summaries[size]
    # Extract the field that uses explicit candidate-minus-baseline naming.
    pair=js.get(f'paired_bootstrap_{size}_minus_{previous}',{}).get('deltas_candidate_minus_baseline')
    if pair is None:
        raise ValueError(f'Missing paired delta for {size} stage')
    adjacent.append({'from':previous,'to':size,'delta':pair})
    previous=size

out=ROOT/'reports'
out.mkdir(exist_ok=True)
# Learning-curve plot drawn with Pillow; separate panels keep scales legible.
xs=np.array([800,1500,2500,4000,6000,8000,10000])
plot=out/'data_size_learning_curve_2026-10-31.png'
canvas=Image.new('RGB',(1800,720),'white')
draw=ImageDraw.Draw(canvas)
try:
    font=ImageFont.truetype('arial.ttf',18)
    small=ImageFont.truetype('arial.ttf',14)
    titlefont=ImageFont.truetype('arial.ttf',23)
except OSError:
    font=small=titlefont=ImageFont.load_default()
draw.text((45,22),'DenseNet-121 data-size curve — fixed 1,000-case development panel',fill='#111827',font=titlefont)
panels=[('Macro AUROC','macro_auroc_defined_labels',.60,.82,'#2563eb'),('Macro average precision','macro_average_precision_positive_labels',0,.20,'#0891b2'),('Micro F1 @ threshold 0.5','micro_f1_at_0_5',.14,.29,'#7c3aed')]
for (title,key,ymin,ymax,color),left in zip(panels,[55,655,1255]):
    top,bottom=100,570
    right=left+510
    draw.text((left+35,68),title,fill='#111827',font=font)
    for j in range(5):
        value=ymin+(ymax-ymin)*j/4
        yline=bottom-round((bottom-top)*j/4)
        draw.line((left+70,yline,right,yline),fill='#d1d5db',width=1)
        draw.text((left+5,yline-9),f'{value:.2f}',fill='#4b5563',font=small)
    draw.line((left+70,top,left+70,bottom),fill='#374151',width=2)
    draw.line((left+70,bottom,right,bottom),fill='#374151',width=2)
    vals=[metrics_by_size[int(n)][key] for n in xs]
    points=[]
    for idx,(n,val) in enumerate(zip(xs,vals)):
        x=left+70+round((right-(left+70))*idx/(len(xs)-1))
        yline=bottom-round((val-ymin)/(ymax-ymin)*(bottom-top))
        points.append((x,yline))
        draw.text((x-24,bottom+12),f'{int(n)/1000:g}k' if n>=1000 else '800',fill='#374151',font=small)
    draw.line(points,fill=color,width=4)
    for x,yline in points:
        draw.ellipse((x-7,yline-7,x+7,yline+7),fill=color,outline='white',width=2)
draw.text((45,655),'One seed per size · fixed 10 epochs · reused development panel · F1 threshold is not tuned',fill='#4b5563',font=small)
canvas.save(plot)

fmt=lambda x:f'{x:.4f}'
lines=[]
lines += ['# Project 1: incremental data-size learning curve','', '**Run date:** 2026-10-31  ', '**Question:** Does adding training data help enough to justify its cost, or has the current pipeline plateaued?','']
lines += ['## Protocol','', '- DenseNet-121, ImageNet initialization, 224×224 input, weighted BCE, batch size 4, learning rate 1e-4, seed 20261031, and 10 fixed epochs at each training size.', '- Nested cohorts: each step retains the prior training patients and adds distinct patients from the same local NIH train split.', '- Every checkpoint was scored on the same 1,000 patient-disjoint development cases. The official test split was not read or evaluated.', '- This panel was used repeatedly during development. The paired bootstrap intervals describe uncertainty on these cases; they do not undo panel reuse or seed-selection bias.', '- Each size has one training seed. These are exploratory comparisons, not final independent validation.','']
lines += ['## Final-epoch scores','', '| Train patients | GPU training time | Macro AUROC | Macro AP | Micro F1 @ 0.5 |','|---:|---:|---:|---:|---:|']
for size in xs:
    t=metrics_by_size[int(size)]
    seconds=runtimes[int(size)]
    mins=seconds/60
    lines.append(f'| {int(size):,} | {mins:.1f} min | {fmt(t["macro_auroc_defined_labels"])} | {fmt(t["macro_average_precision_positive_labels"])} | {fmt(t["micro_f1_at_0_5"])} |')
total_seconds=sum(runtimes.values())
lines.append(f'')
lines.append(f'Cumulative measured training time across these seven fixed-epoch runs: {total_seconds/3600:.2f} GPU-hours. No new images were downloaded for this experiment.')
lines += ['', '![DenseNet-121 learning curve across training sizes](data_size_learning_curve_2026-10-31.png)','', 'F1 at 0.5 is a thresholded metric, not accuracy. It is shown at the fixed default threshold to make runs comparable; no threshold tuning is applied.','']
lines += ['## Adjacent data-rung differences','', 'Paired candidate-minus-baseline bootstrap differences (1,000 patient-row resamples):','', '| Step | Δ Macro AUROC (95% interval) | Δ Macro AP (95% interval) | Δ Micro F1 @ 0.5 (95% interval) |','|---|---|---|---|']
for item in adjacent:
    d=item['delta']
    def cell(k):
        o=d[k]
        return f'{o["point_difference"]:+.4f} [{o["lower_2_5_percent"]:+.4f}, {o["upper_97_5_percent"]:+.4f}]'
    lines.append(f'| {item["from"]:,}→{item["to"]:,} | {cell("macro_auroc_defined_labels")} | {cell("macro_average_precision_positive_labels")} | {cell("micro_f1_at_0_5")} |')
lines += ['', '## What the measurements support','', '- The point estimates improved overall from 800 to 6,000 training patients: macro AUROC 0.635→0.744, macro AP 0.081→0.138, and micro F1@0.5 0.177→0.227. This is a useful directional signal, but one seed and a reused development panel prevent a strong generalization claim.', '- The largest clear adjacent AP gain was 4,000→6,000 (+0.0210; paired interval +0.0038 to +0.0510). The AUROC and F1 changes there were near zero.', '- Increasing from 6,000 to 8,000 yielded only +0.006 AP (interval crosses zero); 8,000 to 10,000 reduced AP by about 0.005 (interval crosses zero). AUROC and F1 were effectively flat across both final rungs. On this evidence, the curve has plateaued by roughly 6,000–8,000 for this fixed training recipe.', '- The 10,000 candidate took about an hour to train for 10 epochs on the RTX 3050. It did not improve the final score over 8,000. Therefore, simply using more data is not currently justified as the next move.', '- Absolute quality remains limited: at 10,000, macro AUROC is about 0.746, macro AP about 0.139, and micro F1@0.5 about 0.218. That is not “85% accuracy,” and no overall accuracy metric is claimed.', '- Rare-label estimates are too noisy on this 1,000-case panel: positive counts include 2 Hernia, 3 Edema, and 5 Pneumonia. Their per-label scores should not guide architecture choices.','']
lines += ['## Follow-up after the data-size curve','', 'A subsequent 2,500-patient loss screen compared weighted BCE and Asymmetric Loss; see `asymmetric_loss_ablation_2026-10-31.md`. ASL improved macro AP by +0.0106, but its paired interval includes zero; macro AUROC declined uncertainly, and micro F1 at 0.5 was significantly lower. Do not promote ASL from this screen. A 0.7 threshold showed a better F1 on this same reused validation panel, but that is descriptive and threshold selection here would be biased. Next, calibrate thresholds from training-only out-of-fold predictions or a separate calibration split, then lock the operating point before a single final evaluation on the reserved official test. [Ridnik et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html).','', '## Reproducibility files','', '- `scripts/run_1500_patient_growth_stage.py` through `scripts/run_10000_patient_growth_stage.py`: data-size stage runners.', '- `scripts/run_asymmetric_loss_2500_ablation.py`: matched BCE-vs-ASL screen.', '- `data/NexviaProjectData/nih/learning_curve_*_train_1000_val_20261031/`: manifests, audits, scores, checkpoints, and run metadata.', '- `scripts/summarize_data_size_learning_curve.py`: rebuilds this report and plot from saved data-size artifacts.','']
report=out/'data_size_learning_curve_2026-10-31.md'
report.write_text('\n'.join(lines),encoding='utf-8')
summary={'sizes':{str(n):{'metrics':metrics_by_size[n], 'training_seconds':runtimes[n]} for n in metrics_by_size},'adjacent_deltas':adjacent,'same_validation_ids':len(ids),'labels':labels,'official_test_used':False,'validation_panel_reused':True,'training_seed':20261031,'checkpoints':'last/fixed epoch 10'}
(out/'data_size_learning_curve_2026-10-31.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'report':str(report),'plot':str(plot),'summary':{str(n):{k:metrics_by_size[n][k] for k in ['macro_auroc_defined_labels','macro_average_precision_positive_labels','micro_f1_at_0_5']} for n in metrics_by_size},'adjacent':adjacent},indent=2))
