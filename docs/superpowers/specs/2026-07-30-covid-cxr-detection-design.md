# COVID-19 Detection in Chest X-Rays — Design

**Date:** 2026-07-30
**Status:** Approved
**Scope:** Portfolio repository plus written report, roughly 3 weekends of work.

---

## 1. Purpose and framing

Build a 4-class chest X-ray classifier — Normal / Lung Opacity / Viral Pneumonia / COVID-19 — and measure how much of its accuracy comes from lung pathology rather than from artefacts of how the dataset was assembled.

The repository's claim is not "I achieved X% accuracy." It is **"I achieved X% accuracy, and here is how much of it is real."** The baseline model and the audit that qualifies it are equal deliverables. Roughly every published COVID-CXR portfolio project reports the first number; almost none report the second.

### Why the audit is the centre of the project

Two documented facts about this dataset drive the design:

1. **The Viral Pneumonia class is pediatric.** Those images originate from Kermany et al. — patients aged 1–5 at Guangzhou Women and Children's Medical Center. The COVID images are adults. A network can separate the classes on ribcage size and scapular ossification without ever looking at lung tissue.
2. **Source mixing produces shortcuts.** DeGrave et al. (*Nature Machine Intelligence*, 2021) showed that COVID-CXR models trained on aggregated datasets of exactly this kind key on laterality markers, text burn-ins, patient positioning and scanner processing, then collapse when tested at a new hospital. Because COVID and control images were drawn from different repositories, this dataset is close to a worst case.

A high accuracy number on this dataset is therefore not evidence of a working classifier. Quantifying the gap is the contribution.

### Why four classes rather than three

The obvious framing is 3-class (Normal / Viral Pneumonia / COVID), and it was the initial plan. It was rejected because **it leaves the audit itself confounded.** If the shortcut probes score highly on a 3-class task, the result is uninterpretable: there is no way to tell whether the model found the pediatric artefact in Viral Pneumonia or general source artefacts spread across the whole dataset. One dominant confound masks every other signal, and there is no control to separate them.

Including Lung Opacity supplies that control. Both COVID and Lung Opacity are **adult** images, so the age confound disappears for that specific pair. Every probe can then be reported twice — across all four classes, and restricted to the COVID vs. Lung Opacity pair. If an 8×8 logistic regression separates Viral Pneumonia from everything but fails on COVID vs. Lung Opacity, the pediatric artefact is isolated as the dominant shortcut and the remaining signal is plausibly real pathology.

This control is good but not clean, and the writeup must say so: COVID and Lung Opacity still originate from **different source repositories** (COVID from SIRM and published case reports; Lung Opacity from RSNA). Removing the age confound does not remove source artefacts.

Lung Opacity also makes the task clinically honest. Normal vs. COVID asks whether the lungs are abnormal at all. COVID vs. Lung Opacity asks whether an abnormal lung is COVID or something else — the question a clinician actually faces. Excluding it inflates the headline number by removing the hard decision.

### Success criteria

- A trained DenseNet121 baseline with a full, honestly-reported metric suite
- A lung-masked counterpart, and a measured gap between the two
- Two cheap shortcut probes that bound how much label information sits outside the lungs, each reported both across all four classes and restricted to the COVID vs. Lung Opacity control pair
- Grad-CAM reported as a **number** (attribution mass inside the lung mask), not only as illustrative heatmaps
- A README a reader can skim in five minutes and understand both the result and its limits

### Non-goals

External validation on RSNA or Cohen · deployment or a demo app · ensembles · architecture search · training a segmentation model · CT / 3D imaging · lesion localization or detection.

---

## 2. Data layer

### Source

Kaggle `tawsifurrahman/covid19-radiography-database` (v5). Approximately 21,165 PNG images at 299×299 grayscale, shipped **with lung segmentation masks** for every image. The masks make the entire audit possible and are the reason this dataset was chosen over alternatives.

All four classes are used:

| Class | Approx. count | Population | Primary source |
|---|---|---|---|
| Normal | 10,192 | mixed | RSNA + Kermany |
| Lung Opacity | 6,012 | adult | RSNA |
| COVID | 3,616 | adult | SIRM + published case reports |
| Viral Pneumonia | 1,345 | **pediatric (ages 1–5)** | Kermany / Guangzhou |
| **Total** | **≈ 21,165** | | |

Worst-case imbalance is Normal against Viral Pneumonia at ≈ 7.6 : 1 — identical to the 3-class framing, since neither extreme changes.

**Note on label semantics.** These four labels are not a clean partition of disease space; they are an artefact of which datasets were merged. A pediatric viral pneumonia *is* a lung opacity, and COVID *is* a viral pneumonia. `Lung Opacity` specifically means "visible lung opacity, cause unspecified, not confirmed COVID" — RSNA chose that deliberately vague wording because a radiograph often cannot reveal the cause of an opacity. The classes are mutually exclusive only by provenance bookkeeping. The README must state this plainly.

**Verified at download, 2026-07-30.** All four counts above are exact. Images are 299×299 mode `L`; masks are **256×256 mode `RGB`** — binary content stored across three identical channels, and present for all 21,165 images. Two consequences: masks must be resized with nearest-neighbour to match the images, and any PIL-based mask read yields `(H, W, 3)` and must be collapsed to one channel before being used to index a 2-D array.

### De-duplication

The dataset was aggregated from 43 publications plus the SIRM repository, so exact and near-duplicate images exist. Duplicates spanning a train/test boundary inflate every reported metric.

- **Exact:** MD5 over decoded pixel bytes
- **Near:** perceptual hash (`imagehash`), Hamming distance ≤ 1

**Threshold justification (measured 2026-07-30).** The conventional ≤ 5 removed
1,529 images and produced 172 cross-class groups; visual inspection confirmed
those matches were *different patients*. Chest radiographs are structurally
uniform, so a 64-bit DCT hash matches "frontal CXR" rather than "same image".
At ≤ 1 the pass finds 178 groups and 223 removals with **zero** cross-class
matches, while still finding four times what MD5 alone does. At ≤ 8 union-find
chaining collapses the dataset entirely (14,403 removed from only 355 groups),
which is why `summarise_duplicates` reports `max_group_size`.

**Finding: duplicates are almost entirely a COVID-class phenomenon.**
COVID 214 removed (5.9%), Viral Pneumonia 7 (0.5%), Normal 2 (0.02%),
Lung Opacity 0. This is the aggregation history made measurable — COVID images
were collected from 43 publications and SIRM, where the same case recurs across
papers, whereas Normal and Lung Opacity arrived wholesale from RSNA. It is
independent evidence that the COVID class differs from its controls in
provenance, not only in pathology, and belongs in the README alongside the
shortcut-probe results.
- Run **before** splitting, over the full four-class pool (≈ 21,165 images)
- When a duplicate group is found, retain the first member by sorted filename and drop the rest, so the retained set is deterministic across runs
- Report counts removed, separated into within-class and **cross-class** duplicates. Cross-class duplicates are label noise and are reported in the README as a dataset finding.

### Splitting

Stratified 70/15/15 train/validation/test on the de-duplicated pool, fixed random seed. Split membership is written to `data/splits/{train,val,test}.csv` and **committed to the repository**, so any reader reproduces the exact partition.

The test set is created on day one and is not read again until the final evaluation notebook.

### Documented limitation: no patient-level split

The correct practice is to split by patient ID so that no patient appears in two splits. **This dataset carries no patient identifiers** — files are named `COVID-1.png … COVID-3616.png` with no metadata linking images to patients. Patient-disjoint splitting is therefore impossible.

Near-duplicate removal is a partial mitigation, not a substitute: two different radiographs of the same patient taken days apart will not be caught by a perceptual hash. Some residual patient-level leakage is likely, and reported metrics should be read as upper bounds.

This limitation is stated in the README rather than omitted. (If a patient-disjoint evaluation later becomes a priority, RSNA DICOMs do carry `PatientID` — but external validation is out of scope here.)

---

## 3. Preprocessing

- Resize to **224×224**, bilinear. Sources are 299×299, so downsampling is unavoidable; 224 matches DenseNet121's ImageNet pretraining and is about 1.3× cheaper than 256.
- Replicate grayscale to 3 channels.
- Normalise with `keras.applications.densenet.preprocess_input`, which already applies torch-mode scaling (divide by 255, then ImageNet mean and standard deviation). Do not hand-roll this — mismatched normalisation against pretrained weights is the most common silent bug in this class of project.

Pipeline, driven by the split manifests: decode → resize → cache → augment (train only) → `preprocess_input` → batch → prefetch. Built on `tf.data`.

**Masked variant:** resize the mask to 224×224 with nearest-neighbour, then multiply. The background is **zeroed, not cropped** — cropping to the lung bounding box would introduce lung size and position as a fresh shortcut, defeating the purpose.

---

## 4. Augmentation

Training split only, implemented with Keras preprocessing layers so it executes on-GPU inside the graph. Albumentations was considered and rejected: it requires a `tf.numpy_function` bridge that becomes the pipeline bottleneck under TensorFlow. It remains an option if CLAHE later proves necessary.

| Layer | Setting | Rationale |
|---|---|---|
| `RandomFlip("horizontal")` | — | Anatomically acceptable, and disrupts the L/R laterality-marker shortcut |
| `RandomRotation` | ±0.04 (≈ ±15°) | Patient positioning variance |
| `RandomZoom` | ±10% | Source-to-detector distance variance |
| `RandomTranslation` | ±5% | Framing variance |
| `RandomContrast` | 0.2 | Exposure setting variance |
| `RandomBrightness` | 0.15 | Exposure setting variance |

Excluded: vertical flips (lungs are not vertically symmetric), shear, and hue/saturation operations.

Brightness and contrast jitter are deliberately included. Chest radiograph pixel values are relative and already vary substantially with exposure settings, so this augmentation reflects real acquisition variance. (Hounsfield units, which are absolute, are a CT concept and do not apply.)

---

## 5. Experiments

Four runs, each configured by a YAML file in `configs/`.

| # | Input | Model | Purpose |
|---|---|---|---|
| 1 | Raw image | DenseNet121 | Headline number |
| 2 | Lung-masked | DenseNet121 | Signal from lung tissue only; the gap against run 1 is the finding |
| 3 | Raw image at 8×8 | Multinomial logistic regression | Probe: is the label recoverable from global intensity structure alone? |
| 4 | Lungs blacked out | DenseNet121 | Probe: how well does the model score with the pathology removed? |

Run 3 downsamples the **raw** (unmasked) image to 8×8. Run 4 is the exact inverse of run 2: pixels **inside** the lung mask are zeroed and everything outside is retained.

Runs 3 and 4 are the diagnostic core. If a logistic regression on 64 pixels separates the classes well, or if a network scores highly on images where the lungs have been erased, then the label is predictable from non-pathological signal and the headline number is largely shortcut.

### Control-pair analysis

Every run's metrics are reported twice:

1. **All four classes** — the headline evaluation.
2. **Restricted to COVID vs. Lung Opacity** — computed from the same trained model by scoring only the test images belonging to those two classes and comparing their predicted logits. No retraining is required.

Both classes in the restricted pair are adult, so the pediatric confound is absent. Reading the two views together is what makes the audit interpretable: high probe scores on all four classes combined with near-chance probe scores on the restricted pair localises the shortcut to the pediatric artefact. High probe scores on *both* views indicate source artefacts that pervade the whole dataset.

### Training recipe (runs 1, 2 and 4 — identical)

- **Head:** `GlobalAveragePooling2D → Dropout(0.3) → Dense(4, softmax)`
- **Stage A:** base frozen, Adam lr = 1e-3, up to 8 epochs
- **Stage B:** unfreeze from `conv5_block1` onward, Adam lr = 1e-5, up to 15 epochs
- Batch size 32, mixed precision
- `class_weight='balanced'` for the ≈7.6:1 imbalance. **Class weighting only** — stacking oversampling on top distorts probability calibration and makes the calibration analysis meaningless.
- Early stopping and checkpointing on **validation macro-F1**, not accuracy, since accuracy is dominated by the Normal class. Patience 3 epochs in Stage A, 5 in Stage B; best weights restored at the end of each stage.
- Checkpoint to Google Drive every epoch

**BatchNorm during fine-tuning:** setting `base.trainable = True` in Keras also re-enables BatchNorm statistics updates, which can destabilise a small-batch fine-tune. BatchNorm layers are held in inference mode during Stage B.

Run 3 uses a multinomial logistic regression on 8×8 grayscale (64 features), with a small 2-layer MLP as a secondary check if the linear model underperforms.

### Infrastructure

Training runs on Colab GPUs. The local machine (8 cores, ~7.6 GB RAM, no GPU) handles EDA, hashing-based de-duplication, and results analysis. All seeds fixed and recorded.

The four-class pool is ≈ 40% larger than a three-class pool, so epochs are correspondingly longer — roughly 2–3 additional hours of GPU time across runs 1, 2 and 4 combined. Run 3 is unaffected.

---

## 6. Evaluation

Computed on the frozen test set for every run, and collected into a single table in the README:

- Per-class precision, recall and F1, plus macro-F1
- **COVID sensitivity and specificity**, and specificity at a fixed 95% COVID-sensitivity operating point. Argmax alone hides the trade-off that matters clinically: a false negative discharges an infectious patient.
- Per-class one-vs-rest ROC-AUC and PR-AUC. PR-AUC is reported alongside ROC-AUC because ROC-AUC is optimistic under class imbalance.
- Confusion matrix, 4×4, both raw counts and row-normalised
- **The COVID ↔ Lung Opacity confusion cells, called out explicitly.** These two numbers carry more diagnostic information than the rest of the matrix combined: they measure whether the model can distinguish COVID from another cause of opacity in a same-age population. Expect this to be the model's weakest pair, and expect overall macro-F1 to land well below what a 3-class framing would report. That drop is a more honest number, not a regression.
- **Bootstrap 95% confidence intervals**, 2,000 resamples of the test set. The realised test split holds 200 Viral Pneumonia against 1,529 Normal, 902 Lung Opacity and 511 COVID, so per-class precision is very uneven; reporting bare point estimates would misrepresent it.
- Calibration: reliability diagram and expected calibration error

---

## 7. Confound audit

- **Grad-CAM** on DenseNet121's final convolutional block, per predicted class.
- **Lung Attribution Ratio** — the fraction of Grad-CAM attribution mass falling inside the lung mask, reported as mean with a bootstrap confidence interval across the test set, for runs 1 and 2. This converts explainability from illustration into measurement: "62% of attribution mass inside the lungs for the raw model versus 94% for the masked model" is a result; a heatmap is an anecdote.
- **Lung Attribution Ratio broken out per class.** A low ratio on Viral Pneumonia predictions alongside a high ratio on COVID and Lung Opacity predictions would corroborate the pediatric-shortcut hypothesis from an independent direction.
- **Qualitative panel:** the same six test images shown side by side under the raw and masked models, including at least one COVID case and one Lung Opacity case.
- **Failure gallery:** the highest-confidence incorrect predictions, where shortcut reliance is typically most visible.

---

## 8. Repository structure

```
covid-xray-detection/
├── README.md               # results table, figures, limitations
├── requirements.txt
├── configs/                # one YAML per run
├── data/                   # images gitignored
│   └── splits/{train,val,test}.csv    # committed
├── docs/superpowers/specs/ # this document
├── notebooks/
│   ├── 01_eda_and_dedup.ipynb
│   ├── 02_train.ipynb      # config-driven; runs any of the four experiments
│   ├── 03_evaluate.ipynb
│   └── 04_gradcam_audit.ipynb
├── src/covid_xray/
│   ├── data.py             # manifests, tf.data pipeline, mask application
│   ├── augment.py          # augmentation layer stack
│   ├── models.py           # DenseNet121 builder, probe models
│   ├── train.py            # two-stage loop, callbacks
│   ├── evaluate.py         # metric suite, bootstrap CIs
│   └── gradcam.py          # CAM computation, lung attribution ratio
└── reports/figures/
```

Each module has one responsibility and a small surface: `data.py` turns a config into a `tf.data.Dataset`; `models.py` turns a config into a compiled model; `evaluate.py` turns predictions plus labels into a metrics dictionary; `gradcam.py` turns a model plus an image into a heatmap and a ratio. None needs to read another's internals.

Notebooks hold orchestration and figure generation only. Logic lives in `src/` so it is testable, reviewable, and re-runnable across Colab sessions without copy-paste drift.

---

## 9. Risks

- **The lung masks are model-generated** by the dataset authors, not drawn by radiologists. They are imperfect, and both the masked-model result and the Lung Attribution Ratio inherit that imperfection. This is stated wherever those numbers appear.
- **De-duplication may remove a non-trivial share of images.** Class counts shift, and the README must report post-de-duplication numbers throughout rather than the headline Kaggle figures.
- **Colab sessions terminate without warning.** Mitigated by per-epoch checkpointing to Drive and by keeping training logic in importable modules.
- **Viral Pneumonia is the small class.** Its confidence intervals will be wide. This is reported as a finding about the dataset, not smoothed over.
- **The four labels are not a clean partition** (see §2). COVID, Viral Pneumonia and Lung Opacity overlap conceptually, and the model is being asked to reproduce a provenance-derived taxonomy rather than a diagnostic one. This is the hardest paragraph in the writeup and the easiest to fudge; it must be stated directly rather than buried.
- **Headline macro-F1 will be materially lower than a 3-class framing would produce**, because COVID vs. Lung Opacity is a genuinely hard discrimination. If the number is read out of context it looks like a worse project. The README must frame it as the harder and more meaningful task up front, not defensively at the end.
- **The audit may produce an unflattering result** — the shortcut probes may score high, and the masked model may lose substantial accuracy. That outcome is the project's most valuable output and is reported as prominently as the headline number.

---

## 10. Working agreement

Implementation code is delivered in chat for manual copying into the repository, one complete file per block labelled with its intended path. README and requirements files are written directly to disk.
