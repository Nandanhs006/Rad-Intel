# Rad-Intel — Project Presentation Brief

Status as of 9 October 2026. Written to be read aloud or skimmed before a
panel. Every number here was measured on the held-out 624-image test set and
can be reproduced with `python scripts/evaluate_all.py`.

---

## 1. What the project does, in one paragraph

Rad-Intel takes a frontal chest X-ray and returns three things: a
classification (normal or pneumonia) with a calibrated probability, a visual
explanation showing which part of the lung drove that decision, and a
structured radiology-style report written from those outputs. The three
stages are deliberately separate, so every sentence in the report can be
traced back to a number the model actually produced. It is built for
research and teaching, not diagnosis.

---

## 2. The pipeline, stage by stage

**Stage 1 — Preprocessing.** The image is converted to three channels,
contrast-enhanced with CLAHE, resized to 224×224 and normalised with
ImageNet statistics. CLAHE matters because chest radiographs vary a lot in
exposure; it equalises contrast locally so faint opacities stay visible.

**Stage 2 — Classification.** Five architectures are supported. Three are
trained: DenseNet121, Swin-T and ResNet50. Two are defined but untrained:
the proposed DenseNet+Swin+CBAM hybrid and its no-CBAM ablation.

**Stage 3 — Explainability.** Grad-CAM produces a saliency map from the
final feature tensor. The map is restricted to the lung fields, which are
segmented by a pretrained anatomical network, then reduced to four
radiographic quadrants to give a dominant zone and a left/right balance.

**Stage 4 — Reporting.** The classification and the quadrant statistics are
serialised into a fixed evidence object and sent to Gemini 3.5 Flash-Lite,
which returns an ACR-style report. The language model never sees the image.
If the service is unavailable, a deterministic template renders the same
fields.

**Stage 5 — Serving.** FastAPI exposes each stage as its own endpoint plus a
combined `/analyze`. Results persist to SQLite. A browser UI drives the whole
thing.

---

## 3. Technology used, and why each

| Layer | Choice | Why this one |
|---|---|---|
| Language | Python 3.14 | Ecosystem; everything below assumes it |
| DL framework | PyTorch 2.14 | Pretrained DenseNet/Swin/ResNet available via torchvision |
| Backbones | torchvision | One source for all three, consistent weight format |
| Attention | CBAM, hand-written | Not available pretrained; it is the proposed contribution |
| Explainability | pytorch-grad-cam, LIME | Reference implementations of the two cited methods |
| Lung segmentation | TorchXRayVision PSPNet | Anatomically trained; replaced two heuristics that failed |
| LLM | Gemini 3.5 Flash-Lite | Free tier, fast enough per request, structured output |
| API | FastAPI | Async, automatic OpenAPI docs, Pydantic validation |
| Storage | SQLite + aiosqlite | File-based, no server, sufficient for an audit trail |
| PDF | ReportLab | Self-contained; no LaTeX install needed on the host |
| Training | Google Colab + Kaggle (T4) | Free GPU; two lanes in parallel |
| Testing | pytest, 26 tests | Unit plus API contract regression |

---

## 4. Results

Protocol: official Kaggle split. `train` + `val` merged and re-split 85/15
stratified at seed 42 (4,447 train / 785 validation). The official 624-image
`test` folder is untouched. Two-phase training: 8 epochs with backbones
frozen, then up to 12 epochs end-to-end at discriminative learning rates
(backbones 1e-5, head 1e-4). Focal loss, inverse-frequency class weights,
early stopping on validation AUROC.

| Model | Accuracy | AUROC | AUPRC | Sensitivity | Specificity | F1 | ECE |
|---|---|---|---|---|---|---|---|
| DenseNet121 | **91.5%** | 0.962 | 0.969 | 95.6% | **84.6%** | **0.934** | **0.052** |
| Swin-T | 90.1% | **0.975** | **0.983** | **97.4%** | 77.8% | 0.925 | 0.054 |
| ResNet50 | 89.7% | 0.954 | 0.966 | 93.3% | 83.8% | 0.919 | 0.075 |

Each model uses its own threshold, selected on validation data: 0.463,
0.422 and 0.557 respectively.

### The finding that matters most

Paired tests on the same 624 images:

- **No pair differs significantly in accuracy.** McNemar, Holm-adjusted:
  all p > 0.53. The 1.8-point spread is sampling noise.
- **Swin-T ranks significantly better.** ΔAUROC +0.013 over DenseNet121 and
  +0.021 over ResNet50, bootstrap intervals excluding zero.

So the model that discriminates best has the worst specificity. Ranking
quality and decision quality are different things, and the accuracy column
is not a leaderboard. This is the single most defensible claim in the
project, because it is the one backed by a significance test.

---

## 5. Problems encountered and how each was resolved

These are worth presenting. Most of the technical depth is here.

**1. Predictions were at chance level (~51%).** The config requested the
hybrid architecture, but the only trained checkpoint was a DenseNet121. The
model factory silently discarded a mismatched checkpoint and returned an
untrained network, which still emitted confident-looking probabilities.
*Fixed:* the loader now warns loudly, and the API refuses the request with
HTTP 503 rather than serving an untrained model.

**2. The calibrated threshold was never applied.** Training selected an
operating point by Youden's index and stored it. The evaluation script read
it; the serving code used plain argmax (0.5). Reported metrics and deployed
behaviour described different classifiers, and the UI claimed "calibrated
threshold applied" while it wasn't. *Fixed:* the threshold is read from the
checkpoint at load time. A regression test now asserts the two agree.

**3. The Grad-CAM image never rendered.** The backend returned a complete
data URL; the frontend prepended the prefix a second time. *Fixed* in one
line, with tolerance for both forms.

**4. Swin-T's explanations were pointing the wrong way.** The Swin backbone
emits channels-last activations; the attribution library assumes
channels-first, so it read the 768 channels as image width. Correlation with
an occlusion reference was **−0.12** — the maps highlighted regions the model
demonstrably did not use, while looking entirely plausible. *Fixed* with a
layout adapter: correlation moved to **+0.31**.

**5. Grad-CAM target layers were suboptimal.** DenseNet121 was attributing to
a 32-channel convolution inside the final dense layer rather than the
1024-channel block output the classifier pools. ResNet50 targeted a
convolution whose output had yet to pass through its residual connection.
*Fixed:* deletion AUC improved 0.723 → 0.683; ResNet agreement 0.23 → 0.31.

**6. Saliency kept landing outside the lungs.** Three attempts:
  - A body-silhouette mask kept shoulders, clavicles and neck. It was also
    broken outright, because CLAHE destroys the global intensity structure
    Otsu depends on.
  - An intensity rule ("lung is darker") failed on precisely the cases that
    matter, since consolidation is radiopaque — it carved out the pathology.
  - *Resolved* with a pretrained anatomical segmentation network. Lung
    fields are a distinct class from heart, mediastinum and diaphragm. Peaks
    inside the lung field went from 3/6 to 18/18 across all three models.

**7. Reports asserted findings the model could not support.** The template
claimed "focal airspace consolidation is prominently identified" and
recommended initiating antibiotics — in identical wording at 71% and 99%
confidence. *Fixed:* certainty language is tiered by confidence, the finding
is attributed to the model's region of maximal response rather than to
observed consolidation, and the treatment directive was removed.

**8. The training protocol inverted the ablation.** The trainer only ever
trained the classifier head on frozen cached features. Under that regime the
hybrid scored *below* its own baselines and CBAM appeared to *hurt*, because
the metric was tracking head capacity: DenseNet's head has ~2K parameters
and cannot overfit, the hybrid's has ~950K and reached 100% training
accuracy. *Fixed* by implementing the two-phase protocol the paper already
specified. The hybrid went from 82.2% to 92.0%, specificity 54.7% → 87.2%.

**9. API errors were misreported.** Routes raised a correct HTTP 400 for bad
input, then caught their own exception in a bare `except` and re-raised it as
500. *Fixed,* with regression tests.

---

## 6. What is done

- Three models trained and fully integrated, reproducible end to end
- Calibrated thresholds applied in serving and verified by test
- Grad-CAM corrected and validated against occlusion sensitivity
- Lung-field segmentation for all models
- Live Gemini reporting, with deterministic fallback
- All nine API endpoints working, error contract correct
- SQLite audit trail
- 26 automated tests passing
- Paper: Tables IV and V populated, methodology matched to the code

## 7. What remains

| Item | Blocker |
|---|---|
| Hybrid + no-CBAM training | GPU time; runs interrupted twice |
| Figures 2–4 | Need a hybrid checkpoint; ROC/reliability CSVs already generated |
| Clinical review | Needs a clinician's time |
| Multi-seed runs | ~5× current GPU budget |
| LIME quantitative metrics | Slow; qualitative output works |

The honest framing: the three baselines are complete and statistically
analysed. The proposed architecture has a measured result (92.0% accuracy,
AUROC 0.970) but its checkpoint was lost to a session timeout, so it is not
yet in the integrated system.

---

## 8. Likely panel questions

**"Why is your accuracy only 91%? Published work reports 98%."**
Those numbers usually come from a random split of the training folder, where
images from the same patient appear in both halves. We evaluate on the
official held-out test folder, which is a different distribution — our own
validation accuracy is 95–98%, and it drops to ~90% on the test set. That
gap is the honest measure. A 98% figure on a leaked split is not comparable.

**"Which model is best?"**
They are statistically indistinguishable on accuracy — McNemar gives p > 0.53
for every pair. Swin-T ranks significantly better by AUROC but has the worst
specificity, so it over-calls pneumonia. DenseNet121 is the best-balanced and
is the default.

**"Why does the heatmap sometimes sit outside the lungs?"**
It did, and we measured it: 21–48% of saliency mass fell outside the lung
fields. We verified with occlusion sensitivity that Grad-CAM was reporting
this faithfully, so it is the model attending to framing artefacts, not a bug
in the explanation. This is known shortcut learning on this dataset. We now
restrict the displayed explanation to segmented lung fields and still report
the off-lung fraction, so the behaviour stays visible rather than hidden.

**"Why Grad-CAM and not something newer?"**
It is the standard gradient attribution for CNNs and is cited in the papers
we compare against. More importantly we validated it rather than trusting
it — that validation is what caught the Swin layout bug.

**"Why is the threshold not 0.5?"**
Because the training data is imbalanced 2.9:1 toward pneumonia, so the model's
probabilities skew high. Youden's index selects the point that balances
sensitivity and specificity on validation data. Using 0.5 would over-call
pneumonia on every borderline film.

**"Does the LLM diagnose the patient?"**
No. It never sees the image. It receives a fixed set of numbers — predicted
class, probability, threshold, quadrant scores — and writes them up. Any
factual error has to originate in those numbers or in a schema violation, and
both are machine-checkable.

**"How do you stop the LLM hallucinating findings?"**
Three ways: it has no access to the image, so it cannot invent what it cannot
see; the certainty language is tiered by confidence; and a validator rejects
anatomy or numbers absent from the evidence object. We also removed a
treatment recommendation the earlier template was emitting, because an image
classifier has no basis to advise on antibiotics.

**"Why does it miss some pneumonia cases?"**
Sensitivity is 93–97% depending on the model, so 10–26 of 390 test cases are
missed. That is the cost of maintaining specificity. A screening deployment
would lower the threshold to trade specificity for sensitivity; the system
reports probabilities, so that choice is available.

**"Is this usable in a hospital?"**
No, and it is not presented as one. Single institution, pediatric only, no
patient-level split guarantee, no external validation, no prospective study.
It is a research and teaching system, and the interface states that.

**"What would you do with more time?"**
Finish the hybrid and the ablation, run multiple seeds for confidence
intervals, add lung-field cropping during *training* rather than only at
explanation time to attack the shortcut learning at its source, and validate
on a second dataset from a different institution.

**"What was the hardest problem?"**
Discovering that the training protocol itself was producing the wrong answer.
The ablation said CBAM hurt performance. The code ran without error and the
numbers looked reasonable. Only when we noticed accuracy tracking head size
inversely did we find that freezing both backbones was measuring head
capacity rather than architecture. It is a reminder that a result can be
precisely wrong.
