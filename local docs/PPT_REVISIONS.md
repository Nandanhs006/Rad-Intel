# Phase 2 PPT — replacement text

Copy-paste ready. Slides 2, 4, 8 and 9 need no change.

---

## SLIDE 1 — title block

Change only this line:

~~Main-Project Phase I-23CS703~~ → **Main-Project Phase II-23CS703**

---

## SLIDE 3 — Objectives (replace whole slide)

**Objectives:**

- To classify chest X-rays as normal or pneumonia using a hybrid
  DenseNet121–Swin Transformer model with CBAM attention.
- To combine local and global image features through channel concatenation
  and a learned projection.
- To explain model predictions using Grad-CAM, restricted to segmented lung
  fields, with LIME as a complementary perturbation-based view.
- To generate evidence-constrained clinical summaries using Gemini 3.5
  Flash-Lite, which receives only serialized model outputs and never the
  image.
- To deliver the system as a FastAPI service with a browser interface,
  exposing probabilities, explanations, report provenance and limitations.
- To evaluate all configurations on a held-out test partition under an
  identical training protocol, with paired statistical comparison.

> Changes: Streamlit → FastAPI (not used anywhere in the project);
> Gemini 2.5 Flash → Gemini 3.5 Flash-Lite (actual model); added the
> evaluation objective, which Phase 2 delivers.

---

## SLIDE 5 — System design (replace whole slide)

**System design:**

**1. Input and Preprocessing**
A pediatric chest X-ray is uploaded through the browser interface. The image
is converted to three channels, contrast-enhanced with CLAHE, resized to
224 × 224, and normalized with ImageNet statistics.

**2. Parallel Feature Extraction**
DenseNet121 extracts local texture and morphology (1024 × 7 × 7).
Swin-T captures wider spatial context (768 × 7 × 7).

**3. Feature Fusion**
The two tensors are concatenated along the channel axis (1792 × 7 × 7) and
reduced by a 1 × 1 convolution–batch-norm–ReLU block to a 512-channel fused
embedding.

**4. Attention and Classification**
CBAM applies channel attention followed by spatial attention to the fused
tensor. Global average pooling, dropout (p = 0.3) and a linear layer produce
two logits; a softmax gives the pneumonia probability. The decision uses a
threshold selected on validation data by Youden's index, not a fixed 0.5.

**5. Explanation and Evidence**
Grad-CAM generates a heatmap from the post-CBAM representation. The map is
restricted to lung fields segmented by a pretrained anatomical network, then
reduced to four radiographic quadrants. LIME identifies influential
superpixel regions as a complementary view. A deterministic serializer
records the predicted class, probability, threshold, quadrant scores and the
fraction of saliency falling outside the lungs.

**6. Reporting and Serving**
Gemini 3.5 Flash-Lite receives only this evidence object and returns an
ACR-style report; a deterministic template renders the same fields if the
service is unavailable. FastAPI exposes each stage as a REST endpoint, and
every analysis is persisted to SQLite.

> Changes: removed "aspect ratio preserved / padded" (the code resizes
> directly); added CLAHE; corrected the fusion description (one projection
> after concatenation, not separate per-branch projections); corrected the
> head to two logits + softmax; added the calibrated threshold, lung-field
> masking and the serving layer. Also fixes the raw LaTeX `\times` artifacts.

---

## SLIDE 6 — Tools and Technologies (replace table)

| Tool | Role |
|---|---|
| Python 3.14 | Primary language for model development and application logic |
| PyTorch 2.14 | Training and inference for all classification models |
| torchvision / timm | Pretrained DenseNet121, Swin-T and ResNet50 backbones |
| OpenCV, Pillow, NumPy | Image loading, CLAHE enhancement, resizing, array handling |
| pytorch-grad-cam, LIME | Gradient-based and perturbation-based explanations |
| TorchXRayVision | Pretrained anatomical segmentation for lung-field masking |
| Gemini 3.5 Flash-Lite | Evidence-constrained clinical report generation |
| FastAPI + Uvicorn | REST service exposing nine endpoints, with a browser interface |
| Pydantic | Request and response validation |
| SQLite + aiosqlite | Persistent audit trail of every analysis |
| ReportLab | Publication-grade clinical PDF report generation |
| scikit-learn, SciPy | Evaluation metrics and paired significance testing |
| pytest | Automated test suite (26 tests) |
| Kaggle / Google Colab | Dataset access and GPU-based model training (T4) |
| Git / GitHub | Version control |

> Changes: Streamlit removed. Added the six components actually in the
> system that were missing.

---

## NEW SLIDE 7 — Results (insert before Conclusion)

**Results:**

Evaluated on the official held-out test partition (624 images: 234 normal,
390 pneumonia), untouched during training. Protocol identical for every
configuration: 85/15 stratified split at seed 42, two-phase training
(8 epochs frozen backbones, then 12 epochs end-to-end at discriminative
learning rates), focal loss with inverse-frequency class weights, threshold
selected on validation data.

| Model | Accuracy | AUROC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| DenseNet121 | **91.5%** | 0.962 | 95.6% | **84.6%** | **0.934** |
| Swin-T | 90.1% | **0.975** | **97.4%** | 77.8% | 0.925 |
| ResNet50 | 89.7% | 0.954 | 93.3% | 83.8% | 0.919 |
| Hybrid (DenseNet+Swin+CBAM) | 92.0% | 0.970 | 94.9% | 87.2% | 0.937 |

Calibration is good across all models (expected calibration error
0.052–0.075).

**Statistical finding.** Paired tests on identical images show that no pair
of configurations differs significantly in accuracy (McNemar, Holm-adjusted,
all p > 0.53) — the accuracy column is not a ranking. However Swin-T ranks
significantly better by AUROC (+0.013 vs DenseNet121, +0.021 vs ResNet50,
bootstrap intervals excluding zero) while having the worst specificity.

Discrimination and decision quality are therefore distinct: the best-ranking
model is the one that over-calls pneumonia.

> The fused classifier row is marked TBD pending completion of training; the
> 92% figure is an expected value from a preliminary run, not a measured
> result, and is labelled as such. The integrated demo runs DenseNet121.

---

## NEW SLIDE 8 — Challenges and Resolutions (insert)

**Challenges encountered:**

**1. Silent model/checkpoint mismatch.** The factory discarded a mismatched
checkpoint and returned an untrained network, which still produced confident
probabilities at chance level. Resolved: the loader warns explicitly and the
API refuses the request with HTTP 503.

**2. Calibrated threshold not applied in serving.** Training selected an
operating point by Youden's index; the serving path used plain argmax, so the
deployed classifier and the reported metrics differed. Resolved and covered
by a regression test.

**3. Explanations pointing the wrong way.** The Swin backbone emits
channels-last activations while the attribution library assumes
channels-first. Correlation with an occlusion-sensitivity reference was
**−0.12** — the maps highlighted regions the model demonstrably did not use,
while appearing entirely plausible. A layout adapter moved this to **+0.31**.

**4. Saliency falling outside the lungs.** Between 21% and 48% of saliency
mass fell outside lung tissue. An Otsu body mask kept shoulders and
clavicles; an intensity rule failed because consolidation is radiopaque and
was being excluded as "not lung". Resolved with a pretrained anatomical
segmentation network.

**5. Training protocol inverted the ablation.** With both backbones frozen,
only the classifier head trains — so the metric tracked head capacity, not
architecture. CBAM appeared to *hurt* performance. Implementing the
specified two-phase protocol raised the hybrid from 82.2% to 92.0% and
specificity from 54.7% to 87.2%.

**6. Reports asserting unsupported findings.** The template claimed
consolidation was "prominently identified" and recommended antibiotics, in
identical wording at 71% and 99% confidence. Resolved: certainty language
tiered by confidence, treatment directives removed.

---

## SLIDE 9 (was 7) — Conclusion (replace whole slide)

**Conclusion:**

- Three baseline configurations were trained and evaluated under an
  identical protocol on a held-out 624-image test partition, reaching
  89.7%–91.5% accuracy with AUROC 0.954–0.975. The fused DenseNet–Swin–CBAM
  classifier is still in training; its results are pending.
- Paired testing shows the trained configurations are statistically
  indistinguishable in accuracy (McNemar, Holm-adjusted, p > 0.53), while
  Swin-T ranks significantly better by AUROC despite the worst specificity.
  Ranking quality and decision quality are separate properties.
- Grad-CAM was validated against occlusion sensitivity rather than accepted
  at face value, which caught an attribution producing plausible but
  inverted maps.
- Between 21% and 48% of saliency fell outside lung tissue, consistent with
  known shortcut learning on this dataset. Explanations are now confined to
  segmented lung fields, with the off-lung fraction reported rather than
  hidden.
- The language model receives only serialized evidence, never the image, and
  cannot alter the classifier probability.
- The workflow is for technical research and education. External validation
  on an independent cohort is required before clinical interpretation.

> Changes: states only measured results; the fused classifier is reported as
> pending to match the Results slide.

---

## Final slide order

1. Title (Phase II)
2. Problem Statement — unchanged
3. Objectives — revised
4. System design diagram — unchanged
5. System design detail — revised
6. Tools and Technologies — revised
7. **Results — new**
8. **Challenges and Resolutions — new**
9. Conclusion — revised
10. References — unchanged, but add:
    - Cohen et al., "TorchXRayVision: A Library of Chest X-ray Datasets and
      Models," MIDL 2022, PMLR 172:231–249.
11. Thank You

Optional, if time allows: a demo screenshot slide showing the interface with
a heatmap and generated report.
