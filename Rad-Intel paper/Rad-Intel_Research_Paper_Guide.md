# Research Paper Development Guide — Rad-Intel
### A Data-Driven Clinical Decision Support Framework for Automated Pneumonia Detection

This document is a complete content package for writing an IEEE BIBE / BIBM-quality research paper on this project. No experimental results exist yet, so this guide covers every section that can be written now, flags every section that depends on results, and lays out exactly what experiments need to be run to fill those gaps. It draws on the Synopsis, Objectives, Phase 1 report, and the accompanying paper already produced for the project, and extends them with material those documents are missing (particularly literature on LLM-based radiology reporting, and the formal methodology needed for a conference submission).

---

## 1. Target Venue and Formatting Requirements

**Target conferences:** IEEE BIBE (International Conference on Bioinformatics and Bioengineering) and IEEE BIBM (International Conference on Bioinformatics and Biomedicine). Both are IEEE Computer Society conferences, indexed in IEEE Xplore, and share the same broad formatting conventions.

Known, verified requirements (confirm exact numbers against the specific year's Call for Papers, since limits shift year to year):

- **Layout:** IEEE two-column conference format (IEEEtran template, LaTeX or Word).
- **Review type:** BIBM uses double-blind review — no author names or affiliations anywhere in the submitted PDF.
- **Length:** BIBM/BIBE regular papers are typically capped at 6-8 pages including references and figures; short/workshop papers are shorter (BIBM's 2026 Doctoral Forum, for comparison, caps at 6 pages excluding references). Confirm the exact cap on the current CFP before drafting — this determines how much of section 3 below you can actually fit in.
- **Abstract:** 150-250 words, no citations, no undefined abbreviations.
- **Index terms:** 3-6 keywords, comma-separated, placed directly under the abstract.
- **References:** IEEE numbered style, cited in order of first appearance, matched to the reference list by bracketed number (e.g., `[1]`).
- **File format:** PDF only, checked through IEEE PDF eXpress before final submission.
- **Figures:** must be legible in print, at minimum 300 DPI, referenced in text before they appear.
- **Ethics/data statement:** since this work uses medical imaging data, include a one-line statement confirming the dataset is public, de-identified, and used under its original license (this is standard practice for BIBE/BIBM submissions using clinical data and is often explicitly requested).

---

## 2. Candidate Paper Titles

Pick one, or use as a base to iterate on once results exist (titles at this venue tend to name the mechanism, not just the task):

- "A Hybrid DenseNet-Swin Transformer Framework with CBAM Attention for Explainable Pneumonia Detection and LLM-Based Report Generation from Chest X-Rays"
- "Rad-Intel: An Explainable Deep Learning Framework Integrating Attention Mechanisms and Large Language Models for Automated Pneumonia Diagnosis"
- "Toward Trustworthy Chest X-Ray Diagnosis: Combining Attention-Guided CNN-Transformer Fusion with LLM-Generated Clinical Reports"

Avoid vague titles ("A Deep Learning Approach to Pneumonia Detection") — BIBE/BIBM reviewers expect the title to signal the specific technical contribution.

---

## 3. Full Section-by-Section Content Plan

### 3.1 Abstract (template)

A BIBE/BIBM abstract follows a fixed rhetorical shape: problem, gap, method, result, significance — each in one to two sentences. Template:

> Pneumonia remains a leading cause of morbidity and mortality worldwide, and manual chest X-ray interpretation is time-consuming and subject to inter-observer variability. [existing CNN/Transformer methods and their limitation, e.g., single-scale feature extraction, lack of interpretability, or purely categorical output that omits a written diagnostic rationale]. This paper proposes Rad-Intel, a framework that fuses DenseNet-based local feature extraction with Swin Transformer-based global context modeling, refined through a CBAM attention module, to classify chest X-rays as normal or pneumonia-affected. Model predictions are made interpretable through Grad-CAM and LIME, and are further translated into structured, human-readable clinical reports using a large language model (Gemini API). On the [dataset name] dataset, the proposed framework achieves [accuracy]% accuracy, [F1]% F1-score, and an AUC of [value], outperforming baseline DenseNet and Swin Transformer models trained independently by [X] and [Y] percentage points respectively. [Closing sentence on clinical significance / novelty of combining XAI with automated reporting].

Everything in brackets needs to be filled in after experiments (Section 5 gives the exact experiment plan).

### 3.2 Index Terms

Suggested: `Pneumonia detection, Chest X-ray, DenseNet, Swin Transformer, CBAM, Explainable AI, Grad-CAM, Large Language Models, Clinical Decision Support`

### 3.3 I. Introduction

Structure (roughly 4-6 paragraphs):

1. **Clinical motivation** — burden of pneumonia, reliance on chest radiography, radiologist shortage/workload, value of AI-assisted triage. This material already exists in the Synopsis and can be expanded with recent statistics (WHO/global pneumonia burden — verify current figures via a fresh search rather than reusing old ones, since epidemiological statistics date quickly).
2. **Limitations of existing automated approaches** — pure CNNs miss global context; pure Transformers need large data and compute; most published systems output a label with no explanation, which limits clinical trust and adoption. This is the "gap" paragraph and is the most important one for a BIBE/BIBM reviewer.
3. **What interpretability and reporting add** — explain why a black-box label is insufficient for a clinical decision support tool, and how Grad-CAM/LIME plus an LLM-generated narrative report close that gap.
4. **Contributions** — end the introduction with an explicit, numbered list. Reviewers scan for this. Example:
   - A hybrid DenseNet-Swin Transformer architecture with CBAM attention for pneumonia classification from chest X-rays.
   - An integrated explainability pipeline combining Grad-CAM and LIME for visual justification of predictions.
   - An LLM-based reporting module (Gemini API) that converts model output and saliency information into a structured clinical report.
   - A quantitative and qualitative evaluation against baseline architectures and ablated variants of the proposed model.
5. **Paper organization** — one sentence per remaining section.

### 3.4 II. Related Work

Organize by subsection rather than a flat list — this is what distinguishes a BIBE/BIBM-quality related work section from a general survey paragraph. Suggested subsections, mapped to what the Phase 1 report's literature survey already covers and where it needs extension:

**A. CNN-based approaches for pneumonia/chest X-ray classification** — the report's existing survey table already covers several CNN-based studies; keep these but re-verify each citation's exact venue/year before reuse.

**B. Transformer and hybrid CNN-Transformer architectures for medical imaging** — the report covers Swin Transformer-related work; expand slightly with general ViT-in-medical-imaging context.

**C. Attention mechanisms (SE, CBAM) in medical image analysis** — the report references CBAM's use elsewhere; add the original CBAM citation (Section 7 below).

**D. Explainable AI for medical imaging (Grad-CAM, LIME, and successors)** — the report discusses these techniques but the current reference list does not appear to cite the original Grad-CAM and LIME papers directly. This is a gap to fix — a reviewer will expect the foundational method papers cited, not just applications of them.

**E. LLM-based and vision-language radiology report generation** — this is the biggest content gap. Neither the Synopsis, the report, nor the paper currently surveys this literature, even though automated report generation is one of the project's stated contributions. Section 7 below lists real, verified candidate papers for this subsection (MIMIC-CXR, IHRAS, RaDialog, Gemini-in-medicine). Without this subsection, a BIBE/BIBM reviewer is very likely to flag "insufficient related work" on the paper's core novelty claim.

**F. Summary table** — end with a comparison table (method, architecture, dataset, XAI used, reporting capability, accuracy) positioning Rad-Intel's gap: most prior work does either classification+XAI or report generation, rarely both in one pipeline with the specific DenseNet+Swin+CBAM combination.

### 3.5 III. Materials and Methods

This is the section that needs the most new writing beyond what the report already contains, since the report describes the model conceptually but not at implementation/formal specification level.

**A. Dataset**
State explicitly and precisely (fill in once the dataset is finalized):
- Dataset name and source (e.g., Kaggle "Chest X-Ray Images (Pneumonia)" — Kermany et al. — or NIH ChestX-ray14, or a combination).
- Total image count, class distribution (normal vs. pneumonia, and bacterial/viral split if used), image resolution and format.
- Train/validation/test split ratios and whether the split is patient-wise (important to state explicitly — patient-level leakage between train/test is a known issue in public chest X-ray datasets and reviewers may ask about it directly).
- Any class imbalance and how it is handled (class weighting, oversampling, focal loss).

**B. Preprocessing**
Resizing dimensions, normalization (mean/std used, ImageNet stats if using pretrained weights), augmentation techniques applied (rotation, flip, contrast jitter — state ranges), and justification for each (e.g., avoid vertical flips for chest X-rays since anatomical orientation is clinically meaningful).

**C. Proposed Architecture**
Formalize what the report describes conceptually. Structure as:
- Local branch: DenseNet (specify variant, e.g., DenseNet121), pretrained on ImageNet, feature map extracted before the classification head.
- Global branch: Swin Transformer (specify variant, e.g., Swin-T), pretrained, feature map extracted similarly.
- Fusion strategy: specify exactly how the two feature maps are combined — concatenation followed by a projection layer, or cross-attention. This needs to be a concrete, decided design choice, not left implicit, since it is the paper's central architectural contribution.
- CBAM: applied to the fused feature map. Formalize using the standard CBAM formulation:

  Given an intermediate feature map **F** ∈ R^(C×H×W), CBAM sequentially infers a channel attention map **M_c** and a spatial attention map **M_s**:

  ```
  F'  = M_c(F) ⊗ F
  F'' = M_s(F') ⊗ F'
  ```

  where ⊗ denotes element-wise multiplication. Channel attention is computed from both average-pooled and max-pooled descriptors passed through a shared MLP; spatial attention is computed from a channel-pooled descriptor passed through a 7×7 convolution. (Cite Woo et al., 2018 — full reference in Section 7.)

- Classification head: pooling + fully connected layer(s) + softmax/sigmoid for binary classification, with dropout rate specified.
- Include an architecture diagram (block diagram showing both branches, fusion point, CBAM, classifier head, and the downstream XAI + LLM modules) — this single figure is typically the most-referenced figure in the whole paper.

**D. Explainability Module**
- Grad-CAM: describe how the class-discriminative localization map is computed from gradients flowing into the final convolutional layer, and specify exactly which layer is used (this detail matters for reproducibility).
- LIME: describe the perturbation-based local surrogate model approach as applied to image superpixels, and state number of samples/superpixels used.
- State how these two are used together — e.g., Grad-CAM for a fast heatmap shown to the user, LIME as a secondary explanation for validation/comparison.

**E. LLM-Based Clinical Reporting Module**
This is the part the current documents describe least precisely, and one of the most reviewer-interesting parts of the paper. Specify:
- What structured input is passed to Gemini (e.g., predicted class, confidence score, a textual description of Grad-CAM-highlighted lung regions, patient metadata if available).
- The prompt design/template used, and whether it is a single-shot prompt or a structured system+user prompt with output formatting constraints (e.g., forcing a specific report structure: Findings / Impression / Recommendation, mirroring real radiology report structure).
- Which Gemini model is used (Gemini 2.5 Flash, per the tech stack decision) and why.
- Safeguards against hallucination — e.g., explicitly constraining the LLM to only reference findings supported by the model's own prediction and saliency map, not to introduce unsupported clinical claims.
- A worked example: one sample input (image + prediction + heatmap description) and the resulting generated report, shown as a figure or listing.

**F. Training Configuration**
Provide a hyperparameters table once finalized:

| Hyperparameter | Value |
|---|---|
| Optimizer | e.g., AdamW |
| Learning rate | e.g., 1e-4 with cosine decay |
| Batch size | e.g., 32 |
| Epochs | e.g., 50, with early stopping |
| Loss function | Binary cross-entropy or focal loss (state which, and why if class-imbalanced) |
| Hardware | GPU used (e.g., Colab T4) |
| Framework | PyTorch version, key library versions |

**G. Evaluation Protocol**
State every metric formally with its formula, since BIBE/BIBM reviewers expect precision here:
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, AUC-ROC, and the confusion matrix.
- For the reporting module: since there is no single agreed metric for LLM-generated clinical report quality, specify a combination — e.g., ROUGE-L/BLEU against reference reports if ground-truth reports are available in the dataset (MIMIC-CXR has these; the Kaggle Kermany dataset does not), plus a qualitative rubric-based expert or self-assessment (accuracy, completeness, absence of hallucinated findings) if no reference reports exist.

### 3.6 IV. Experimental Results (currently empty — see Section 5 for how to generate this)

Plan the subsections now so results can be dropped straight in once available:

- **A. Implementation details** — restate hardware/software briefly.
- **B. Quantitative classification performance** — main results table: proposed model vs. baselines (plain DenseNet121, plain Swin-T, ResNet50, VGG16, EfficientNet-B0), each with Accuracy/Precision/Recall/F1/AUC. Include the confusion matrix and ROC curve as figures.
- **C. Ablation study** — a table isolating the contribution of each component: DenseNet-only, Swin-only, DenseNet+Swin (no CBAM), full model (DenseNet+Swin+CBAM). This is one of the most important tables for a BIBE/BIBM audience — it directly substantiates the paper's architectural claims.
- **D. Comparison with prior published work** — reproduce the comparison table already present in the Phase 1 report's literature survey, adding a final row for this work's own results, on the same or a comparable dataset.
- **E. Statistical significance** — report results as mean ± standard deviation across k-fold cross-validation or multiple random seeds, and apply a paired significance test (e.g., McNemar's test between the proposed model and the strongest baseline) rather than reporting a single run.
- **F. Qualitative explainability results** — sample chest X-rays with Grad-CAM and LIME overlays, including at least one correctly classified and one misclassified example (showing failure cases builds credibility with reviewers).
- **G. LLM report generation quality** — sample generated reports next to the corresponding prediction/heatmap, plus whatever quantitative or rubric-based evaluation was defined in 3.5G.
- **H. Inference time / efficiency** — end-to-end latency (image in to report out), useful for a "clinical decision support" framing since deployability matters at this venue.

### 3.7 V. Discussion

Write once results exist. Cover: what the ablation results imply about each architectural component's contribution; how the framework's explainability compares qualitatively to prior XAI-only work; where the LLM reporting module adds value beyond a bare label; failure modes and likely causes (e.g., dataset bias, ambiguous X-rays, LLM hallucination risk); and limitations (single dataset, binary rather than multi-class diagnosis, no external/multi-institution validation, no radiologist user study yet).

### 3.8 VI. Conclusion and Future Work

Summarize contributions in 3-4 sentences (mirror the introduction's contribution list, do not introduce new claims here). Future work: multi-class disease classification beyond binary normal/pneumonia, external validation on a second dataset, a prospective radiologist evaluation of report quality and trust, and extending the reporting module to a conversational/query interface (this last point ties directly back to the project's fourth objective on LLM-based query support).

### 3.9 References

IEEE numbered format, cited in order of appearance. Keep the existing literature-survey citations from the Phase 1 report (re-verify each one's exact bibliographic details before reuse — do not assume the report's formatting is submission-ready), and add the foundational method citations and LLM-reporting citations listed in Section 7 below, which are currently missing.

---

## 4. Figures and Tables Checklist

- Fig. 1 — Overall system architecture / block diagram (data flow from X-ray input through classification, XAI, and LLM report output).
- Fig. 2 — CBAM module internal diagram (channel + spatial attention).
- Fig. 3 — Sample dataset images (normal vs. pneumonia) with class distribution bar chart.
- Fig. 4 — Training/validation loss and accuracy curves.
- Fig. 5 — Confusion matrix.
- Fig. 6 — ROC curve (proposed model vs. baselines, overlaid).
- Fig. 7 — Grad-CAM / LIME visualizations, correct and misclassified examples side by side.
- Fig. 8 — Example end-to-end output: input X-ray, prediction, heatmap, and generated LLM report shown together.
- Table 1 — Related work comparison table.
- Table 2 — Dataset statistics (class counts, split sizes).
- Table 3 — Hyperparameters.
- Table 4 — Main results (proposed vs. baselines).
- Table 5 — Ablation study.
- Table 6 — Comparison with prior published results.

---

## 5. Experiment Plan to Generate the Missing Results

Concrete sequence to go from "no results" to a fillable Section IV:

1. **Finalize and document the dataset** — download, verify licensing, compute class distribution, define a patient-wise train/val/test split.
2. **Reproduce baselines first** — train plain DenseNet121, plain Swin-T, ResNet50, VGG16, EfficientNet-B0 independently, using identical preprocessing/splits, to establish a fair comparison baseline before touching the proposed fusion model.
3. **Build and train the proposed fusion model** (DenseNet + Swin Transformer + CBAM), using the same protocol.
4. **Run the ablation variants** — DenseNet-only, Swin-only, fusion without CBAM, full model — all under identical training settings.
5. **Repeat training across at least 3-5 seeds or k-folds** for every model in steps 2-4, so results can be reported as mean ± std rather than a single number, and run a significance test between the proposed model and the best baseline.
6. **Generate Grad-CAM and LIME outputs** on a representative sample of test images, including at least one clear failure case.
7. **Wire up the Gemini API reporting module** end-to-end, and generate reports for the same sample set used in step 6.
8. **Evaluate report quality** using whichever protocol was fixed in Section 3.5G (reference-based metric if reference reports exist, rubric-based otherwise).
9. **Measure end-to-end inference latency** on the deployment hardware actually used.
10. **Assemble all tables/figures listed in Section 4** directly from the logs/outputs of steps 2-9.

---

## 6. Writing Style Guidelines for BIBE/BIBM-Level Quality

- Write in third person, past tense for methodology and results ("The model was trained…" not "We train the model…" — either convention is accepted, but stay consistent throughout).
- Every claim of superiority must be backed by a number in a table, not asserted in prose alone.
- Avoid marketing language ("cutting-edge," "revolutionary," "state-of-the-art" without a citation backing the comparison). Reviewers at this venue are domain experts and this language reads as unsubstantiated.
- Define every abbreviation on first use, even common ones in this specific paper (CBAM, XAI, Grad-CAM, LIME, LLM).
- Keep the Introduction's contribution list and the Conclusion's summary consistent in wording — reviewers cross-check these.
- Quantify limitations honestly in the Discussion; a paper that names its own limitations reads as more credible to this audience than one that omits them.
- Every figure and table must be referenced by number in the text before or where it appears ("as shown in Fig. 3" / "Table 2 summarizes...").

---

## 7. Verified Reference Candidates (fill the current gaps)

The existing literature survey (in the Phase 1 report) covers CNN/Transformer-based pneumonia classification but is missing citations for the foundational explainability and attention techniques the framework itself uses, and for LLM-based radiology reporting, which is central to the project's novelty claim. The following are real, verifiable candidates — confirm exact page numbers/volume details against the publisher page before final submission.

**Foundational method papers (currently uncited but directly used by the framework):**

- S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," in *Proc. European Conference on Computer Vision (ECCV)*, Munich, Germany, 2018, pp. 3-19.
- R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618-626.
- M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 1135-1144.

**Dataset and LLM-based radiology reporting (the biggest current gap in related work):**

- A. E. W. Johnson et al., "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports," *Scientific Data*, vol. 6, 2019.
- IHRAS study on integrating CNN classification, Grad-CAM, segmentation, and an LLM for automated chest X-ray report generation, published in a 2025 special issue on AI in medical imaging (verify exact citation on the publisher's page before use — this paper is architecturally very close to Rad-Intel's own pipeline and is worth reading closely, not just citing).
- RaDialog: a large vision-language model for radiology report generation and conversational assistance, accepted at MIDL 2025 (also available as an arXiv preprint) — relevant for the report-generation and future conversational-query direction mentioned in your fourth project objective.
- Google DeepMind's evaluation of Gemini models' medical capabilities (2024 arXiv preprint, "Capabilities of Gemini models in medicine") — directly relevant since the project uses the Gemini API for reporting, and gives citable evidence for why Gemini is a reasonable LLM choice for this task.
- A 2025-2026 hybrid lesion-aware chest X-ray report generation framework combining contrastive learning with an LLM-based aggregator, published in *Scientific Reports* — useful as a recent point of comparison for the "hybrid classification + LLM reporting" framing.

Search each of these on Google Scholar or the publisher site to get the exact, final citation before submission — do not copy the descriptions above directly into the paper as citations; they are pointers for you or your co-authors to locate and verify the primary source.

---

## 8. Final Pre-Submission Checklist

- [ ] Confirm exact page limit and template from the current year's BIBE/BIBM Call for Papers.
- [ ] Remove all author names/affiliations for double-blind submission.
- [ ] All results reported as mean ± std across multiple runs, not single-run numbers.
- [ ] Ablation study included and clearly tied to the architecture description.
- [ ] Every figure/table referenced in text.
- [ ] Every abbreviation defined on first use.
- [ ] Reference list in strict IEEE numbered order matching in-text citation order.
- [ ] Dataset licensing/ethics statement included.
- [ ] Run the final PDF through IEEE PDF eXpress before submission.
- [ ] Spell-check and consistency pass (contribution list in Introduction matches Conclusion summary).
