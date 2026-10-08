# Reference List — Rad-Intel
### Categorized Literature Survey for Pneumonia Detection + Explainable AI + LLM Reporting

Every entry below was located through live web search and is a real, verifiable publication (not generated from memory). Bibliographic details (author, year, venue) are given where found; verify exact page numbers/DOIs against the publisher page before final citation, since some are recent preprints. Entries are grouped by theme and tagged by relevance to Rad-Intel:

- **[High]** — directly matches the project's architecture, technique, or core claim (DenseNet + Swin Transformer + CBAM + Grad-CAM/LIME + LLM reporting, applied to pneumonia/chest X-rays).
- **[Medium]** — same problem domain or adjacent technique, useful for comparison, baselines, or context.
- **[Low-but-useful]** — general-purpose or tangential, but supplies a citable fact, method, or framing the paper will need somewhere (statistics, a supporting technique, general LLM/CDSS context).

---

## A. Foundational Architectures and Attention Mechanisms

1. **Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. CVPR, pp. 4700-4708.** [High] — The original DenseNet paper; the exact backbone your framework's local-feature branch is built on.
2. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, pp. 9992-10002.** [High] — The original Swin Transformer paper; the global-context branch of your architecture.
3. **Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.** [High] — Introduces the Vision Transformer; needed to explain why transformers apply to images at all before introducing Swin.
4. **Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.** [Medium] — The base transformer/self-attention paper; useful as the root citation before ViT/Swin, but not medical-specific.
5. **Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. ECCV, pp. 3-19.** [High] — The original CBAM paper; currently uncited in your report despite CBAM being a core component of your model.
6. **He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR, pp. 770-778.** [Medium] — ResNet; a standard baseline architecture your ablation/comparison table should include.
7. **Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, pp. 6105-6114.** [Medium] — Another standard efficient baseline for the comparison table.
8. **HiFuse: Hierarchical Multi-Scale Feature Fusion Network for Medical Image Classification.** [Medium] — A CNN-Transformer fusion network for medical imaging; useful for justifying your own DenseNet-Swin fusion design choice.

## B. CBAM / Attention-Based Pneumonia Detection (closest prior work to your method)

9. **Dey, U. K. CBAM-Enhanced DenseNet121 for Multi-Class Chest X-Ray Classification with Grad-CAM Explainability. (arXiv 2604.12305).** [High] — Near-identical combination to your framework (CBAM + DenseNet121 + Grad-CAM) but for 3-class bacterial/viral classification; a key direct comparator.
10. **An Enhanced Deep Learning Framework for Pneumonia Detection in Chest X-rays (CBAM + DenseNet-121). SN Computer Science, 2025.** [High] — Reports DenseNet-121 baseline at 95.57% rising to 98.64% with CBAM on the Kermany dataset — a directly citable number for your own ablation discussion.
11. **Enhanced X-ray image classification for pneumonia detection using deep learning based CBAM and SE mechanisms. ScienceDirect, 2025.** [High] — Compares CBAM against SE attention specifically for pneumonia; useful for your attention-mechanism ablation argument.
12. **AXNet: Attention-enhanced X-ray network for pneumonia detection (CBAM + ECA). ScienceDirect, 2026.** [High] — Combines CBAM with a second lightweight attention mechanism (ECA) on the same Kaggle pediatric dataset your project likely uses.
13. **Attention-Enhanced CBAM-VGG-16 for Optimized Pneumonia Detection in Chest X-Ray Imaging (ResearchGate, 2025).** [High] — CBAM applied to a different backbone (VGG-16); useful comparator for backbone-choice discussion.
14. **Multi-Model Convolutional Neural Network Ensemble with CBAM and Grad-CAM explainability. IJSAT, 2025.** [High] — Combines CBAM, ensembling, and Grad-CAM in one pipeline much like your proposed system; reports 98.44% accuracy, 0.99 F1.
15. **Explainable Hybrid ConvNeXt–Vision Transformer Model for Pneumonia Detection (CBAM + ViT-B/16). Research Square preprint, 2026.** [High] — A hybrid CNN+Transformer+CBAM design, structurally the closest published analogue to your DenseNet+Swin+CBAM architecture.
16. **Interpretable CNN-Multilevel Attention Transformer for Rapid Recognition of Pneumonia from Chest X-Ray Images. (arXiv 2210.16584).** [High] — Combines multilevel attention with a Transformer for pneumonia, directly relevant to your fusion + attention design.

## C. CNN–Transformer Hybrid Architectures for Chest X-Ray/Pneumonia Classification

17. **Taslimi, S., et al. SwinCheX: Multi-label classification on chest X-ray images with transformers. (arXiv 2206.04246).** [High] — Swin Transformer applied directly to ChestX-ray14; a core architectural precedent for your global-context branch.
18. **CheX-DS: Improving Chest X-ray Image Classification with Ensemble Learning Based on DenseNet and Swin Transformer. (arXiv 2505.11168).** [High] — The closest published match to your exact architecture (DenseNet + Swin Transformer ensemble); reports 83.76% average AUC on ChestX-ray14 — a benchmark number worth citing directly.
19. **Mustapha, B., Zhou, Y., Shan, C., & Xiao, Z. Enhanced Pneumonia Detection in Chest X-Rays Using Hybrid Convolutional and Vision Transformer Networks. Current Medical Imaging, 2025.** [High] — Hybrid CNN + modified Swin Transformer blocks specifically for pneumonia; explicitly names future work in explainable AI, i.e., the exact gap Rad-Intel fills.
20. **Improved swin transformer-based thorax disease classification with optimal feature selection using chest X-ray (IMSTrans). PLOS ONE, 2025.** [High] — A Swin Transformer variant combined with an auto-encoder attention module for thorax disease classification.
21. **Efficient pneumonia detection using Vision Transformers on chest X-rays. PMC, 2024.** [High] — Direct hybrid CNN-feature-extractor + ViT design for pneumonia; useful architecture-design reference.
22. **Explainable Artificial Intelligence Model for Pneumonia Detection: A Hybrid CNN-ViT and Grad-CAM. (ResearchGate, 2025).** [High] — Combines a CNN-ViT hybrid with Grad-CAM explainability for pneumonia, reporting 96.5% accuracy.
23. **Enhanced swin transformer based tuberculosis classification with segmentation using chest X-ray. PubMed, 2025.** [Medium] — Same technique family (Swin Transformer) applied to a related chest-X-ray disease (TB) rather than pneumonia.
24. **Improved EATFormer: A Vision Transformer for Medical Image Classification. (arXiv 2403.13167).** [Medium] — Benchmarks a hierarchical Transformer against DenseNet/ResNet/EfficientNet on a chest X-ray dataset; useful comparison table source.
25. **A novel method to enhance pneumonia detection via a model-level ensembling of CNN and vision transformer. (arXiv 2401.02358).** [High] — Another CNN+ViT ensembling approach specifically for pneumonia detection.
26. **Pneumonia Detection on Chest X-ray Images Using Ensemble of Deep Convolutional Neural Networks (DenseNet169 + MobileNetV2 + ViT). (arXiv 2312.07965 / MDPI Applied Sciences).** [High] — Ensembles DenseNet with a Vision Transformer for pneumonia; directly relevant precedent.

## D. Explainable AI (Grad-CAM, LIME, SHAP) Applied to Chest X-Ray / Pneumonia

27. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV, pp. 618-626.** [High] — The original Grad-CAM paper; currently missing from your report's reference list despite being a core module.
28. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. ACM SIGKDD (KDD), pp. 1135-1144.** [High] — The original LIME paper; also missing from your current citations.
29. **Explainable AI for Chest X-Ray Diagnosis: Enhancing Transparency and Trust in Medical Deep Learning Systems. Academia.edu preprint, 2025.** [High] — Integrates Grad-CAM, LIME, and SHAP together on NIH ChestX-ray14, essentially a template for your own explainability section.
30. **Interpretable Deep Learning for Pneumonia Detection Using Chest X-Ray Images. MDPI Information, 2025.** [High] — Directly compares Grad-CAM, LIME, and SHAP for pneumonia interpretability and reports quantitative "interpretability scores" — useful for your evaluation-metric design.
31. **EXPLAINABLE DEEP LEARNING FOR AUTOMATED PNEUMONIA DETECTION USING MOBILENETV2 AND GRAD-CAM CNN. TechRxiv, 2025.** [High] — Also deploys the model in Streamlit with Grad-CAM overlays, matching your proposed interface design exactly.
32. **Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations. (arXiv 2511.00456).** [High] — Benchmarks seven architectures (including DenseNet121 and a ViT) under identical conditions with Grad-CAM, a good template for your own baseline comparison table.
33. **Comparative Analysis of Deep Learning Architectures for Multi-Disease Classification of Single-Label Chest X-rays (ConvNeXt + Grad-CAM). (arXiv 2603.13392).** [Medium] — Multi-disease rather than binary pneumonia, but a clean recent example of Grad-CAM reporting style.
34. **FLPneXAINet: Federated deep learning and explainable AI (LIME + Grad-CAM) for pneumonia prediction. PMC, 2025.** [Medium] — Combines both your XAI methods but adds federated learning, which is out of scope for your project; useful only for the XAI half.
35. **QCSA Network: Quaternion Channel-Spatial Attention for pneumonia detection. PMC, 2023.** [Medium] — A different (quaternion-based) channel-spatial attention approach; relevant as an attention-mechanism alternative to CBAM.
36. **spoluan/explainable-ai-pneumonia (GitHub, VGG19 + Grad-CAM).** [Medium] — Not peer-reviewed, but a clean practical Grad-CAM-on-pneumonia implementation reference if you need an engineering example rather than a citation.

## E. Datasets and Benchmarks

37. **Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR, pp. 2097-2106 / 3462-3471.** [High] — The NIH ChestX-ray14 dataset origin paper (108,948 images, 32,717 patients); essential dataset citation.
38. **Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.** [High] — 224,316 radiographs from Stanford; standard alternative/complementary benchmark dataset.
39. **Johnson, A. E. W., Pollard, T. J., Berkowitz, S. J., Greenbaum, N. R., Lungren, M. P., Deng, C., Mark, R. G., & Horng, S. (2019). MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data, 6.** [High] — 377,110 images with paired free-text radiology reports; the dataset your LLM-reporting evaluation would need if you want reference-based report metrics (BLEU/ROUGE).
40. **Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.** [High] — The pediatric pneumonia chest X-ray dataset (5,863 images) almost certainly used in your project; also the original source of the "normal vs. pneumonia" framing.
41. **Benchmarking CXR Foundation Models With Publicly Available MIMIC-CXR and NIH-CXR14 Datasets. (arXiv 2512.06014).** [Low-but-useful] — A recent (Dec 2025) benchmarking study of foundation-model embeddings on the same two datasets; useful context for "recent trends" but not directly about your architecture.
42. **ChEX: Interactive Localization and Region Description in Chest X-rays. (arXiv 2404.15770).** [Low-but-useful] — Uses MIMIC-CXR for grounded region description; tangential but shows how datasets are used for text-image grounding, relevant background for your LLM module.

## F. LLM-Based Radiology Report Generation (fills the current biggest gap in your related work)

43. **Hybrid framework for lesion-aware, clinically coherent chest X-ray report generation using contrastive learning and large language models (CLALA-Net). Scientific Reports, 2026.** [High] — An LLM-based aggregator converts classification output into a clinically coherent report — structurally the closest published analogue to your reporting module.
44. **Recent Progress in Deep Learning for Chest X-Ray Report Generation. MDPI, 2026 (review covering 2018-2025, focus on 2024-2025).** [High] — A recent comprehensive review of the entire subfield; the single most useful source to build your "LLM reporting" related-work subsection from.
45. **Pham, T. T., et al. Look & Mark: Leveraging Radiologist Eye Fixations and Bounding Boxes in Multimodal LLMs for Chest X-ray Report Generation. (arXiv 2505.22222).** [Medium] — Recent multimodal LLM approach; its reference list itself is a good source of further citations (MAIRA-1, CXR-LLaVA, etc.).
46. **Hyland, S. L., et al. MAIRA-1: A specialised large multimodal model for radiology report generation. arXiv, 2023.** [Medium] — A specialized (not general-purpose) LLM for radiology reporting; useful contrast case against your general-purpose Gemini-based approach.
47. **Pellegrini, C., et al. RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance. (arXiv 2311.18681).** [High] — Directly relevant to your fourth project objective (LLM-based query support) since it also supports conversational follow-up, not just one-shot reports.
48. **Automated Radiology Report Labeling in Chest X-Ray Pathologies: Development and Evaluation of a Large Language Model Framework. PMC, 2025.** [High] — Uses a GPT-based LLM to label/structure chest X-ray findings, achieving F1 0.90, beating CheXpert's own labeler — relevant precedent for using an LLM downstream of a classifier.
49. **A Review of Longitudinal Radiology Report Generation: Dataset Composition, Methods, and Performance Evaluation. (arXiv 2510.12444).** [Medium] — Focuses on longitudinal (multi-visit) reporting, a possible future-work direction, less central to your current single-image pipeline.
50. **Calibrated Confidence Expression for Radiology Report Generation. (arXiv 2603.29492).** [Medium] — Addresses LLM overconfidence in generated reports; directly relevant to your hallucination-safeguard design (Section 3.5E of the paper guide).
51. **Ostmeier, S., et al. GREEN: Generative Radiology Report Evaluation and Error Notation. arXiv preprint, 2024 (cited within [50] above).** [Medium] — A candidate automatic metric for evaluating your LLM-generated reports beyond BLEU/ROUGE.
52. **Kaur, N., Mittal, A., & Singh, G. (2022). Methods for automatic generation of radiological reports of chest radiographs: a comprehensive survey. Multimedia Tools and Applications.** [Medium] — An earlier (pre-LLM-era) survey; useful for showing the field's evolution from templated to LLM-based reporting.

## G. LLM Medical/Diagnostic Capability Evaluation (relevant to justifying the Gemini API choice)

53. **Saab, K., Tu, T., Weng, W.-H., Tanno, R., Stutz, D., et al. (2024). Capabilities of Gemini models in medicine. (arXiv 2404.18416).** [High] — Google's own evaluation of Gemini on medical tasks; the most direct, citable justification for choosing Gemini as your reporting LLM.
54. **Digital Diagnostics: The Potential of Large Language Models in Recognizing Symptoms of Common Illnesses. (arXiv 2405.06712).** [Medium] — Compares GPT-4, Gemini, and GPT-3.5 on diagnostic accuracy; useful general LLM-in-medicine comparator, not imaging-specific.
55. **Evaluating Bard/Gemini Pro and GPT-4 Vision Against Student Performance in Medical Visual Question Answering. PMC, 2024.** [Medium] — Evaluates Gemini specifically on image-based medical questions, closer to your use case than pure text evaluations.
56. **Diagnostic Performances of GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro in "Diagnosis Please" Cases. medRxiv, 2024.** [Medium] — Radiology-specific diagnostic quiz benchmark comparing frontier LLMs including Gemini.
57. **High Consistency, Limited Accuracy: Evaluating Large Language Models for Binary Medical Diagnosis (incl. Gemini-2.0-Flash). medRxiv, 2025.** [Medium] — Directly evaluates a Gemini Flash-family model (the same family you plan to use) on binary diagnostic consistency and accuracy — a useful caveat/limitation citation.
58. **Evaluating ChatGPT, Gemini and other LLMs in orthopaedic diagnostics: A prospective clinical study. ScienceDirect, 2024.** [Low-but-useful] — Different specialty (orthopaedics) but a template for how to run a prospective LLM-diagnosis comparison study, if you extend Rad-Intel to a user study.
59. **A Comparative Evaluation of GPT-4 Turbo and Gemini-Pro in Medical Licensing Exams. PMC, 2026.** [Low-but-useful] — Text-only benchmark, not imaging; tangential but shows Gemini's general medical reasoning limitations relative to GPT-4.
60. **Large language models' capabilities in responding to tuberculosis medical questions: testing ChatGPT, Gemini, and Copilot. Scientific Reports, 2025.** [Low-but-useful] — Different disease and task (patient education Q&A, not report generation), included only for breadth on Gemini-in-medicine evaluation.
61. **Evaluating the Reasoning Capabilities of Large Language Models for Medical Coding and Hospital Readmission Risk Stratification. PMC, 2025.** [Low-but-useful] — Different clinical task entirely (coding/readmission), useful only as a general "LLMs in clinical workflows" citation.

## H. Clinical Decision Support Systems — Trust, Adoption, and Framing

62. **Trust in Artificial Intelligence–Based Clinical Decision Support Systems Among Health Care Workers: Systematic Review. JMIR, 2025.** [Medium] — Directly supports your Introduction's framing of why explainability and reporting matter for clinical trust/adoption.
63. **AI-Driven Clinical Decision Support Systems: An Ongoing Pursuit of Potential. PMC, 2024.** [Medium] — General AI-CDSS review; good source for defining "clinical decision support system" formally in your Introduction.
64. **Artificial Intelligence in Clinical Decision Support: a Focused Literature Survey. PMC (IMIA Yearbook), 2019.** [Low-but-useful] — Older but foundational CDSS + AI survey; useful for historical framing.
65. **Achieving large-scale clinician adoption of AI-enabled decision support. PMC, 2024.** [Low-but-useful] — Discusses why AI-CDSS tools fail to reach clinical practice despite regulatory approval — useful for your Discussion/limitations section.
66. **AI-based clinical decision support in modern medical physics: Selection, acceptance, commissioning, and quality assurance. PMC.** [Low-but-useful] — Deployment/QA-focused rather than model-focused; only tangentially relevant unless you discuss deployment governance.

## I. Supporting Techniques (Training Methodology)

67. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017/2018). Focal Loss for Dense Object Detection. ICCV / IEEE TPAMI.** [Medium] — The standard loss function for class-imbalanced classification; directly relevant if your Kermany-dataset pneumonia/normal split is imbalanced (it typically is, roughly 3:1).
68. **Convolutional Neural Networks based Focal Loss for Class Imbalance Problem (case study). (arXiv 2001.03329).** [Low-but-useful] — Not chest X-ray specific, but a clean applied comparison of focal loss vs. cross-entropy with DenseNet-121/ResNet-50, directly transferable to your training setup.

## J. Surveys and Meta-Analyses (broad grounding citations)

69. **Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., van der Laak, J. A. W. M., van Ginneken, B., & Sánchez, C. I. (2017). A Survey on Deep Learning in Medical Image Analysis. Medical Image Analysis, 42, 60-88.** [High] — The most-cited foundational survey of deep learning in medical imaging; a near-mandatory citation for any paper in this space.
70. **Liu, X., Faes, L., Kale, A. U., Wagner, S. K., et al. (2019). A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis. The Lancet Digital Health, 1(6), e271-e297.** [High] — The definitive meta-analysis on "AI vs. clinician" diagnostic accuracy; directly supports claims like "matches/exceeds radiologist performance," and also flags the field's poor external-validation practices — useful for your Discussion/limitations.
71. **Advancement of Deep Learning in Pneumonia and COVID-19 Classification and Localization: A Qualitative and Quantitative Analysis. (arXiv 2111.08606).** [Medium] — A review specifically scoped to pneumonia/COVID-19 chest X-ray classification; good source of further baseline numbers.
72. **An ensemble-based approach by fine-tuning deep transfer learning models to classify pneumonia from chest X-ray images (DenseNet-201, etc.). (arXiv 2011.05543).** [Medium] — Another DenseNet-variant pneumonia baseline with full architecture description, useful for your methodology comparison.

## K. Epidemiological Context (for your Introduction's motivation paragraph)

73. **World Health Organization. Child mortality (under 5 years) fact sheet, updated 2026.** [High] — Current statistic: sub-Saharan Africa and Southern Asia account for over 80% of under-five deaths; pneumonia remains among the leading causes.
74. **UNICEF Data. Pneumonia in Children Statistics, 2025.** [High] — Pneumonia kills over 700,000 children under 5 annually (~2,000/day), including ~190,000 newborns — a strong, current, citable opening statistic for your Introduction.
75. **Forum of International Respiratory Societies (FIRS) / GBD 2023 estimates via IHME, World Pneumonia Day statement, Nov 2025.** [Medium] — More recent (2023) global figure: pneumonia killed 2.5 million people overall, including 610,000 children under five — an alternative/complementary statistic to the UNICEF figure above.
76. **WHO. Pneumonia and diarrhoea account for 23% of under-five mortality... guideline, Dec 2024.** [Medium] — Useful for framing pneumonia alongside diarrhoea as the two dominant preventable causes of child death.

---

## Notes on Using This List

- **Coverage gaps closed:** the biggest additions relative to your existing Phase 1 report are Section D (foundational Grad-CAM/LIME citations, currently missing despite being used), Section F (LLM-based radiology reporting — previously absent entirely), and Section C (CBAM+DenseNet+Swin-specific prior work, which lets you position Rad-Intel precisely against near-identical existing systems rather than only against generic CNN baselines).
- **Recency:** several Section B/C/D papers are from late 2025 and early-to-mid 2026 preprints (arXiv, Research Square, TechRxiv) — strong for showing the field is active, but confirm peer-review status before treating them as equal-weight to published journal/conference papers in your related work.
- **Verify before submission:** entries sourced from aggregator pages (Semantic Scholar, ResearchGate, SciRP reference pages) should be cross-checked against the publisher's own page for exact volume/page numbers before going into your final IEEE-formatted reference list.
