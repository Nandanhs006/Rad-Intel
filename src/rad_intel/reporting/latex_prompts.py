"""
Strict Prompt Engineering & Rules for Clinical LaTeX Radiology Report Generation.
Adheres strictly to the American College of Radiology (ACR) and RSNA standardized
structured reporting guidelines, with rigid anti-hallucination guardrails and LaTeX syntax constraints.
"""

LATEX_REPORT_SYSTEM_PROMPT = r"""You are an expert AI clinical radiologist assistant and medical informatics specialist specializing in chest radiography (CXR) interpretation.
Your task is to generate a standardized, publication-grade, ACR-compliant clinical radiology report formatted ENTIRELY in valid LaTeX, based STRICTLY on the objective quantitative model predictions and Grad-CAM explainability localization data provided.

======================================================================
STRICT LATEX FORMATTING & SYNTAX RULES:
======================================================================
1. OUTPUT FORMAT:
   - Generate ONLY clean, syntactically valid LaTeX code.
   - Do NOT wrap your output in conversational filler, preamble commentary, or explanations.
   - If markdown code blocks are used, use only ```latex ... ```.

2. ALLOWED PACKAGES & DOCUMENT SETUP:
   - Use standard LaTeX packages: geometry (margin=0.75in), xcolor, booktabs, tabularx, tcolorbox, fancyhdr, enumitem, amsmath.
   - Do NOT use obscure, non-standard, or external package dependencies.

3. CHARACTER ESCAPING:
   - ALWAYS properly escape reserved LaTeX characters in all text, tables, and patient data:
     * % must be written as \%
     * & must be written as \&
     * _ must be written as \_
     * # must be written as \#
     * $ must be written as \$
     * { and } must be written as \{ and \}

4. DOCUMENT STRUCTURE:
   The LaTeX document MUST be organized with the following exact structural hierarchy:
   - HEADER: Institution title ("RAD-INTEL CLINICAL DECISION SUPPORT SYSTEM"), Department of Thoracic Radiology, Report ID, and Timestamp.
   - PATIENT DEMOGRAPHICS & STUDY METADATA: Tabular layout with Patient ID, Age, Sex, Clinical Indication, Modality, Examination View.
   - QUANTITATIVE AI MODEL FINDINGS:
     * Primary Classification Verdict (\textbf{NORMAL} or \textbf{PNEUMONIA})
     * Diagnostic Confidence Score percentage
     * Class Probabilities table (Normal vs Pneumonia)
   - EXPLAINABLE AI (GRAD-CAM) ANATOMICAL LOCALIZATION:
     * Dominant Lung Zone description and intensity score
     * Bilateral Involvement status
     * 4-Zone Quadrant Saliency table:
       - Right Upper Zone (RUZ)
       - Right Lower Zone (RLZ)
       - Left Upper Zone (LUZ)
       - Left Lower Zone (LLZ)
   - FORMAL ACR CLINICAL REPORT SECTIONS:
     * \section*{EXAMINATION}
     * \section*{CLINICAL INDICATION}
     * \section*{TECHNIQUE}
     * \section*{FINDINGS} (Bulleted \begin{itemize} covering: Lungs & Airspaces, Pleural Spaces, Cardiomediastinal Silhouette, Osseous Structures)
     * \section*{IMPRESSION} (Numbered \begin{enumerate} synthesizing the primary diagnostic conclusion)
     * \section*{RECOMMENDATIONS} (Numbered \begin{enumerate} offering actionable clinical next steps)
   - CLINICAL DECISION SUPPORT REGULATORY NOTICE:
     * Use a tcolorbox or framed box containing the mandatory medicolegal verification statement.
   - PHYSICIAN SIGNATURE ATTESTATION:
     * Attending Radiologist signature and date placeholder lines.

======================================================================
STRICT CLINICAL & ANTI-HALLUCINATION GUARDRAILS:
======================================================================
1. GROUNDING: Ground every single clinical statement EXCLUSIVELY on the provided classification label, confidence score, and Grad-CAM lung zone saliency data.
2. NO FABRICATION: DO NOT invent, assume, or extrapolate prior examinations, prior surgical procedures, unprovided patient names, or extraneous laboratory values.
3. NORMAL FINDINGS: If the classification verdict is NORMAL:
   - Report lung fields as clear, well-expanded, and free of focal consolidation, interstitial infiltrates, pleural effusion, or pneumothorax.
   - Impression must state no acute cardiopulmonary abnormality.
4. PNEUMONIA FINDINGS: If the classification verdict is PNEUMONIA:
   - Describe airspace consolidation, opacification, or infiltrates specifically localized to the dominant lung zone(s) indicated in the input.
   - Note bilateral involvement accurately according to the provided bilateral status flag.
   - Mention that cardiomediastinal contours and osseous structures remain within normal limits unless otherwise indicated.
5. RECOMMENDATIONS: Keep recommendations clinically appropriate:
   - For Pneumonia: Clinical correlation with sputum/blood cultures, inflammatory markers, empiric therapy as indicated, and 4-6 week follow-up radiograph.
   - For Normal: Routine clinical follow-up; investigate alternative non-pulmonary etiologies if symptoms persist.
6. REGULATORY NOTICE: Must state that Rad-Intel is an automated AI decision-support system and that all findings must be reviewed and signed off by a board-certified radiologist prior to clinical intervention.
"""

LATEX_REPORT_USER_PROMPT_TEMPLATE = r"""Generate the complete, formal LaTeX radiology report for the following verified chest radiograph analysis:

[QUANTITATIVE CLASSIFICATION]
- Primary Diagnosis: {prediction_class}
- Diagnostic Confidence: {confidence_pct:.1f}\%
- Model Architecture: {model_name}
- Class Probabilities: Normal = {prob_normal:.3f}, Pneumonia = {prob_pneumonia:.3f}

[EXPLAINABLE AI ANATOMICAL LOCALIZATION]
- Dominant Lung Zone: {dominant_zone_desc} (intensity: {dominant_intensity:.3f})
- Bilateral Involvement: {bilateral_status}
- Quadrant Intensity Scores:
  * Right Upper Zone (RUZ): {ruz:.3f}
  * Right Lower Zone (RLZ): {rlz:.3f}
  * Left Upper Zone (LUZ): {luz:.3f}
  * Left Lower Zone (LLZ): {llz:.3f}
- Saliency Summary: {saliency_summary}

[PATIENT & STUDY METADATA]
- Report ID: {report_id}
- Examination Date: {exam_date}
- Patient ID: {patient_id}
- Age: {patient_age}
- Sex: {patient_sex}
- Clinical History / Indication: {clinical_history}
- Modality: Digital Radiography (Frontal Chest PA/AP)

Generate the complete, compilable LaTeX document starting with \documentclass[11pt,a4paper]{{article}} and ending with \end{{document}}.
Adhere strictly to all formatting and clinical rules.
"""

OFFLINE_LATEX_REPORT_TEMPLATE = r"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{xcolor}}
\usepackage{{fancyhdr}}
\usepackage{{tcolorbox}}
\usepackage{{enumitem}}

\definecolor{{mednavy}}{{RGB}}{{15, 23, 42}}
\definecolor{{medblue}}{{RGB}}{{30, 58, 138}}
\definecolor{{medgray}}{{RGB}}{{100, 116, 139}}
\definecolor{{alertred}}{{RGB}}{{185, 28, 28}}
\definecolor{{safeemerald}}{{RGB}}{{4, 120, 87}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\footnotesize \textcolor{{medgray}}{{\textbf{{RAD-INTEL}} | Clinical Decision Support System}}}}
\fancyhead[R]{{\footnotesize \textcolor{{medgray}}{{Study ID: {report_id}}}}}
\fancyfoot[C]{{\footnotesize \textcolor{{medgray}}{{Page \thepage\ of 1 | Confidential Medical Record}}}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\renewcommand{{\footrulewidth}}{{0.4pt}}

\begin{{document}}

% HEADER BANNER
\noindent
\begin{{minipage}}{{0.65\textwidth}}
    {{\LARGE \textbf{{\textcolor{{medblue}}{{RAD-INTEL CLINICAL REPORT}}}}}}\\[2pt]
    {{\footnotesize \textcolor{{medgray}}{{Automated Chest Radiography Interpretation \& Explainability Network}}}}\\[1pt]
    {{\footnotesize Department of Diagnostic Radiology \& Thoracic Imaging}}
\end{{minipage}}
\hfill
\begin{{minipage}}{{0.32\textwidth}}
    \begin{{flushright}}
        \footnotesize
        \textbf{{Report ID:}} {report_id}\\[1pt]
        \textbf{{Date:}} {exam_date}\\[1pt]
        \textbf{{Engine:}} {engine}
    \end{{flushright}}
\end{{minipage}}

\vspace{{6pt}}
\noindent\textcolor{{medblue}}{{\rule{{\textwidth}}{{1.5pt}}}}
\vspace{{6pt}}

% PATIENT & STUDY DEMOGRAPHICS
\noindent
\textbf{{\large \textcolor{{mednavy}}{{PATIENT \& STUDY DEMOGRAPHICS}}}}\\[4pt]
\begin{{tabularx}}{{\textwidth}}{{@{{}}lXlX@{{}}}}
    \textbf{{Patient ID:}} & {patient_id} & \textbf{{Modality:}} & Digital Chest Radiography (PA/AP) \\
    \textbf{{Age:}} & {patient_age} & \textbf{{Examination:}} & Chest 1 View \\
    \textbf{{Sex:}} & {patient_sex} & \textbf{{Indication:}} & {clinical_history} \\
\end{{tabularx}}

\vspace{{10pt}}

% AI QUANTITATIVE ASSESSMENT & EXPLAINABILITY
\noindent
\textbf{{\large \textcolor{{mednavy}}{{AI DIAGNOSTIC ASSESSMENT \& EXPLAINABILITY}}}}\\[4pt]
\begin{{tcolorbox}}[colback={verdict_bg},colframe={verdict_frame},arc=2mm,boxrule=1pt,width=\textwidth]
    \noindent
    \begin{{minipage}}{{0.45\textwidth}}
        {{\small \textbf{{PRIMARY DIAGNOSTIC VERDICT:}}}}\\[2pt]
        {{\Large \textbf{{\textcolor{{{verdict_color}}}{{{prediction_class}}}}}}}\\[3pt]
        {{\small \textbf{{Model Confidence:}} \textbf{{{confidence_pct:.1f}\%}}}}
    \end{{minipage}}
    \hfill
    \begin{{minipage}}{{0.50\textwidth}}
        \small
        \textbf{{Class Probabilities:}}\\
        $\bullet$ Normal: {prob_normal:.3f} \quad $\bullet$ Pneumonia: {prob_pneumonia:.3f}\\[2pt]
        \textbf{{Dominant Zone:}} {dominant_zone_desc}\\[1pt]
        \textbf{{Bilateral Involvement:}} {bilateral_status}
    \end{{minipage}}
\end{{tcolorbox}}

\vspace{{6pt}}
\noindent
\textbf{{\small \textcolor{{mednavy}}{{Grad-CAM Anatomical Quadrant Saliency Distribution:}}}}\\[2pt]
\begin{{tabularx}}{{\textwidth}}{{XXXX}}
    \toprule
    \textbf{{Right Upper Zone}} & \textbf{{Right Lower Zone}} & \textbf{{Left Upper Zone}} & \textbf{{Left Lower Zone}} \\
    \midrule
    {ruz:.3f} & {rlz:.3f} & {luz:.3f} & {llz:.3f} \\
    \bottomrule
\end{{tabularx}}

\vspace{{10pt}}

% FORMAL CLINICAL REPORT SECTIONS
\section*{{EXAMINATION}}
Chest Radiograph (Single Frontal Projection).

\section*{{CLINICAL INDICATION}}
Patient ID: {patient_id}; Age: {patient_age}; Sex: {patient_sex}. History: {clinical_history}.

\section*{{TECHNIQUE}}
Digital frontal radiographic acquisition of the thorax. Standard exposure parameters.

\section*{{FINDINGS}}
\begin{{itemize}}[leftmargin=1.5em, itemsep=2pt]
{findings_items}
\end{{itemize}}

\section*{{IMPRESSION}}
\begin{{enumerate}}[leftmargin=1.5em, itemsep=2pt]
{impression_items}
\end{{enumerate}}

\section*{{RECOMMENDATIONS}}
\begin{{enumerate}}[leftmargin=1.5em, itemsep=2pt]
{recommendation_items}
\end{{enumerate}}

\vspace{{12pt}}

% REGULATORY NOTICE BOX
\begin{{tcolorbox}}[colback=gray!8,colframe=gray!40,arc=1.5mm,boxrule=0.6pt,width=\textwidth]
    \footnotesize \textcolor{{medgray}}{{\textbf{{REGULATORY NOTICE \& VERIFICATION:}} This document was generated by the Rad-Intel Deep Learning Clinical Decision Support Framework. This system is designed solely to augment clinical judgment and does not constitute an autonomous medical diagnosis. All computer vision findings and AI-synthesized narratives must be independently verified by a licensed radiologist or attending physician prior to patient management decisions.}}
\end{{tcolorbox}}

\vspace{{14pt}}
\noindent
\begin{{minipage}}{{0.5\textwidth}}
    \footnotesize
    \textbf{{Interpreting Radiologist Attestation:}}\\[16pt]
    \rule{{0.85\textwidth}}{{0.4pt}}\\[2pt]
    Attending Radiologist, M.D. / D.O.
\end{{minipage}}
\hfill
\begin{{minipage}}{{0.4\textwidth}}
    \begin{{flushright}}
        \footnotesize
        \textbf{{Verification Timestamp:}}\\[16pt]
        \rule{{0.8\textwidth}}{{0.4pt}}\\[2pt]
        Electronically Verified Date \& Time
    \end{{flushright}}
\end{{minipage}}

\end{{document}}
"""
