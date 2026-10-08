// Rad-Intel Minimalist UI Logic

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImg = document.getElementById("preview-img");
  const previewClear = document.getElementById("preview-clear");

  const btnSamplePneumonia = document.getElementById("btn-sample-pneumonia");
  const btnSampleNormal = document.getElementById("btn-sample-normal");
  const btnAnalyze = document.getElementById("btn-analyze");

  const metaToggle = document.getElementById("meta-toggle");
  const metaFields = document.getElementById("meta-fields");
  const inputPatientId = document.getElementById("input-patient-id");
  const selectModel = document.getElementById("select-model");
  const inputPatientAge = document.getElementById("input-patient-age");
  const selectPatientSex = document.getElementById("select-patient-sex");
  const inputPatientHistory = document.getElementById("input-patient-history");

  const resultsEmpty = document.getElementById("results-empty");
  const loadingPanel = document.getElementById("loading-panel");
  const loadingStep = document.getElementById("loading-step");
  const resultsContent = document.getElementById("results-content");

  const verdictBanner = document.getElementById("verdict-banner");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictConfidence = document.getElementById("verdict-confidence");

  const probNormalPct = document.getElementById("prob-normal-pct");
  const probPneumoniaPct = document.getElementById("prob-pneumonia-pct");
  const probFillNormal = document.getElementById("prob-fill-normal");
  const probFillPneumonia = document.getElementById("prob-fill-pneumonia");

  const imgOriginal = document.getElementById("img-original");
  const imgGradcam = document.getElementById("img-gradcam");
  const findingsText = document.getElementById("findings-text");

  const reportEngine = document.getElementById("report-engine");
  const reportFindings = document.getElementById("report-findings");
  const reportImpression = document.getElementById("report-impression");
  const reportRecommendations = document.getElementById("report-recommendations");

  const btnDownloadPdf = document.getElementById("btn-download-pdf") || document.getElementById("btn-download-report");
  const btnCopyReport = document.getElementById("btn-copy-report");
  const btnDownloadLatex = document.getElementById("btn-download-latex");
  const toast = document.getElementById("toast");

  // State
  let selectedFile = null;
  let currentPreviewUrl = null;
  let lastAnalysisResponse = null;
  let loadingInterval = null;

  // 1. Drag & Drop Handlers
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-active");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-active");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-active");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  previewClear.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  function clearSelectedFile() {
    selectedFile = null;
    currentPreviewUrl = null;
    fileInput.value = "";
    previewContainer.style.display = "none";
    dropzone.style.display = "block";
    btnAnalyze.disabled = true;
  }

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      showToast("Please upload an image file (JPEG or PNG).");
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      currentPreviewUrl = e.target.result;
      previewImg.src = currentPreviewUrl;
      dropzone.style.display = "none";
      previewContainer.style.display = "block";
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // 2. Quick Sample CXR Loaders
  async function loadSampleCXR(filename, defaultPatientId, defaultHistory) {
    clearSelectedFile();
    showToast(`Loading sample: ${filename}...`);
    try {
      const res = await fetch(`/sample_images/${filename}`);
      if (!res.ok) throw new Error("Could not fetch sample CXR");
      const blob = await res.blob();
      const file = new File([blob], filename, { type: "image/jpeg" });

      if (defaultPatientId && !inputPatientId.value) {
        inputPatientId.value = defaultPatientId;
      }
      if (defaultHistory && !inputPatientHistory.value) {
        inputPatientHistory.value = defaultHistory;
      }

      handleFile(file);
      showToast("Sample loaded. Click 'Analyze Radiograph' to run diagnosis.");
    } catch (err) {
      console.error(err);
      showToast("Failed to load sample image.");
    }
  }

  btnSamplePneumonia.addEventListener("click", () => {
    loadSampleCXR(
      "sample_pneumonia_person100_bacteria_475.jpeg",
      "PT-PNA-475",
      "52M presenting with high fever (39°C), chills, productive rust-colored sputum for 5 days."
    );
  });

  btnSampleNormal.addEventListener("click", () => {
    loadSampleCXR(
      "sample_normal_IM-0001-0001.jpeg",
      "PT-NORM-001",
      "Routine pre-operative medical clearance screening. Patient is asymptomatic with clear breath sounds."
    );
  });

  // 3. Metadata Accordion Toggle
  metaToggle.addEventListener("click", () => {
    const isOpen = metaFields.classList.toggle("open");
    metaToggle.setAttribute("aria-expanded", isOpen);
    metaToggle.innerHTML = isOpen
      ? "<span>⚙️</span> Clinical Context & Model Settings ▴"
      : "<span>⚙️</span> Clinical Context & Model Settings ▾";
  });

  // 4. Analyze Radiograph Flow
  btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Reset view to loading
    resultsEmpty.style.display = "none";
    resultsContent.style.display = "none";
    loadingPanel.style.display = "block";
    btnAnalyze.disabled = true;

    // Simulated progress telemetry
    const steps = [
      "Standardizing image & applying CLAHE enhancement...",
      "Extracting local and global deep neural representations...",
      "Computing Grad-CAM anatomical saliency maps...",
      "Synthesizing structured clinical report with Gemini AI...",
    ];
    let stepIndex = 0;
    loadingStep.textContent = steps[stepIndex];
    loadingInterval = setInterval(() => {
      stepIndex = (stepIndex + 1) % steps.length;
      loadingStep.textContent = steps[stepIndex];
    }, 1800);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_name", selectModel.value);
    formData.append("patient_id", inputPatientId.value || "Anonymous");
    formData.append("patient_age", inputPatientAge.value || "N/A");
    formData.append("patient_sex", selectPatientSex.value || "N/A");
    formData.append("patient_history", inputPatientHistory.value || "Clinical screening evaluation.");

    try {
      const response = await fetch("/api/v1/analyze", {
        method: "POST",
        body: formData,
      });

      clearInterval(loadingInterval);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Analysis request failed.");
      }

      const data = await response.json();
      lastAnalysisResponse = data;
      renderResults(data);
    } catch (err) {
      clearInterval(loadingInterval);
      loadingPanel.style.display = "none";
      resultsEmpty.style.display = "flex";
      btnAnalyze.disabled = false;
      alert(`Analysis Error: ${err.message}`);
    }
  });

  // 5. Render Results Function
  function renderResults(data) {
    loadingPanel.style.display = "none";
    resultsContent.style.display = "block";
    btnAnalyze.disabled = false;

    const predClass = data.prediction.prediction_class;
    const conf = data.prediction.confidence;
    const isPneumonia = predClass === "PNEUMONIA";

    // Diagnosis Banner
    verdictBanner.className = `verdict-banner ${isPneumonia ? "pneumonia" : "normal"}`;
    verdictIcon.textContent = isPneumonia ? "⚠️" : "✅";
    verdictTitle.textContent = isPneumonia
      ? "PNEUMONIA DETECTED"
      : "NO PNEUMONIA DETECTED (NORMAL)";
    verdictConfidence.textContent = `${(conf * 100).toFixed(1)}% Confidence`;

    // Probability Bars
    const normalProb = data.prediction.probabilities.NORMAL || 0;
    const pnaProb = data.prediction.probabilities.PNEUMONIA || 0;
    const normalPct = (normalProb * 100).toFixed(1) + "%";
    const pnaPct = (pnaProb * 100).toFixed(1) + "%";

    probNormalPct.textContent = normalPct;
    probPneumoniaPct.textContent = pnaPct;
    probFillNormal.style.width = normalPct;
    probFillPneumonia.style.width = pnaPct;

    // Visualizer Images
    imgOriginal.src = currentPreviewUrl;
    if (data.explanation && data.explanation.overlay_base64) {
      imgGradcam.src = `data:image/png;base64,${data.explanation.overlay_base64}`;
    }

    // Anatomical Localization
    if (data.explanation && data.explanation.localization) {
      const loc = data.explanation.localization;
      let text = loc.dominant_zone_description || "Lung fields demonstrate baseline distribution.";
      if (loc.is_bilateral) {
        text += " Bilateral opacity distribution identified.";
      } else {
        text += " Unilateral presentation.";
      }
      findingsText.textContent = text;
    }

    // Clinical Report
    if (data.report) {
      reportEngine.textContent = data.report.engine || "Gemini 3.5 Flash-Lite";
      const sec = data.report.sections || {};
      reportFindings.textContent = sec.findings || "No focal findings.";
      reportImpression.textContent = sec.impression || "Evaluation completed.";
      reportRecommendations.textContent = sec.recommendations || "Clinical correlation advised.";
    }

    // Smooth scroll to results
    resultsContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // 6. Report Copy, PDF & LaTeX Download Actions
  btnCopyReport.addEventListener("click", () => {
    if (!lastAnalysisResponse || !lastAnalysisResponse.report) return;
    const r = lastAnalysisResponse.report;
    const textToCopy = r.full_markdown || [
      `RAD-INTEL CLINICAL RADIOLOGY REPORT`,
      `==================================`,
      `Impression: ${r.sections?.impression || ""}`,
      `Findings: ${r.sections?.findings || ""}`,
      `Recommendations: ${r.sections?.recommendations || ""}`,
    ].join("\n\n");

    navigator.clipboard.writeText(textToCopy).then(() => {
      showToast("Report copied to clipboard!");
    });
  });

  if (btnDownloadPdf) {
    btnDownloadPdf.addEventListener("click", async () => {
      if (!lastAnalysisResponse || !lastAnalysisResponse.report) return;
      const r = lastAnalysisResponse.report;
      const analysisId = lastAnalysisResponse.analysis_id || "CXR";
      const filename = `Rad_Intel_Report_${analysisId.substring(0, 8).toUpperCase()}.pdf`;

      // 1. Download directly from base64 if available
      if (r.pdf_base64) {
        try {
          const byteCharacters = atob(r.pdf_base64);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: "application/pdf" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          showToast("Clinical PDF report downloaded!");
          return;
        } catch (e) {
          console.warn("Base64 decode failed, falling back to server fetch", e);
        }
      }

      // 2. Fallback to API route download
      try {
        const resp = await fetch(`/api/v1/analyze/${analysisId}/pdf`);
        if (!resp.ok) throw new Error("PDF download failed");
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast("Clinical PDF report downloaded!");
      } catch (err) {
        showToast("Error downloading PDF report.");
      }
    });
  }

  if (btnDownloadLatex) {
    btnDownloadLatex.addEventListener("click", () => {
      if (!lastAnalysisResponse || !lastAnalysisResponse.report) return;
      const r = lastAnalysisResponse.report;
      const latex = r.latex_content || "% No LaTeX source available";
      const blob = new Blob([latex], { type: "application/x-tex;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Rad_Intel_Report_${(lastAnalysisResponse.analysis_id || "CXR").substring(0, 8).toUpperCase()}.tex`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast("LaTeX report source downloaded (.tex)!");
    });
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => {
      toast.classList.remove("show");
    }, 2800);
  }
});
