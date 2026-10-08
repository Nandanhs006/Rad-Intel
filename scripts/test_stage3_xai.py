r"""
Stage 3 Verification Script: Image Preprocessing & Explainability (Grad-CAM & LIME).
Run with: .\.venv\Scripts\python.exe scripts/test_stage3_xai.py
"""

import numpy as np
import cv2

from rad_intel.preprocessing.transforms import default_preprocessor
from rad_intel.models.factory import create_model
from rad_intel.xai.gradcam import GradCAMExplainer
from rad_intel.xai.lime_explainer import LIMECXRExplainer

def create_synthetic_cxr() -> np.ndarray:
    """Creates a synthetic chest X-ray like image with simulated lung fields and opacity."""
    img = np.ones((512, 512), dtype=np.uint8) * 40
    # Left and right simulated lung regions
    cv2.ellipse(img, (180, 260), (90, 160), 0, 0, 360, 140, -1)
    cv2.ellipse(img, (332, 260), (90, 160), 0, 0, 360, 140, -1)
    # Simulated focal pneumonia consolidation in right lower lobe (image left)
    cv2.circle(img, (190, 320), 45, 210, -1)
    # Gaussian blur to mimic radiograph appearance
    img = cv2.GaussianBlur(img, (25, 25), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def test_stage3():
    print("=" * 65)
    print(" RAD-INTEL :: STAGE 3 VERIFICATION (Preprocessing & XAI)")
    print("=" * 65)

    # 1. Synthetic CXR generation & Preprocessing
    print("\n[1] Testing Image Preprocessing & CLAHE Enhancement:")
    synthetic_cxr = create_synthetic_cxr()
    print(f"    - Original synthetic CXR shape: {synthetic_cxr.shape}")

    tensor, rgb_processed = default_preprocessor.preprocess(synthetic_cxr)
    print(f"    - Processed tensor shape:       {list(tensor.shape)} (dtype={tensor.dtype})")
    print(f"    - Preprocessed RGB shape:       {rgb_processed.shape} (dtype={rgb_processed.dtype})")
    assert tensor.shape == (1, 3, 224, 224)
    assert rgb_processed.shape == (224, 224, 3)
    print("    [PASSED] CXR preprocessor verified.")

    # 2. Hybrid Model Forward Pass
    print("\n[2] Initializing Hybrid Model for Saliency Analysis:")
    model = create_model("hybrid", num_classes=2, pretrained=False)
    print("    - Model instantiated.")

    # 3. Grad-CAM & Anatomical Quadrant Mapping
    print("\n[3] Testing Grad-CAM & Anatomical Localization Mapping:")
    gradcam = GradCAMExplainer(model=model, method="gradcam")
    explanation = gradcam.explain(input_tensor=tensor, original_rgb=rgb_processed, target_category=1)

    heatmap = explanation["heatmap"]
    print(f"    - Grad-CAM heatmap shape:       {heatmap.shape}")
    print(f"    - Heatmap value range:          min={heatmap.min():.4f}, max={heatmap.max():.4f}")
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    loc = explanation["localization"]
    print(f"    - Dominant Zone:                {loc['dominant_zone']} ({loc['dominant_zone_description']})")
    print(f"    - Zone Intensities:             {loc['zone_scores']}")
    print(f"    - Bilateral Involvement:        {loc['is_bilateral']}")
    print(f"    - Summary Statement:            {loc['distribution_summary']}")

    overlay_b64 = explanation["overlay_base64"]
    print(f"    - Overlay Data URI length:      {len(overlay_b64)} chars (starts with: {overlay_b64[:30]}...)")
    assert overlay_b64.startswith("data:image/png;base64,")
    print("    [PASSED] Grad-CAM and anatomical mapping verified.")

    # 4. LIME Superpixel Explainer
    print("\n[4] Testing LIME Superpixel Explainer:")
    lime_explainer = LIMECXRExplainer(model=model)
    lime_result = lime_explainer.explain(
        image_rgb=rgb_processed,
        target_category=1,
        num_samples=50,  # fast test
        num_features=3,
    )
    print(f"    - LIME overlay shape:           {lime_result['overlay_rgb'].shape}")
    print(f"    - LIME Base64 length:           {len(lime_result['overlay_base64'])} chars")
    assert lime_result["overlay_base64"].startswith("data:image/png;base64,")
    print("    [PASSED] LIME superpixel explainer verified.")

    print("\n" + "=" * 65)
    print(" STAGE 3 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_stage3()
