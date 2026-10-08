r"""
Stage 2 Verification Script: Deep Learning Model Architecture.
Run with: .\.venv\Scripts\python.exe scripts/test_stage2_model.py
"""

import torch
import torch.nn.functional as F

from rad_intel.models.factory import create_model, get_model_target_layer
from rad_intel.models.hybrid import HybridDenseNetSwinCBAM
from rad_intel.models.cbam import CBAM

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_stage2():
    print("=" * 65)
    print(" RAD-INTEL :: STAGE 2 VERIFICATION (Deep Learning Architecture)")
    print("=" * 65)

    device = torch.device("cpu")
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    print(f"[*] Input Tensor shape: {list(dummy_input.shape)} on {device}")

    # 1. Test CBAM Standalone
    print("\n[1] Testing CBAM Dual Attention Module:")
    cbam = CBAM(in_planes=512, ratio=16, kernel_size=7).to(device)
    dummy_feat = torch.randn(2, 512, 7, 7, device=device)
    cbam_out = cbam(dummy_feat)
    print(f"    - Input feature map:  {list(dummy_feat.shape)}")
    print(f"    - Output feature map: {list(cbam_out.shape)}")
    assert cbam_out.shape == dummy_feat.shape, "CBAM altered tensor spatial/channel dimensions."
    assert not torch.isnan(cbam_out).any(), "CBAM output contains NaNs."
    print("    [PASSED] CBAM channel & spatial attention passed.")

    # 2. Test Models & Ablation Variants
    models_to_test = [
        ("hybrid", "Proposed Hybrid (DenseNet121 + Swin-T + CBAM)"),
        ("hybrid_no_cbam", "Ablation: Hybrid without CBAM"),
        ("densenet121", "Ablation: Standalone DenseNet121"),
        ("swin_t", "Ablation: Standalone Swin-T"),
        ("resnet50", "Benchmark Baseline: ResNet50"),
    ]

    print("\n[2] Testing Forward Passes & Target Layer Resolution:")
    for model_name, desc in models_to_test:
        print(f"\n  Testing: {desc} [{model_name}]")
        model = create_model(model_name=model_name, num_classes=2, pretrained=False, device=device)
        params = count_parameters(model)

        with torch.no_grad():
            logits = model(dummy_input)
            probs = F.softmax(logits, dim=-1)

        print(f"    - Total Trainable Parameters: {params:,}")
        print(f"    - Output Logits shape:        {list(logits.shape)}")
        print(f"    - Class Probabilities:        {probs[0].tolist()}")

        assert logits.shape == (2, 2), f"Expected output shape (2, 2), got {logits.shape}"
        assert not torch.isnan(logits).any(), f"Logits contain NaNs for model {model_name}"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, device=device), atol=1e-5), "Probabilities do not sum to 1.0"

        # Check target layer for Grad-CAM
        target_layer = get_model_target_layer(model)
        print(f"    - Grad-CAM Target Layer:      {target_layer.__class__.__name__}")
        assert target_layer is not None, f"Target layer resolution failed for {model_name}"

    print("\n" + "=" * 65)
    print(" STAGE 2 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_stage2()
