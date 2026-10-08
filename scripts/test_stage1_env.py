r"""
Stage 1 Verification Script: Environment, Dependencies & Configuration.
Run with: .\.venv\Scripts\python.exe scripts/test_stage1_env.py
"""

import sys
import importlib

def test_stage1():
    print("=" * 60)
    print(" RAD-INTEL :: STAGE 1 VERIFICATION (Environment & Config)")
    print("=" * 60)

    # 1. Check Python Version
    py_ver = sys.version.split()[0]
    print(f"[1] Python Runtime: {py_ver}")
    assert sys.version_info >= (3, 14), f"Expected Python >= 3.14, got {py_ver}"
    print("    [PASSED] Python version is 3.14+")

    # 2. Check Core Dependencies
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "PyTorch Image Models (timm)"),
        ("pytorch_grad_cam", "Grad-CAM"),
        ("lime", "LIME"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("pydantic_settings", "Pydantic Settings"),
        ("google.genai", "Google GenAI SDK"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-Learn"),
        ("aiosqlite", "AioSQLite"),
    ]

    print("\n[2] Checking Required Packages:")
    all_packages_ok = True
    for module_name, label in required_packages:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    - {label:<32}: OK (version {ver})")
        except ImportError as e:
            print(f"    - {label:<32}: FAILED ({e})")
            all_packages_ok = False

    assert all_packages_ok, "One or more required packages are missing."

    # 3. Check Hardware & PyTorch Acceleration
    import torch
    print("\n[3] PyTorch Hardware Acceleration:")
    print(f"    - PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"    - CUDA available: {cuda_avail}")
    if cuda_avail:
        print(f"    - GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("    - Computing on CPU (Expected for local environments without NVIDIA GPU)")

    # 4. Check Rad-Intel Settings & Config
    print("\n[4] Rad-Intel Configuration Module:")
    from rad_intel.config import settings
    print(f"    - Project Name: {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"    - Target Device: {settings.torch_device}")
    print(f"    - Default Model: {settings.DEFAULT_MODEL}")
    print(f"    - Class Names: {settings.CLASS_NAMES}")
    print(f"    - Gemini Model: {settings.GEMINI_MODEL}")
    print(f"    - Gemini API Key configured: {'YES' if settings.GEMINI_API_KEY else 'NO (offline fallback mode will be used)'}")

    print("\n" + "=" * 60)
    print(" STAGE 1 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_stage1()
