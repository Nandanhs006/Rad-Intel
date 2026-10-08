"""
High-Speed Direct Batch Predictor for Rad-Intel.
Evaluates thousands of chest radiographs directly through the trained DenseNet121 / Hybrid PyTorch
model with OpenCV CLAHE preprocessing, bypassing browser and HTTP overhead.

Outputs results into the exact CSV format expected by crawler.py:
image,verdict
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rad_intel.api.dependencies import ModelManager
from rad_intel.config import settings
from rad_intel.preprocessing.transforms import default_preprocessor


class RadiographDataset(Dataset):
    """Loads and standardizes chest X-ray images with CLAHE enhancement."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        self.preprocessor = default_preprocessor

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        filepath = os.path.join(self.folder_path, filename)
        try:
            tensor, _ = self.preprocessor.preprocess(filepath, device="cpu")
            return filename, tensor.squeeze(0), True
        except Exception as e:
            # Fallback zero tensor if image is corrupted
            zero_tensor = torch.zeros((3, settings.IMAGE_SIZE, settings.IMAGE_SIZE), dtype=torch.float32)
            return filename, zero_tensor, False


def run_batch_prediction(
    input_dir: str,
    output_csv: str,
    model_name: str = "densenet121",
    batch_size: int = 32,
    num_workers: int = 2,
):
    print("=" * 70)
    print("Rad-Intel High-Speed Batch Inference")
    print(f"Target Directory : {input_dir}")
    print(f"Output CSV       : {output_csv}")
    print(f"Model Arch       : {model_name}")
    print(f"Batch Size       : {batch_size}")
    print("=" * 70)

    start_time = time.perf_counter()

    # Initialize model manager and load trained weights
    manager = ModelManager()
    model = manager.get_model(model_name)
    model.eval()
    device = manager.device
    print(f"[OK] Model loaded on {device}: {model_name}")

    dataset = RadiographDataset(input_dir)
    total_images = len(dataset)
    print(f"[OK] Discovered {total_images} images to evaluate.")

    if total_images == 0:
        print("No valid images found in directory. Exiting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    normal_count = 0
    pneumonia_count = 0
    processed_count = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "verdict"])
        f.flush()

        with torch.inference_mode():
            for batch_files, batch_tensors, valid_mask in loader:
                batch_tensors = batch_tensors.to(device)
                logits = model(batch_tensors)
                probs = torch.softmax(logits, dim=1)
                pred_indices = torch.argmax(probs, dim=1)

                for fname, p_idx, is_valid in zip(batch_files, pred_indices, valid_mask):
                    if not is_valid:
                        verdict = "ERROR_READING_IMAGE"
                    elif p_idx.item() == 1:
                        verdict = "PNEUMONIA DETECTED"
                        pneumonia_count += 1
                    else:
                        verdict = "NO PNEUMONIA DETECTED (NORMAL)"
                        normal_count += 1

                    writer.writerow([fname, verdict])
                    processed_count += 1

                f.flush()

                elapsed = time.perf_counter() - start_time
                fps = processed_count / elapsed if elapsed > 0 else 0
                pct = (processed_count / total_images) * 100
                print(
                    f"\rProgress: [{processed_count}/{total_images}] ({pct:5.1f}%) | "
                    f"Speed: {fps:4.1f} img/s | Normal: {normal_count} | Pneumonia: {pneumonia_count}",
                    end="",
                    flush=True,
                )

    total_time = time.perf_counter() - start_time
    print()
    print("=" * 70)
    print("Batch Evaluation Completed Successfully!")
    print(f"Total Images Evaluated : {processed_count}")
    print(f"Total Time Elapsed     : {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Average Throughput     : {processed_count / total_time:.1f} images/second")
    print(f"Verdict Summary        : Normal: {normal_count} | Pneumonia: {pneumonia_count}")
    print(f"Results File           : {output_csv}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Rad-Intel Batch Evaluation")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\digitals\capstone\sample\normal1",
        help="Path to folder containing images",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=r"C:\digitals\capstone\test\results3.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="densenet121",
        help="Model architecture (densenet121, hybrid, resnet50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default: 2)",
    )

    args = parser.parse_args()
    run_batch_prediction(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
