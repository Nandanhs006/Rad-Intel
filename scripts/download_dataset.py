"""
Helper script to download the Kaggle Chest X-Ray Pneumonia dataset.
Run with: .\.venv\Scripts\python.exe scripts/download_dataset.py
"""

import sys
import kagglehub

def main():
    print("=" * 60)
    print(" Downloading 'paultimothymooney/chest-xray-pneumonia'...")
    print(" Note: This dataset is approximately 1.15 GB.")
    print("=" * 60)

    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("\n[SUCCESS] Dataset downloaded successfully!")
        print("Path to dataset files:", path)
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
