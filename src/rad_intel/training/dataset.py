"""
Dataset loader and medical augmentations for Chest X-Ray pneumonia classification.
Implements stratified splitting, CLAHE enhancement, class weighting, and PyTorch DataLoaders.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from rad_intel.config import settings
from rad_intel.preprocessing.transforms import CXRPreprocessor


def find_dataset_dir(custom_path: str | Path | None = None) -> Path:
    """Finds the Kaggle chest_xray dataset directory across common cache paths."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))

    # Check settings/environment if configured
    candidates.append(Path(settings.DATASET_DIR) if hasattr(settings, "DATASET_DIR") else None)

    # Check kagglehub cache directory
    kagglehub_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "paultimothymooney" / "chest-xray-pneumonia" / "versions" / "2" / "chest_xray"
    candidates.append(kagglehub_cache)

    # Check local data folder
    candidates.append(Path("data") / "chest_xray")
    candidates.append(Path("chest_xray"))

    for c in candidates:
        if c is not None and c.exists() and (c / "train").exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not find Kaggle chest_xray dataset directory. "
        "Please specify data_dir or run 'python scripts/download_dataset.py'."
    )


def apply_clahe_to_pil(img: Image.Image) -> Image.Image:
    """Applies CLAHE on the luminance channel of an RGB PIL Image."""
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_chan)
    enhanced = cv2.merge((cl, a_chan, b_chan))
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def get_transforms(
    image_size: int = settings.IMAGE_SIZE,
    is_train: bool = True,
    use_clahe: bool = settings.APPLY_CLAHE,
) -> Callable:
    """Builds PyTorch data transform pipeline with medical augmentations."""
    transform_list = []

    if use_clahe:
        transform_list.append(T.Lambda(apply_clahe_to_pil))

    if is_train:
        transform_list.extend([
            T.Resize((image_size + 24, image_size + 24)),
            T.RandomCrop((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=settings.NORM_MEAN, std=settings.NORM_STD),
        ])
    else:
        transform_list.extend([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=settings.NORM_MEAN, std=settings.NORM_STD),
        ])

    return T.Compose(transform_list)


class CXRDataset(Dataset):
    """PyTorch Dataset for Chest Radiographs with labels (0=Normal, 1=Pneumonia)."""

    def __init__(
        self,
        samples: list[tuple[Path | str, int]],
        transform: Callable | None = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            # Fallback OpenCV read
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                raise RuntimeError(f"Error opening image {img_path}: {e}")
            img_rgb = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            tensor = T.ToTensor()(img_rgb)

        return tensor, label


def load_dataset_splits(
    data_dir: Path | str | None = None,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]], torch.Tensor]:
    """
    Loads samples from train, val, and test subdirectories.
    Re-splits the 5,232 images in train+val into a stratified 85/15 train/val split.
    Keeps the 624 test images strictly separate for final evaluation.
    Computes inverse frequency class weights to counteract class imbalance.

    Returns:
        (train_samples, val_samples, test_samples, class_weights_tensor)
    """
    root = find_dataset_dir(data_dir)

    def collect_from_folder(folder_name: str) -> list[tuple[Path, int]]:
        items: list[tuple[Path, int]] = []
        normal_dir = root / folder_name / "NORMAL"
        pneumonia_dir = root / folder_name / "PNEUMONIA"

        if normal_dir.exists():
            for p in normal_dir.iterdir():
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    items.append((p, 0))

        if pneumonia_dir.exists():
            for p in pneumonia_dir.iterdir():
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    items.append((p, 1))

        return items

    # Collect raw Kaggle splits
    raw_train = collect_from_folder("train")
    raw_val = collect_from_folder("val")
    test_samples = collect_from_folder("test")

    # Combine train and Kaggle's 16-image val to create a proper 85/15 stratified split
    combined_pool = raw_train + raw_val
    paths = [p for p, _ in combined_pool]
    labels = [l for _, l in combined_pool]

    train_paths, val_paths, train_y, val_y = train_test_split(
        paths,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    train_samples = list(zip(train_paths, train_y))
    val_samples = list(zip(val_paths, val_y))

    # Calculate inverse class frequency weights
    count_0 = sum(1 for _, y in train_samples if y == 0)
    count_1 = sum(1 for _, y in train_samples if y == 1)
    total = count_0 + count_1

    # Standard balanced weighting: total / (n_classes * count)
    weight_0 = total / (2.0 * count_0) if count_0 > 0 else 1.0
    weight_1 = total / (2.0 * count_1) if count_1 > 0 else 1.0
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)

    return train_samples, val_samples, test_samples, class_weights


def get_dataloaders(
    data_dir: Path | str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.15,
    seed: int = 42,
    use_clahe: bool = settings.APPLY_CLAHE,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Builds ready-to-use PyTorch DataLoaders for train, val, and test splits."""
    train_samples, val_samples, test_samples, class_weights = load_dataset_splits(
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
    )

    train_transform = get_transforms(is_train=True, use_clahe=use_clahe)
    eval_transform = get_transforms(is_train=False, use_clahe=use_clahe)

    train_dataset = CXRDataset(train_samples, transform=train_transform)
    val_dataset = CXRDataset(val_samples, transform=eval_transform)
    test_dataset = CXRDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, class_weights
