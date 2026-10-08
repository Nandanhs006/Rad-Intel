"""
Training pipeline and optimizer engine for Rad-Intel deep learning architectures.
Supports fast feature-cached training for CPU and end-to-end fine-tuning for GPU.
Implements Early Stopping, Cosine Annealing, Focal Loss, and Checkpointing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from rad_intel.config import settings
from rad_intel.models.factory import create_model
from rad_intel.models.hybrid import HybridDenseNetSwinCBAM
from rad_intel.models.swin import SwinTransformerBaseline
from rad_intel.models.factory import ResNet50Baseline
from rad_intel.models.densenet import DenseNet121Baseline
from rad_intel.training.dataset import (
    CXRDataset,
    get_transforms,
    load_dataset_splits,
)
from rad_intel.training.losses import get_loss_function
from rad_intel.training.metrics import EvaluationMetrics, compute_metrics


@dataclass
class TrainingConfig:
    model_type: str = "hybrid"  # "hybrid" or "densenet121"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_type: str = "focal"  # "focal" or "weighted_ce"
    focal_gamma: float = 2.0
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("weights")
    checkpoint_name: str | None = None
    val_split: float = 0.15
    seed: int = 42
    use_feature_cache: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    # Phase 2: end-to-end fine-tuning. train_cached() alone freezes both
    # backbones, which measures head capacity rather than architecture quality:
    # the hybrid's ~950K-parameter head reaches 100% train accuracy on frozen
    # features while DenseNet's ~2K-parameter head cannot overfit, so the
    # comparison inverts. Section IV-A of the paper specifies two-phase
    # training, which this implements.
    finetune_epochs: int = 10
    finetune_lr_head: float = 1e-4
    finetune_lr_backbone: float = 1e-5
    data_dir: Path | str | None = None


def stratified_subsample(
    samples: list[tuple[Path, int]], max_n: int, seed: int = 42
) -> list[tuple[Path, int]]:
    """Subsamples a dataset while strictly preserving the class distribution ratio."""
    if len(samples) <= max_n:
        return samples
    labels = [l for _, l in samples]
    sub_paths, _, sub_labels, _ = train_test_split(
        [p for p, _ in samples],
        labels,
        train_size=max_n,
        stratify=labels,
        random_state=seed,
    )
    return list(zip(sub_paths, sub_labels))


class HybridTrainer:
    """
    Manages end-to-end training and evaluation of Rad-Intel models (Hybrid and DenseNet121).
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.checkpoint_name:
            # Name the checkpoint after the architecture that produced it.
            # Defaulting every non-DenseNet run to "best_hybrid_model.pt" made
            # a Swin or ResNet checkpoint masquerade as the hybrid.
            self.checkpoint_name = f"best_{self.config.model_type}_model.pt"
        else:
            self.checkpoint_name = self.config.checkpoint_name

        self.checkpoint_path = self.checkpoint_dir / self.checkpoint_name

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def extract_features(
        self,
        model: nn.Module,
        dataset: CXRDataset,
        batch_size: int = 16,
        desc: str = "Extracting features",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passes images through frozen extractors to obtain intermediate feature representations.
        For Hybrid: (B, 1792, 7, 7) spatial feature map.
        For DenseNet121: (B, 1024, 7, 7) spatial feature map.
        """
        model.eval()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        all_feats: list[torch.Tensor] = []
        all_labels: list[int] = []

        total = len(dataset)
        processed = 0
        start_t = time.time()

        print(f"[{desc}] Starting on {total} images (device: {self.device})...", flush=True)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device)

                if isinstance(model, (DenseNet121Baseline, SwinTransformerBaseline)):
                    # Both expose .extractor -> (B, C, 7, 7); C is 1024 / 768.
                    feat = model.extractor(images)
                elif isinstance(model, ResNet50Baseline):
                    # Everything up to, but excluding, avgpool + fc.
                    r = model.model
                    x = r.conv1(images); x = r.bn1(x); x = r.relu(x); x = r.maxpool(x)
                    x = r.layer1(x); x = r.layer2(x); x = r.layer3(x)
                    feat = r.layer4(x)  # (B, 2048, 7, 7)
                elif isinstance(model, HybridDenseNetSwinCBAM):
                    local_feat = model.densenet_branch(images)
                    global_feat = model.swin_branch(images)
                    feat = torch.cat([local_feat, global_feat], dim=1)  # (B, 1792, 7, 7)
                else:
                    raise ValueError(f"Unsupported model type for extraction: {type(model)}")

                all_feats.append(feat.cpu())
                all_labels.extend(labels.tolist())

                processed += images.size(0)
                if (batch_idx + 1) % 10 == 0 or processed == total:
                    elapsed = time.time() - start_t
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(
                        f"  [{desc}] {processed}/{total} images "
                        f"({processed/total*100:.1f}%) - {rate:.1f} img/s",
                        flush=True,
                    )

        feature_tensor = torch.cat(all_feats, dim=0)
        label_tensor = torch.tensor(all_labels, dtype=torch.long)
        return feature_tensor, label_tensor

    # ------------------------------------------------------------------
    # Phase 2: end-to-end fine-tuning
    # ------------------------------------------------------------------
    def _run_epoch_full(self, model, loader, criterion, optimizer=None):
        """One pass over real images through the whole network."""
        train = optimizer is not None
        model.train(train)
        total_loss, probs, labels = 0.0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, lbls in loader:
                images, lbls = images.to(self.device), lbls.to(self.device)
                if train:
                    optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, lbls)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                probs.extend(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().tolist())
                labels.extend(lbls.cpu().tolist())
        return total_loss / max(len(labels), 1), np.array(labels), np.array(probs)

    def finetune(self, model, train_ds, val_ds, test_ds, class_weights):
        """
        Unfreeze everything and fine-tune end to end with discriminative
        learning rates: a higher rate for the newly initialised head, a lower
        one for the pretrained backbones so their features are not destroyed.
        """
        print(f"\n[PHASE 2] End-to-end fine-tuning for {self.config.finetune_epochs} epochs...", flush=True)

        for p in model.parameters():
            p.requires_grad = True

        # Split parameters: anything in a pretrained backbone gets the low rate.
        backbone_markers = ("densenet", "swin", "extractor", "model.conv1", "model.bn1", "model.layer")
        backbone, head = [], []
        for name, param in model.named_parameters():
            (backbone if any(m in name for m in backbone_markers) else head).append(param)
        print(f"  backbone tensors: {len(backbone)} @ lr={self.config.finetune_lr_backbone}")
        print(f"  head tensors:     {len(head)} @ lr={self.config.finetune_lr_head}")

        criterion = get_loss_function(
            loss_type=self.config.loss_type,
            class_weights=class_weights.to(self.device),
            gamma=self.config.focal_gamma,
        )
        optimizer = AdamW(
            [
                {"params": backbone, "lr": self.config.finetune_lr_backbone},
                {"params": head, "lr": self.config.finetune_lr_head},
            ],
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.finetune_epochs, eta_min=1e-7)

        # Augmentation is applied only in phase 2; phase 1 reads cached features
        # from deterministic transforms and cannot be augmented.
        train_loader = DataLoader(
            CXRDataset(train_ds.samples, transform=get_transforms(is_train=True)),
            batch_size=self.config.batch_size, shuffle=True, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=2)

        best_auc, best_state, best_epoch, patience = -1.0, None, 0, 0
        for epoch in range(1, self.config.finetune_epochs + 1):
            tr_loss, tr_y, tr_p = self._run_epoch_full(model, train_loader, criterion, optimizer)
            va_loss, va_y, va_p = self._run_epoch_full(model, val_loader, criterion)
            scheduler.step()
            tr_m, va_m = compute_metrics(tr_y, tr_p), compute_metrics(va_y, va_p)
            print(
                f"  FT Epoch {epoch:02d}/{self.config.finetune_epochs} | "
                f"Train Loss: {tr_loss:.4f}, Acc: {tr_m.accuracy*100:.2f}% | "
                f"Val Loss: {va_loss:.4f}, Acc: {va_m.accuracy*100:.2f}%, AUC: {va_m.auc_roc:.4f}, "
                f"Sens: {va_m.sensitivity*100:.1f}%, Spec: {va_m.specificity*100:.1f}%",
                flush=True,
            )
            if va_m.auc_roc > best_auc:
                best_auc, best_epoch, patience = va_m.auc_roc, epoch, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.config.patience:
                    print(f"  [Early Stopping] No improvement for {self.config.patience} epochs.")
                    break

        print(f"\n[PHASE 2] Best validation AUC {best_auc:.4f} at fine-tune epoch {best_epoch}")
        if best_state:
            model.load_state_dict(best_state)
        return model

    def _save_checkpoint(self, model, final_test_metrics, best_epoch, class_weights):
        """Persist the trained model plus the metadata serving depends on."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_type": self.config.model_type,
            "epoch": best_epoch,
            "metrics": final_test_metrics.to_dict(),
            "optimal_threshold": final_test_metrics.optimal_threshold,
            "class_weights": class_weights.tolist(),
            "timestamp": time.time(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"\n[SUCCESS] Checkpoint saved successfully to: {self.checkpoint_path}")
        if self.config.model_type == "hybrid":
            torch.save(checkpoint, self.checkpoint_dir / "best_hybrid_model.pt")

    def train_cached(self) -> tuple[nn.Module, EvaluationMetrics]:
        """
        Fast two-stage training for CPU / local environments:
        1. Pretrained backbone(s) extract feature maps once.
        2. Attention / classification head trained with CosineAnnealing & Focal Loss.
        3. Full end-to-end model is saved as a single unified checkpoint.
        """
        print("\n" + "=" * 60, flush=True)
        print(f" RAD-INTEL TRAINING: {self.config.model_type.upper()} Pretrained Feature Cache", flush=True)
        print("=" * 60, flush=True)

        # 1. Load dataset splits
        train_samples, val_samples, test_samples, class_weights = load_dataset_splits(
            data_dir=self.config.data_dir,
            val_split=self.config.val_split,
            seed=self.config.seed,
        )

        if self.config.max_train_samples and len(train_samples) > self.config.max_train_samples:
            print(f"[INFO] Stratified subsampling training set to {self.config.max_train_samples} samples.", flush=True)
            train_samples = stratified_subsample(train_samples, self.config.max_train_samples, seed=self.config.seed)

        if self.config.max_val_samples and len(val_samples) > self.config.max_val_samples:
            print(f"[INFO] Stratified subsampling validation set to {self.config.max_val_samples} samples.", flush=True)
            val_samples = stratified_subsample(val_samples, self.config.max_val_samples, seed=self.config.seed)

        if self.config.max_test_samples and len(test_samples) > self.config.max_test_samples:
            print(f"[INFO] Stratified subsampling test set to {self.config.max_test_samples} samples.", flush=True)
            test_samples = stratified_subsample(test_samples, self.config.max_test_samples, seed=self.config.seed)

        print(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        print(f"Class Weights (Normal vs Pneumonia): {class_weights.tolist()}")

        # 2. Build model with ImageNet pretrained backbones
        print(f"\n[1/4] Initializing pretrained {self.config.model_type} architecture...")
        model = create_model(self.config.model_type, pretrained=True).to(self.device)  # type: ignore

        # 3. Setup datasets
        train_ds = CXRDataset(train_samples, transform=get_transforms(is_train=False))
        val_ds = CXRDataset(val_samples, transform=get_transforms(is_train=False))
        test_ds = CXRDataset(test_samples, transform=get_transforms(is_train=False))

        # 4. Feature caching
        cache_dir = Path(".cache_features")
        cache_dir.mkdir(exist_ok=True)
        prefix = f"{self.config.model_type}"
        train_cache_path = cache_dir / f"{prefix}_train_{len(train_samples)}.pt"
        val_cache_path = cache_dir / f"{prefix}_val_{len(val_samples)}.pt"
        test_cache_path = cache_dir / f"{prefix}_test_{len(test_samples)}.pt"

        print(f"\n[2/4] Extracting / loading feature maps for {self.config.model_type}...")
        if train_cache_path.exists():
            print(f"  Loading cached training features from {train_cache_path}...")
            train_feats, train_labels = torch.load(train_cache_path, weights_only=True)
        else:
            train_feats, train_labels = self.extract_features(
                model, train_ds, batch_size=self.config.batch_size, desc="Train Features"
            )
            torch.save((train_feats, train_labels), train_cache_path)

        if val_cache_path.exists():
            print(f"  Loading cached validation features from {val_cache_path}...")
            val_feats, val_labels = torch.load(val_cache_path, weights_only=True)
        else:
            val_feats, val_labels = self.extract_features(
                model, val_ds, batch_size=self.config.batch_size, desc="Val Features"
            )
            torch.save((val_feats, val_labels), val_cache_path)

        train_loader = DataLoader(
            TensorDataset(train_feats, train_labels),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_feats, val_labels),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # 5. Define head forward pass and trainable parameters
        if isinstance(model, (DenseNet121Baseline, SwinTransformerBaseline)):
            head_params = list(model.classifier.parameters())

            def forward_head(feat: torch.Tensor) -> torch.Tensor:
                pooled = model.pool(feat).flatten(1)
                return model.classifier(pooled)

        elif isinstance(model, ResNet50Baseline):
            # torchvision ResNet keeps its head at .model.fc (Dropout + Linear).
            head_params = list(model.model.fc.parameters())

            def forward_head(feat: torch.Tensor) -> torch.Tensor:
                pooled = model.model.avgpool(feat).flatten(1)
                return model.model.fc(pooled)

        elif isinstance(model, HybridDenseNetSwinCBAM):
            head_params = (
                list(model.fusion_conv.parameters())
                + list(model.cbam.parameters())
                + list(model.classifier.parameters())
            )

            def forward_head(feat: torch.Tensor) -> torch.Tensor:
                projected = model.fusion_conv(feat)
                if model.use_cbam:
                    refined = model.cbam(projected)
                else:
                    refined = projected
                pooled = model.global_pool(refined).flatten(1)
                return model.classifier(pooled)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        # 6. Loss, Optimizer, Scheduler
        criterion = get_loss_function(
            loss_type=self.config.loss_type,
            class_weights=class_weights.to(self.device),
            gamma=self.config.focal_gamma,
        )
        optimizer = AdamW(head_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=1e-5)

        # 7. Training loop
        print(f"\n[3/4] Training Head Parameters for {self.config.epochs} epochs...")
        best_val_auc = 0.0
        best_metrics: EvaluationMetrics | None = None
        best_model_state: dict[str, Any] = {}
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            total_train_loss = 0.0
            correct = 0
            total_samples = 0

            for feats, labels in train_loader:
                feats = feats.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = forward_head(feats)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total_samples += len(labels)

            scheduler.step()
            train_acc = correct / total_samples
            avg_train_loss = total_train_loss / total_samples

            # Validation
            model.eval()
            val_probs: list[float] = []
            val_y: list[int] = []
            val_loss = 0.0

            with torch.no_grad():
                for feats, labels in val_loader:
                    feats = feats.to(self.device)
                    labels = labels.to(self.device)

                    logits = forward_head(feats)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)

                    probs = torch.softmax(logits, dim=1)[:, 1]
                    val_probs.extend(probs.cpu().tolist())
                    val_y.extend(labels.cpu().tolist())

            avg_val_loss = val_loss / len(val_y)
            val_metrics = compute_metrics(val_y, val_probs)

            print(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Acc: {val_metrics.accuracy*100:.2f}%, "
                f"AUC: {val_metrics.auc_roc:.4f}, Sens: {val_metrics.sensitivity*100:.1f}%, "
                f"Spec: {val_metrics.specificity*100:.1f}%"
            )

            if val_metrics.auc_roc > best_val_auc:
                best_val_auc = val_metrics.auc_roc
                best_metrics = val_metrics
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"  [Early Stopping] No improvement for {self.config.patience} epochs.")
                    break

        print(f"\n[INFO] Best Validation achieved at Epoch {best_epoch} (AUC: {best_val_auc:.4f})")

        if best_model_state:
            model.load_state_dict(best_model_state)

        # 7b. Phase 2: unfreeze and fine-tune end to end. Skipped when
        # finetune_epochs == 0, which reproduces the old frozen-backbone
        # behaviour.
        if self.config.finetune_epochs > 0:
            model = self.finetune(model, train_ds, val_ds, test_ds, class_weights)

        # 8. Evaluate on strictly held-out test images
        print(f"\n[4/4] Evaluating Best Model on Held-Out Test Set ({len(test_samples)} images)...")
        if self.config.finetune_epochs > 0:
            # The backbones changed during phase 2, so cached features are stale.
            test_loader = DataLoader(
                test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=2
            )
            _, test_y, test_p = self._run_epoch_full(
                model, test_loader,
                get_loss_function(
                    loss_type=self.config.loss_type,
                    class_weights=class_weights.to(self.device),
                    gamma=self.config.focal_gamma,
                ),
            )
            opt_thresh = best_metrics.optimal_threshold if best_metrics else 0.5
            final_test_metrics = compute_metrics(test_y, test_p, threshold=opt_thresh)
            print("\n" + final_test_metrics.summary_table(
                f"Final Test Set Results ({self.config.model_type.upper()})"))
            self._save_checkpoint(model, final_test_metrics, best_epoch, class_weights)
            return model, final_test_metrics

        if test_cache_path.exists():
            print(f"  Loading cached test features from {test_cache_path}...")
            test_feats, test_labels = torch.load(test_cache_path, weights_only=True)
        else:
            test_feats, test_labels = self.extract_features(
                model, test_ds, batch_size=self.config.batch_size, desc="Test Features"
            )
            torch.save((test_feats, test_labels), test_cache_path)

        test_loader = DataLoader(
            TensorDataset(test_feats, test_labels),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        model.eval()
        test_probs: list[float] = []
        test_y: list[int] = []

        with torch.no_grad():
            for feats, labels in test_loader:
                feats = feats.to(self.device)
                logits = forward_head(feats)
                probs = torch.softmax(logits, dim=1)[:, 1]
                test_probs.extend(probs.cpu().tolist())
                test_y.extend(labels.tolist())

        opt_thresh = best_metrics.optimal_threshold if best_metrics else 0.5
        final_test_metrics = compute_metrics(test_y, test_probs, threshold=opt_thresh)

        print("\n" + final_test_metrics.summary_table(f"Final Test Set Results ({self.config.model_type.upper()})"))

        # 9. Save complete model checkpoint
        self._save_checkpoint(model, final_test_metrics, best_epoch, class_weights)

        # Previously this also wrote best_hybrid_model.pt whenever that file was
        # absent, so training ANY architecture produced a file claiming to be the
        # hybrid. Only the hybrid writes the hybrid checkpoint.
        if self.config.model_type == "hybrid":
            torch.save(checkpoint, self.checkpoint_dir / "best_hybrid_model.pt")
            print(f"[INFO] Synced active default checkpoint: {self.checkpoint_dir / 'best_hybrid_model.pt'}")

        return model, final_test_metrics
