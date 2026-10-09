# Table IV source data — two-phase protocol (paper IV-A)

Protocol: 85/15 stratified split (seed 42), phase 1 = 8 epochs frozen-backbone
head training on cached features, phase 2 = up to 12 epochs end-to-end
fine-tuning (backbones 1e-5, head 1e-4, batch 12), focal loss gamma=2,
inverse-frequency class weights [1.9385, 0.6738], early stopping patience 5 on
validation AUROC. Test = held-out official 624-image folder.

| Model | Acc | AUROC | Sens | Spec | Prec | F1 | Bal acc | t* | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet121    | 91.51 | 0.9618 | 95.64 | 84.62 | 91.20 | 0.9337 | 0.9013 | 0.4628 | 198 | 36 | 17 | 373 |
| Swin-T         | 90.06 | 0.9749 | 97.44 | 77.78 | 87.96 | 0.9246 | 0.8761 | 0.4221 | 182 | 52 | 10 | 380 |
| ResNet50       | 89.90 | 0.9545 | 93.33 | 84.19 | 90.77 | 0.9204 | 0.8876 | 0.5569 | 197 | 37 | 26 | 364 |
| Fusion w/o CBAM| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | | | | |
| **Complete (hybrid)** | **91.99** | **0.9701** | **94.87** | **87.18** | **92.50** | **0.9367** | **0.9103** | 0.5826 | 204 | 30 | 20 | 370 |

Hybrid run time: 48.5 min on a Colab T4. Phase 2 best val AUROC 0.9976 at
fine-tune epoch 4; early stopping at epoch 9.

## Superseded: frozen-backbone protocol (NOT for the paper)

Kept only to document why the protocol changed. With both backbones frozen,
only the head trains, so the ranking tracked head size rather than
architecture: every hybrid reached 99-100% train accuracy on cached features
while DenseNet's ~2K-parameter head could not overfit. Under that protocol
CBAM appeared to *hurt* (82.21 vs 83.65), which inverted the ablation.

| Model | Acc | AUROC | Sens | Spec |
|---|---|---|---|---|
| DenseNet121 | 90.38 | 0.9472 | 93.59 | 85.04 |
| Swin-T | 79.33 | 0.9480 | 98.21 | 47.86 |
| ResNet50 | 86.38 | 0.9500 | 96.41 | 69.66 |
| Fusion w/o CBAM | 83.65 | 0.9548 | 98.97 | 58.12 |
| Complete (hybrid) | 82.21 | 0.9443 | 98.72 | 54.70 |

Effect of adding phase 2 to the hybrid: accuracy 82.21 -> 91.99,
AUROC 0.9443 -> 0.9701, specificity 54.70 -> 87.18.

## Separate: notebook two-phase full fine-tune (different code path)

The standalone Colab notebook trains its own model class built on timm's Swin
rather than torchvision's, so its checkpoint does not load into the repo
(8 of 924 tensors match). Reported here for reference only; not a Table IV row.

Accuracy 92.95 | AUROC 0.9788 | Sens 92.05 | Spec 94.44 | t* 0.6817
Confusion: TN 221, FP 13, FN 31, TP 359


## Reading the two-phase table

Run times: DenseNet 43.2 min, Swin-T 36.7 min, ResNet50 36.2 min, hybrid 48.5 min.

The complete classifier leads on accuracy (91.99), specificity (87.18),
precision (92.50), F1 (0.9367) and balanced accuracy (0.9103), but NOT on
AUROC: Swin-T ranks better (0.9749 vs 0.9701) while sitting at a far worse
operating point (specificity 77.78 vs 87.18). The honest claim is that fusion
buys a better-balanced decision, not better ranking.

CAUTION on margins. Hybrid 91.99 vs DenseNet121 91.51 is 574 vs 571 correct
out of 624 - a three-image difference. With one seed and no confidence
interval this is well inside sampling noise and must not be reported as a
decisive win. Specificity (87.18 vs 84.62, i.e. 204 vs 198 true negatives) and
balanced accuracy are the more defensible comparisons, and even those are
modest. Multiple seeds would be needed to claim significance.

Every baseline still shows the sensitivity/specificity asymmetry: all four
exceed 93% sensitivity while specificity ranges 77.8-87.2%, consistent with
the 2.9:1 class ratio in the development set.
