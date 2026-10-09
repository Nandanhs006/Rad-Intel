# Table IV source data — two-phase protocol (paper IV-A)

Protocol: 85/15 stratified split (seed 42), phase 1 = 8 epochs frozen-backbone
head training on cached features, phase 2 = up to 12 epochs end-to-end
fine-tuning (backbones 1e-5, head 1e-4, batch 12), focal loss gamma=2,
inverse-frequency class weights [1.9385, 0.6738], early stopping patience 5 on
validation AUROC. Test = held-out official 624-image folder.

| Model | Acc | AUROC | Sens | Spec | Prec | F1 | Bal acc | t* | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet121    | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | | | | |
| Swin-T         | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | | | | |
| ResNet50       | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | | | | |
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
