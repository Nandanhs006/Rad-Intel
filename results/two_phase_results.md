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
| Fusion w/o CBAM| 93.43 | 0.9765 | 94.62 | 91.45 | 94.86 | 0.9474 | 0.9303 | 0.7169 | 214 | 20 | 21 | 369 |
| **Complete (hybrid)** | 91.99 | 0.9736 | 95.90 | 85.47 | 91.67 | 0.9373 | 0.9069 | 0.5574 | 200 | 34 | 16 | 374 |

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


## Paired statistical comparison (added after full evaluation)

Computed from per-image predictions on the same 624 test images, so valid
from single runs: McNemar compares two classifiers on identical inputs.

### Thresholded accuracy - McNemar, Holm-adjusted over 3 comparisons

| Comparison | A-only-right | B-only-right | p | Holm p | Verdict |
|---|---|---|---|---|---|
| DenseNet121 vs Swin-T   | 28 | 19 | 0.243 | 0.531 | not significant |
| DenseNet121 vs ResNet50 | 33 | 22 | 0.177 | 0.531 | not significant |
| Swin-T vs ResNet50      | 27 | 25 | 0.890 | 0.890 | not significant |

No pair differs significantly in accuracy. The spread 91.51 / 90.06 / 89.74
is NOT a ranking.

### AUROC - paired bootstrap, 2000 resamples

| Comparison | dAUROC [95% CI] | Verdict |
|---|---|---|
| DenseNet121 - Swin-T   | -0.0133 [-0.0222, -0.0055] | SIGNIFICANT (Swin-T better) |
| DenseNet121 - ResNet50 | +0.0076 [-0.0018, +0.0176] | not significant |
| Swin-T - ResNet50      | +0.0206 [+0.0105, +0.0315] | SIGNIFICANT (Swin-T better) |

Swin-T ranks significantly better than both CNNs while having the worst
specificity (0.778). Discrimination and decision quality separate cleanly:
the configurations are statistically indistinguishable at their operating
points, and only the ranking differs.


## FINAL — all five configurations, two-phase protocol

| Model | Acc | AUROC | AUPRC | Sens | Spec | F1 | BalAcc | ECE |
|---|---|---|---|---|---|---|---|---|
| DenseNet121 | 91.51 | 0.962 | 0.969 | 0.956 | 0.846 | 0.934 | 0.901 | 0.052 |
| Swin-T | 90.06 | 0.975 | 0.983 | 0.974 | 0.778 | 0.925 | 0.876 | 0.054 |
| ResNet50 | 89.74 | 0.954 | 0.966 | 0.933 | 0.838 | 0.919 | 0.885 | 0.075 |
| Fusion w/o CBAM | **93.43** | **0.977** | 0.981 | 0.946 | **0.914** | **0.947** | **0.930** | 0.078 |
| Complete (CBAM) | 91.99 | 0.974 | 0.972 | **0.959** | 0.855 | 0.937 | 0.907 | 0.062 |

### Complete classifier vs each (McNemar + paired bootstrap, Holm over 4)

| vs | dAUROC [95% CI] | McNemar p | Holm p | b / c | Verdict |
|---|---|---|---|---|---|
| DenseNet121 | +0.012 [+0.005,+0.020] | 0.728 | 0.728 | 18/15 | AUROC sig.; accuracy tied |
| Swin-T | -0.001 [-0.006,+0.003] | 0.036 | 0.143 | 20/8 | ns after correction |
| ResNet50 | +0.019 [+0.009,+0.031] | 0.065 | 0.195 | 32/18 | AUROC sig.; accuracy ns |
| Fusion w/o CBAM | -0.003 [-0.006,+0.001] | 0.122 | **0.244** | 9/18 | **CBAM: no measurable effect** |

CONCLUSION: fusion is supported (both fused configs beat all baselines;
AUROC significantly higher than DenseNet121 and ResNet50). Post-fusion CBAM
is NOT supported - no significant difference, point estimates favour the
ablated model. Same direction observed under the earlier frozen protocol.
