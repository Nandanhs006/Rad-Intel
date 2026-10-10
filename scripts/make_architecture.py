"""
System architecture figure.

Same left-to-right stage layout as the original diagram, corrected to match
the implementation: CLAHE and a direct resize in preprocessing, concatenation
before projection in fusion, a two-logit softmax head at a calibrated
threshold, explanations confined to segmented lung fields, and Gemini 3.5
Flash-Lite as the summarizer.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg

C = {"grey":("#7c828c","#f3f4f6"), "green":("#3f7d50","#eef6f1"),
     "blue":("#2f5fa8","#eef2fb"), "teal":("#1f7a76","#eaf6f5"),
     "purple":("#6b4fb0","#f2eefb"), "orange":("#c2771a","#fdf4e6"),
     "slate":("#44506b","#eef0f6")}

fig, ax = plt.subplots(figsize=(19.0, 7.4))
ax.set_xlim(0, 190); ax.set_ylim(0, 74); ax.axis("off")
fig.patch.set_facecolor("white")

TOP, BOT = 58.0, 10.0           # every group box spans this band
MID = (TOP + BOT) / 2
stages = []                     # (x0, x1) per stage, for the flow arrows


def group(x0, w, title, colour):
    edge, face = C[colour]
    ax.add_patch(FancyBboxPatch((x0, BOT), w, TOP - BOT,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                ec=edge, fc=face, lw=1.6, zorder=1))
    ax.text(x0 + w / 2, TOP + 3.4, title, ha="center", va="center",
            fontsize=11, color=edge, fontweight="bold", zorder=3)
    stages.append((x0, x0 + w))
    return x0 + w / 2


def sub(cx, w, ytop, h, lines, colour, fs=8.6, bold0=False):
    edge, _ = C[colour]
    ax.add_patch(FancyBboxPatch((cx - w / 2, ytop - h), w, h,
                                boxstyle="round,pad=0.25,rounding_size=0.7",
                                ec=edge, fc="white", lw=1.0, zorder=2))
    n = len(lines)
    for i, t in enumerate(lines):
        ax.text(cx, ytop - h / (n + 1) * (i + 1), t, ha="center", va="center",
                fontsize=fs if i == 0 else fs - 1.0,
                color="#17171b" if i == 0 else "#5a5a64",
                fontweight="bold" if (bold0 and i == 0) else "normal", zorder=3)


def arrow(x1, y1, x2, y2, lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=lw, color="#2b2b30", zorder=4))


def stack(cx, w, items, colour, top=52.0, bottom=13.0, fs=8.6, bold0=False):
    """Evenly distribute sub-boxes down the group box."""
    n = len(items)
    gap = 2.6
    h = (top - bottom - gap * (n - 1)) / n
    for i, lines in enumerate(items):
        sub(cx, w, top - i * (h + gap), h, lines, colour, fs=fs, bold0=bold0)


# 1 Input
cx = group(2, 17, "Input", "grey")
ax.text(cx, 53.5, "Chest X-Ray", ha="center", fontsize=9, color="#17171b")
try:
    ax.imshow(mpimg.imread("sample_images/sample_pneumonia_person100_bacteria_478.jpeg"),
              extent=[cx - 6.6, cx + 6.6, 24, 50], aspect="auto", zorder=2, cmap="gray")
except Exception:
    pass

# 2 Preprocessing
cx = group(24, 20, "Preprocessing", "green")
stack(cx, 17, [["CLAHE", "clip 2.0, 8x8 tiles"], ["Resize 224 x 224", "no padding"],
               ["ImageNet", "Normalization"], ["Training", "Augmentation"]], "green")

# 3 Parallel feature extraction
cx3 = group(50, 23, "Parallel Feature Extraction", "blue")
sub(cx3, 20, 50.0, 15.0, ["DenseNet121", "Local features", "1024 x 7 x 7"], "blue", bold0=True)
sub(cx3, 20, 29.0, 15.0, ["Swin-T", "Global context", "768 x 7 x 7"], "blue", bold0=True)

# 4 Feature fusion
cx = group(79, 21, "Feature Fusion", "blue")
sub(cx, 18, 47.0, 13.0, ["Channel Concatenation", "1792 x 7 x 7"], "blue", fs=8.2)
sub(cx, 18, 30.0, 13.0, ["1x1 Conv - BN - ReLU", "512 x 7 x 7"], "blue")

# 5 CBAM
cx5 = group(106, 17, "CBAM Refinement", "green")
sub(cx5, 14, 45.0, 11.0, ["Channel", "Attention"], "green")
sub(cx5, 14, 30.0, 11.0, ["Spatial", "Attention"], "green")

# 6 Classifier
cx6 = group(129, 19, "Classifier", "teal")
stack(cx6, 16, [["Global Average", "Pooling"], ["Dropout (0.3)", "+ Linear"],
                ["Two Logits", "+ Softmax"], ["Calibrated Threshold", "t* (Youden)"]], "teal")

# 7 Explainability
cx7 = group(154, 23, "Explainability", "purple")
sub(cx7, 20, 52.0, 8.0, ["Lung-Field Segmentation"], "purple", fs=8.4)
sub(cx7 - 5.0, 9.4, 41.0, 25.0, ["Grad-CAM", "Post-CBAM", "saliency map"], "purple", fs=8.2)
sub(cx7 + 5.0, 9.4, 41.0, 25.0, ["LIME", "Superpixel", "weights"], "purple", fs=8.2)

# 8 Evidence serializer
cx = group(183, 21, "Evidence Serializer", "orange")
stack(cx, 18, [["Label + Probability", "+ Threshold"],
               ["Quadrant Scores", "+ Off-Lung Fraction"],
               ["Version and", "Checkpoint IDs"]], "orange", fs=8.3)

# 9 Constrained summary
cx9 = group(210, 23, "Constrained Summary", "orange")
sub(cx9, 19, 54.0, 7.0, ["Gemini 3.5 Flash-Lite"], "orange", fs=8.3)
sub(cx9, 19, 45.2, 7.0, ["Fixed JSON Schema"], "orange", fs=8.3)
sub(cx9, 19, 36.4, 7.0, ["Validation"], "orange", fs=8.3)
sub(cx9, 19, 27.6, 7.0, ["One Repair Attempt"], "orange", fs=8.0)
sub(cx9, 19, 18.8, 7.0, ["Template Fallback"], "orange", fs=8.0)
for _y in (47.0, 38.2, 29.4, 20.6):
    arrow(cx9, _y, cx9, _y - 2.2, lw=1.0)
ax.text(cx9 + 11.2, 31.0, "invalid", ha="center", fontsize=7.0, color="#9a6a1e")

# 10 Interface
cx10 = group(239, 20, "Research Interface", "slate")
stack(cx10, 17, [["Probability + Threshold"], ["Grad-CAM and LIME"],
                 ["Evidence Summary"], ["Limitations"]], "slate", top=52.0, bottom=22.0, fs=7.4)
ax.add_patch(FancyBboxPatch((cx10 - 8.5, 13.5), 17, 6.2,
                            boxstyle="round,pad=0.25,rounding_size=0.7",
                            ec="#1b2541", fc="#1b2541", lw=1.0, zorder=2))
ax.text(cx10, 16.6, "Research Use Only", ha="center", va="center",
        fontsize=8.4, color="white", fontweight="bold", zorder=3)

ax.set_xlim(-2, 262)

# flow arrows between consecutive stages
for (_, a), (b, _) in zip(stages, stages[1:]):
    arrow(a + 0.6, MID, b - 0.6, MID)
# branch into, and merge out of, the two backbones
x_in, x_out = stages[2][0] - 0.6, stages[2][1] + 0.6
for y in (42.5, 21.5):
    ax.plot([x_in - 1.4, x_in - 1.4], [MID, y], color="#2b2b30", lw=1.2, zorder=3)
    arrow(x_in - 1.4, y, x_in + 0.8, y, lw=1.2)
    ax.plot([x_out + 1.4, x_out + 1.4], [y, MID], color="#2b2b30", lw=1.2, zorder=3)
    arrow(x_out - 0.8, y, x_out + 1.4, y, lw=1.2)

# annotation routes
ax.text(133, 70.0, "Post-CBAM feature tensor", ha="center", fontsize=8.4, color="#2b2b30")
ax.plot([cx5, cx5, 149], [62.0, 67.0, 67.0], color="#2b2b30", lw=1.1, zorder=3)
arrow(149, 67.0, 149, 62.5, lw=1.1)
ax.text(176, 70.0, "Selected class score", ha="center", fontsize=8.4, color="#2b2b30")
ax.plot([cx6, cx6, 164], [62.0, 64.5, 64.5], color="#2b2b30", lw=1.1, zorder=3)
arrow(164, 64.5, 164, 62.5, lw=1.1)
ax.text(140, 3.2, "Classifier probability", ha="center", fontsize=8.4, color="#2b2b30")
ax.plot([cx6, cx6, 159], [9.0, 6.0, 6.0], color="#2b2b30", lw=1.1, zorder=3)
arrow(159, 6.0, 159, 9.4, lw=1.1)
ax.text(58, 1.2, "Input image", ha="center", fontsize=8.4, color="#2b2b30")
ax.plot([10.5, 10.5, cx10], [9.0, 0.2, 0.2], color="#2b2b30", lw=1.1, zorder=3)
arrow(cx10, 0.2, cx10, 12.8, lw=1.1)
ax.set_ylim(-3, 74)

fig.tight_layout(pad=0.2)
fig.savefig("Rad-Intel paper/architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
print("wrote architecture.png")
