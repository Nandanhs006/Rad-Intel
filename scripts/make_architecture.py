"""
System architecture figure, laid out over two rows.

The pipeline has ten stages; placing them in a single row forces the type so
small that the figure is unreadable at column width. Two rows of five keep
the same colour-coded stage design while roughly doubling the usable box
width, so labels stay legible in print.

Content corrected against the implementation: CLAHE and a direct resize in
preprocessing, concatenation before projection in fusion, a two-logit softmax
head at a calibrated threshold, explanations confined to segmented lung
fields, and Gemini 3.5 Flash-Lite as the summarizer.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg

C = {"grey":("#50555e","#e9eaed"), "green":("#1f6b38","#e3f1e8"),
     "blue":("#19459b","#e4ebf9"), "teal":("#0d6b66","#ddf0ee"),
     "purple":("#5433a8","#eae3fa"), "orange":("#a85400","#fdeccf"),
     "slate":("#2c3653","#e5e8ef")}

FS_TITLE, FS_MAIN, FS_SUB, FS_NOTE = 15, 13.5, 11.5, 12

fig, ax = plt.subplots(figsize=(14.0, 9.6))
ax.set_xlim(-4, 110); ax.set_ylim(0, 100); ax.axis("off")
fig.patch.set_facecolor("white")

ROW = {1: (96.0, 57.0), 2: (44.0, 5.0)}      # (top, bottom) per row


def group(x0, w, row, title, colour):
    top, bot = ROW[row]
    edge, face = C[colour]
    ax.add_patch(FancyBboxPatch((x0, bot), w, top - bot - 6.0,
                                boxstyle="round,pad=0.5,rounding_size=1.4",
                                ec=edge, fc=face, lw=2.4, zorder=1))
    ax.text(x0 + w / 2, top - 3.0, title, ha="center", va="center",
            fontsize=FS_TITLE, color=edge, fontweight="bold", zorder=3)
    return x0 + w / 2, x0, x0 + w


def sub(cx, w, ytop, h, lines, colour, fs=FS_MAIN, bold0=False):
    edge, _ = C[colour]
    ax.add_patch(FancyBboxPatch((cx - w / 2, ytop - h), w, h,
                                boxstyle="round,pad=0.3,rounding_size=0.8",
                                ec=edge, fc="white", lw=1.7, zorder=2))
    n = len(lines)
    for i, t in enumerate(lines):
        ax.text(cx, ytop - h / (n + 1) * (i + 1), t, ha="center", va="center",
                fontsize=fs if i == 0 else FS_SUB,
                color="#101014" if i == 0 else "#3d3e46",
                fontweight="bold" if (bold0 and i == 0) else "normal", zorder=3)


def stack(cx, w, items, colour, top, bottom, fs=FS_MAIN, bold0=False, gap=2.0):
    n = len(items); h = (top - bottom - gap * (n - 1)) / n
    for i, lines in enumerate(items):
        sub(cx, w, top - i * (h + gap), h, lines, colour, fs=fs, bold0=bold0)


def arrow(x1, y1, x2, y2, lw=2.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=lw, color="#15151a", zorder=4))


# ================= ROW 1 =================
T1, B1 = 96.0, 57.0
MID1 = (T1 - 6.0 + B1) / 2
cx, _, e1 = group(1.5, 15.0, 1, "Input", "grey")
ax.text(cx, 86.0, "Chest X-Ray", ha="center", fontsize=FS_MAIN, color="#101014")
try:
    ax.imshow(mpimg.imread("sample_images/sample_pneumonia_person100_bacteria_478.jpeg"),
              extent=[cx - 5.6, cx + 5.6, 60, 83], aspect="auto", zorder=2, cmap="gray")
except Exception:
    pass

cxP, s2, e2 = group(19.5, 18.0, 1, "Preprocessing", "green")
stack(cxP, 15.5, [["CLAHE", "clip 2.0, 8x8 tiles"], ["Resize 224 x 224", "no padding"],
                  ["ImageNet", "Normalization"], ["Training", "Augmentation"]],
      "green", 86.0, 59.0)

cxF, s3, e3 = group(41.5, 21.0, 1, "Parallel Feature Extraction", "blue")
sub(cxF, 18.5, 86.0, 12.0, ["DenseNet121", "Local features", "1024 x 7 x 7"], "blue", bold0=True)
sub(cxF, 18.5, 72.0, 12.0, ["Swin-T", "Global context", "768 x 7 x 7"], "blue", bold0=True)

cxU, s4, e4 = group(66.5, 18.5, 1, "Feature Fusion", "blue")
sub(cxU, 16.0, 85.0, 12.0, ["Channel Concat.", "1792 x 7 x 7"], "blue")
sub(cxU, 16.0, 71.0, 12.0, ["1x1 Conv-BN-ReLU", "512 x 7 x 7"], "blue")

cxC, s5, e5 = group(89.0, 17.0, 1, "CBAM Refinement", "green")
sub(cxC, 13.0, 84.0, 11.0, ["Channel", "Attention"], "green")
sub(cxC, 13.0, 71.0, 11.0, ["Spatial", "Attention"], "green")

for a, b in [(e1, s2), (e2, s3), (e3, s4), (e4, s5)]:
    arrow(a + 0.5, MID1, b - 0.5, MID1)
for y in (80.0, 66.0):
    ax.plot([s3 - 2.0, s3 - 2.0], [MID1, y], color="#15151a", lw=1.9, zorder=3)
    arrow(s3 - 2.0, y, s3 + 0.6, y, lw=1.9)
    ax.plot([e3 + 2.0, e3 + 2.0], [y, MID1], color="#15151a", lw=1.9, zorder=3)
    arrow(e3 - 0.6, y, e3 + 2.0, y, lw=1.9)

# ================= ROW 2 =================
T2, B2 = 44.0, 5.0
MID2 = (T2 - 6.0 + B2) / 2
cxCl, s6, e6 = group(1.5, 17.0, 2, "Classifier", "teal")
stack(cxCl, 14.5, [["Global Average", "Pooling"], ["Dropout (0.3)", "+ Linear"],
                   ["Two Logits", "+ Softmax"], ["Calibrated t*", "(Youden)"]],
      "teal", 34.0, 7.0)

cxE, s7, e7 = group(22.0, 19.5, 2, "Explainability", "purple")
sub(cxE, 17.5, 34.0, 6.5, ["Lung-Field Segmentation"], "purple", fs=12.5)
sub(cxE - 4.4, 8.2, 25.5, 17.0, ["Grad-CAM", "post-CBAM", "saliency"], "purple", fs=12.5)
sub(cxE + 4.4, 8.2, 25.5, 17.0, ["LIME", "superpixel", "weights"], "purple", fs=12.5)

cxS, s8, e8 = group(45.5, 19.0, 2, "Evidence Serializer", "orange")
stack(cxS, 16.5, [["Label, Probability", "Threshold"],
                  ["Quadrant Scores", "Off-Lung Fraction"],
                  ["Version and", "Checkpoint IDs"]], "orange", 34.0, 7.0, fs=12.5)

cxG, s9, e9 = group(68.5, 18.0, 2, "Constrained Summary", "orange")
stack(cxG, 14.5, [["Gemini 3.5 Flash-Lite"], ["Fixed JSON Schema"],
                  ["Validation"], ["Repair / Fallback"]], "orange", 34.0, 7.0, fs=12.0)

cxI, s10, e10 = group(90.0, 18.5, 2, "Research Interface", "slate")
stack(cxI, 16.0, [["Probability + Threshold"], ["Grad-CAM and LIME"],
                  ["Evidence Summary"], ["Limitations"]], "slate", 34.0, 12.5, fs=12.0)
ax.add_patch(FancyBboxPatch((cxI - 7.5, 6.5), 15.0, 4.6,
                            boxstyle="round,pad=0.3,rounding_size=0.8",
                            ec="#1b2541", fc="#1b2541", lw=1.5, zorder=2))
ax.text(cxI, 8.8, "Research Use Only", ha="center", va="center",
        fontsize=12.5, color="white", fontweight="bold", zorder=3)

for a, b in [(e6, s7), (e7, s8), (e8, s9), (e9, s10)]:
    arrow(a + 0.5, MID2, b - 0.5, MID2)

# row 1 -> row 2 connector
ax.plot([e5 + 1.2, 108.5, 108.5, -2.8, -2.8], [MID1, MID1, 49.5, 49.5, MID2],
        color="#15151a", lw=1.9, zorder=3)
arrow(-2.8, MID2, s6 - 0.5, MID2, lw=1.9)
ax.text(52.0, 51.3, "fused representation", ha="center", fontsize=FS_NOTE, color="#15151a")

# annotation routes into Explainability
ax.text(31.0, 47.0, "post-CBAM tensor  ·  class score  ·  probability",
        ha="center", fontsize=FS_NOTE, color="#15151a")
arrow(cxE, 45.5, cxE, 38.5, lw=1.9)

fig.tight_layout(pad=0.3)
fig.savefig("Rad-Intel paper/architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote architecture.png")
