"""
System architecture figure, sized for a single IEEE column.

Ten pipeline stages in a 2 x 5 grid. A single wide row, or even two rows of
five, is unreadable once scaled to ~3.5 in: this arrangement keeps each stage
box as wide as the column allows, which is the only way the labels survive.
Flow runs left to right within a row and then down, in a boustrophedon so
no long return connectors are needed.

Content matches the implementation: CLAHE and a direct resize, concatenation
before projection, a two-logit softmax head at a calibrated threshold,
explanations confined to segmented lung fields, Gemini 3.5 Flash-Lite.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"grey":("#4a4f58","#eceef0"), "green":("#1b6334","#e2f0e7"),
     "blue":("#143d91","#e2eaf8"), "teal":("#0a625d","#dcefed"),
     "purple":("#4d2fa0","#e9e2fa"), "orange":("#9c4e00","#fcebcd"),
     "slate":("#28314d","#e4e7ee")}

# (title, colour, body lines)
# grid position (row, col) -> stage.  Row 2 holds the two parallel branches.
STAGES = {
    (0,0): ("Input",               "grey",   ["Chest radiograph"]),
    (0,1): ("Preprocessing",       "green",  ["CLAHE  ·  resize 224", "ImageNet norm  ·  augment"]),
    (1,0): ("DenseNet121",         "blue",   ["local features", "1024 x 7 x 7"]),
    (1,1): ("Swin-T",              "blue",   ["global context", "768 x 7 x 7"]),
    (2,0): ("Feature Fusion",      "blue",   ["concatenate 1792", "1x1 conv  ->  512"]),
    (2,1): ("CBAM",                "green",  ["channel attention", "spatial attention"]),
    (3,1): ("Classifier",          "teal",   ["GAP · dropout · linear", "2 logits · softmax · t*"]),
    (3,0): ("Explainability",      "purple", ["lung-field segmentation", "Grad-CAM  ·  LIME"]),
    (4,0): ("Evidence Serializer", "orange", ["label · probability · t*", "quadrants  ·  off-lung"]),
    (4,1): ("Summary + Interface", "slate",  ["Gemini 3.5 Flash-Lite", "schema · validate · fallback"]),
}

COLS, ROWS = 2, 5
BW, BH = 44.0, 15.0          # box size
GX, GY = 10.0, 5.0           # gaps
W = COLS * BW + GX
H = ROWS * BH + (ROWS - 1) * GY

fig, ax = plt.subplots(figsize=(7.2, 8.4))
ax.set_xlim(-2, W + 2); ax.set_ylim(-2, H + 2); ax.axis("off")
fig.patch.set_facecolor("white")


def box(r, c):
    x = c * (BW + GX)
    y = H - BH - r * (BH + GY)
    return x, y


pos = {}
for (r, c), (title, colour, lines) in STAGES.items():
    x, y = box(r, c)
    edge, face = C[colour]
    ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                                boxstyle="round,pad=0.8,rounding_size=2.2",
                                ec=edge, fc=face, lw=2.6, zorder=2))
    ax.text(x + BW / 2, y + BH - 4.4, title, ha="center", va="center",
            fontsize=15.5, color=edge, fontweight="bold", zorder=3)
    for j, t in enumerate(lines):
        ax.text(x + BW / 2, y + BH - 9.0 - j * 4.4, t, ha="center", va="center",
                fontsize=12.5, color="#24252b", zorder=3)
    pos[(r, c)] = (x, y)


xf0, yf0 = pos[(2,0)]


def edge_pt(rc, side):
    x, y = pos[rc]
    return {"L": (x, y + BH / 2), "R": (x + BW, y + BH / 2),
            "T": (x + BW / 2, y + BH), "B": (x + BW / 2, y)}[side]


def link(a, sa, b, sb, lw=2.2):
    ax.add_patch(FancyArrowPatch(edge_pt(a, sa), edge_pt(b, sb), arrowstyle="-|>",
                                 mutation_scale=17, lw=lw, color="#141418", zorder=4))


link((0,0), "R", (0,1), "L")                    # input -> preprocessing
link((0,1), "B", (1,1), "T")                    # -> Swin branch
xp, yp = pos[(0,1)]; xd, yd = pos[(1,0)]
ax.plot([xp + BW / 2, xd + BW / 2], [yp - 2.5, yp - 2.5], color="#141418", lw=2.2, zorder=3)
ax.plot([xp + BW / 2, xp + BW / 2], [yp, yp - 2.5], color="#141418", lw=2.2, zorder=3)
ax.add_patch(FancyArrowPatch((xd + BW / 2, yp - 2.5), (xd + BW / 2, yd + BH),
                             arrowstyle="-|>", mutation_scale=17, lw=2.2,
                             color="#141418", zorder=4))
xd0, yd0 = pos[(1,0)]
ax.add_patch(FancyArrowPatch((xd0 + BW * 0.28, yd0), (xd0 + BW * 0.28, yf0 + BH),
                             arrowstyle="-|>", mutation_scale=17, lw=2.2,
                             color="#141418", zorder=4))
link((2,0), "R", (2,1), "L")                    # fusion -> CBAM
link((2,1), "B", (3,1), "T")                    # CBAM -> classifier
link((3,1), "L", (3,0), "R")                    # classifier -> explainability
link((3,0), "B", (4,0), "T")                    # -> serializer
link((4,0), "R", (4,1), "L")                    # -> summary + interface

# Swin also feeds fusion: drop below the branch row, run left, enter the
# fusion box from above alongside the DenseNet arrow.
xs, ys = pos[(1,1)]; xf, yf = pos[(2,0)]
ymid = (ys + yf + BH) / 2
xin = xf + BW * 0.72
ax.plot([xs + BW / 2, xs + BW / 2], [ys, ymid], color="#141418", lw=2.2, zorder=3)
ax.plot([xs + BW / 2, xin], [ymid, ymid], color="#141418", lw=2.2, zorder=3)
ax.add_patch(FancyArrowPatch((xin, ymid), (xin, yf + BH),
                             arrowstyle="-|>", mutation_scale=17, lw=2.2,
                             color="#141418", zorder=4))

fig.tight_layout(pad=0.2)
fig.savefig("Rad-Intel paper/architecture.png", dpi=320, bbox_inches="tight",
            facecolor="white")
print("wrote architecture.png")
