"""Generate Figure 2 (ROC + reliability) and Figure 3 (qualitative panel)."""
from __future__ import annotations
import glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "Rad-Intel paper"
NAMES = {"densenet121":"DenseNet121","swin_t":"Swin-T","resnet50":"ResNet50",
         "hybrid_no_cbam":"Fusion w/o CBAM","hybrid":"Complete classifier"}
# Colour-blind safe, distinguishable in greyscale print via linestyle.
STYLE = {"densenet121":("#0072B2","-"), "swin_t":("#E69F00","--"),
         "resnet50":("#009E73","-."), "hybrid_no_cbam":("#CC79A7",(0,(3,1,1,1))),
         "hybrid":("#D55E00","-")}

def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for k, label in NAMES.items():
        f = f"results/roc_{k}.csv"
        if not os.path.exists(f): continue
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        c, ls = STYLE[k]
        ax[0].plot(d[:,0], d[:,1], color=c, ls=ls, lw=1.4, label=label)
    ax[0].plot([0,1],[0,1], color="0.6", lw=.8, ls=":")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC", fontsize=9); ax[0].legend(fontsize=6, loc="lower right")
    ax[0].set_xlim(0,1); ax[0].set_ylim(0,1.005)

    ax[1].plot([0,1],[0,1], color="0.6", lw=.8, ls=":", label="perfect")
    for k, label in NAMES.items():
        f = f"results/reliability_{k}.csv"
        if not os.path.exists(f): continue
        d = np.genfromtxt(f, delimiter=",", skip_header=1)
        if d.ndim == 1: d = d[None]
        c, ls = STYLE[k]
        ax[1].plot(d[:,1], d[:,2], color=c, ls=ls, lw=1.3, marker="o", ms=2.5, label=label)
    ax[1].set_xlabel("Mean predicted probability"); ax[1].set_ylabel("Observed frequency")
    ax[1].set_title("Reliability (15 bins)", fontsize=9)
    ax[1].set_xlim(0,1); ax[1].set_ylim(0,1)
    for a in ax: a.tick_params(labelsize=7); a.grid(alpha=.25, lw=.4)
    fig.tight_layout(pad=.4)
    fig.savefig(f"{OUT}/fig_roc_reliability.png", dpi=320)
    print("wrote fig_roc_reliability.png")

def fig3(model="hybrid"):
    from rad_intel.api.dependencies import model_manager as M
    from rad_intel.preprocessing.transforms import default_preprocessor as P
    from rad_intel.xai.gradcam import GradCAMExplainer
    import random
    rng = random.Random(11)
    root = "Rad-Intel paper/chest_xray/test"
    m = M.get_model(model); ex = GradCAMExplainer(m)
    want = {("PNEUMONIA",True):None, ("NORMAL",True):None,
            ("NORMAL",False):None, ("PNEUMONIA",False):None}
    for truth in ("PNEUMONIA","NORMAL"):
        files = glob.glob(f"{root}/{truth}/*.jpeg"); rng.shuffle(files)
        for f in files[:160]:
            t, rgb = P.preprocess(f)
            cls, conf, probs, _ = M.predict(t, model)
            key = (truth, cls == truth)
            if key in want and want[key] is None:
                want[key] = (f, t, rgb, cls, probs["PNEUMONIA"])
            if all(v is not None for v in want.values()): break
    # Two rows only: one correct and one incorrect case. The false positive is
    # the informative one and the figure is a third of the height.
    rows = [k for k in [("PNEUMONIA",True),("NORMAL",False)] if want[k]]
    titles = {("PNEUMONIA",True):"True positive", ("NORMAL",True):"True negative",
              ("NORMAL",False):"False positive", ("PNEUMONIA",False):"False negative"}
    fig, axes = plt.subplots(len(rows), 2, figsize=(3.5, 1.72*len(rows)))
    if len(rows) == 1: axes = axes[None]
    for r, key in enumerate(rows):
        f, t, rgb, cls, p = want[key]
        res = ex.explain(t, rgb, 1 if cls == "PNEUMONIA" else 0)
        import cv2, base64
        ov = res["overlay_base64"].split(",",1)[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(ov), np.uint8), cv2.IMREAD_COLOR)
        axes[r,0].imshow(rgb); axes[r,1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[r,0].set_ylabel(titles[key], fontsize=6)
        axes[r,0].set_title("Radiograph" if r==0 else "", fontsize=7)
        axes[r,1].set_title("Grad-CAM" if r==0 else "", fontsize=7)
        axes[r,1].text(.98,.03, f"P={p:.2f}", transform=axes[r,1].transAxes,
                       ha="right", va="bottom", fontsize=5.5, color="w",
                       bbox=dict(fc="k", alpha=.55, pad=1, lw=0))
        for a in axes[r]: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=.3)
    fig.savefig(f"{OUT}/fig_qualitative.png", dpi=320)
    print(f"wrote fig_qualitative.png ({len(rows)} rows)")

if __name__ == "__main__":
    fig2(); fig3()
