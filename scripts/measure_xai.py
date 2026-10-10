"""Measure Table VI: explanation behaviour for Grad-CAM and LIME."""
from __future__ import annotations
import argparse, glob, json, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import torch.nn.functional as F

from rad_intel.api.dependencies import model_manager as M
from rad_intel.preprocessing.transforms import default_preprocessor as P
from rad_intel.xai.gradcam import GradCAMExplainer
from rad_intel.xai.lime_explainer import LIMECXRExplainer
from rad_intel.xai.visualizer import normalize_heatmap


def curve_auc(model, t, cat, order, steps=20, insert=False):
    """Deletion/insertion AUC: blank (or restore) top-ranked pixels in order."""
    flat = t.view(1, 3, -1).clone()
    base = torch.zeros_like(flat)
    scores, per = [], max(1, len(order) // steps)
    for s in range(steps + 1):
        x = (base.clone() if insert else flat.clone())
        idx = torch.as_tensor(order[: s * per].copy(), dtype=torch.long)
        if insert:
            x[0, :, idx] = flat[0, :, idx]
        else:
            x[0, :, idx] = 0.0
        with torch.no_grad():
            scores.append(F.softmax(model(x.view_as(t)), dim=-1)[0, cat].item())
    return float(np.trapezoid(scores, dx=1.0 / steps))


def concentration(h):
    """Share of total saliency in the top 10% of pixels. Higher = more focal."""
    v = np.sort(h.ravel())[::-1]
    k = max(1, int(.10 * v.size))
    return float(v[:k].sum() / (v.sum() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--n-lime", type=int, default=8, help="LIME is slow; smaller sample")
    ap.add_argument("--data-dir", default="Rad-Intel paper/chest_xray")
    args = ap.parse_args()

    rng = random.Random(42)
    files = sorted(glob.glob(f"{args.data_dir}/test/*/*.jpeg"))
    sample = rng.sample(files, args.n)

    model = M.get_model(args.model); model.eval()
    gc = GradCAMExplainer(model=model)
    out = {"model": args.model, "n_gradcam": args.n, "n_lime": args.n_lime}

    # ---- Grad-CAM ----
    dele, ins, conc, stab, times = [], [], [], [], []
    for f in sample:
        t, rgb = P.preprocess(f)
        cls, _, _, _ = M.predict(t, args.model)
        cat = 1 if cls == "PNEUMONIA" else 0
        t0 = time.perf_counter(); h = gc.generate_heatmap(t, cat); times.append(time.perf_counter() - t0)
        h = normalize_heatmap(h)
        order = np.argsort(h.ravel())[::-1]
        dele.append(curve_auc(model, t, cat, order))
        ins.append(curve_auc(model, t, cat, order, insert=True))
        conc.append(concentration(h))
        # Stability: small label-preserving shift, rank correlation of the maps.
        t2 = torch.roll(t, shifts=(3, 3), dims=(2, 3))
        h2 = normalize_heatmap(gc.generate_heatmap(t2, cat))
        h2 = np.roll(h2, shift=(-3, -3), axis=(0, 1))
        from scipy.stats import spearmanr
        stab.append(spearmanr(h.ravel(), h2.ravel()).statistic)
    out["gradcam"] = {"deletion_auc": round(float(np.mean(dele)), 4),
                      "insertion_auc": round(float(np.mean(ins)), 4),
                      "concentration": round(float(np.mean(conc)), 4),
                      "stability": round(float(np.nanmean(stab)), 4),
                      "runtime_s": round(float(np.mean(times)), 3)}
    print("gradcam:", out["gradcam"], flush=True)

    # ---- LIME ----
    lime = LIMECXRExplainer(model=model, device=M.device)
    ld, li, lc, lt, iou = [], [], [], [], []
    for f in sample[: args.n_lime]:
        t, rgb = P.preprocess(f)
        cls, _, _, _ = M.predict(t, args.model)
        cat = 1 if cls == "PNEUMONIA" else 0
        t0 = time.perf_counter()
        res = lime.explain(image_rgb=rgb, target_category=cat)
        lt.append(time.perf_counter() - t0)
        lh = res.get("heatmap")
        if lh is None:
            continue
        lh = normalize_heatmap(np.asarray(lh, dtype=np.float32))
        order = np.argsort(lh.ravel())[::-1]
        ld.append(curve_auc(model, t, cat, order))
        li.append(curve_auc(model, t, cat, order, insert=True))
        lc.append(concentration(lh))
        gh = normalize_heatmap(gc.generate_heatmap(t, cat))
        a, b = gh >= np.quantile(gh, .8), lh >= np.quantile(lh, .8)
        iou.append(float((a & b).sum() / max((a | b).sum(), 1)))
    out["lime"] = {"deletion_auc": round(float(np.mean(ld)), 4) if ld else None,
                   "insertion_auc": round(float(np.mean(li)), 4) if li else None,
                   "concentration": round(float(np.mean(lc)), 4) if lc else None,
                   "runtime_s": round(float(np.mean(lt)), 2) if lt else None}
    out["gradcam_lime_iou"] = round(float(np.mean(iou)), 4) if iou else None
    print("lime:", out["lime"], "| IoU:", out["gradcam_lime_iou"])
    open("results/table6_xai.json", "w").write(json.dumps(out, indent=2))
    print("wrote results/table6_xai.json")


if __name__ == "__main__":
    main()
