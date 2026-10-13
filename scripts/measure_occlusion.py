"""Occlusion-sensitivity reference for the Grad-CAM maps.

Grad-CAM cannot validate itself: a map stays visually plausible whether or not
it reflects what the network used, which is how the Swin channels-last bug
survived inspection. Blanking patches and watching the class probability move
is slow but assumption-free, so it is the reference the attribution maps are
checked against. The reported quantity is the rank correlation between the
Grad-CAM map and the occlusion map over the same image; a negative value means
the explanation points away from the evidence.
"""
from __future__ import annotations
import argparse, glob, json, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from rad_intel.api.dependencies import model_manager as M
from rad_intel.preprocessing.transforms import default_preprocessor as P
from rad_intel.xai.gradcam import GradCAMExplainer
from rad_intel.xai.visualizer import normalize_heatmap

MODELS = ["densenet121", "swin_t", "resnet50", "hybrid_no_cbam", "hybrid"]


def occlusion_map(model, t, cat, patch=32, stride=16, batch=32):
    """Drop in P(class) when each patch is blanked, as a map at stride resolution."""
    _, _, H, W = t.shape
    with torch.no_grad():
        base = F.softmax(model(t), dim=-1)[0, cat].item()
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    coords = [(y, x) for y in ys for x in xs]
    drops = []
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        xb = t.repeat(len(chunk), 1, 1, 1).clone()
        for j, (y, x) in enumerate(chunk):
            xb[j, :, y:y + patch, x:x + patch] = 0.0
        with torch.no_grad():
            p = F.softmax(model(xb), dim=-1)[:, cat]
        drops.extend((base - p).tolist())
    m = np.asarray(drops, dtype=np.float32).reshape(len(ys), len(xs))
    # Only drops are evidence of use; a probability that rises when a region is
    # removed says the region argued against the class, not that it was ignored.
    return np.clip(m, 0.0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--data-dir", default="Rad-Intel paper/chest_xray")
    ap.add_argument("--out", default="results/table6_occlusion.json")
    args = ap.parse_args()

    rng = random.Random(42)
    sample = rng.sample(sorted(glob.glob(f"{args.data_dir}/test/*/*.jpeg")), args.n)
    out = {"n_images": args.n, "patch": 32, "stride": 16, "models": {}}

    for name in MODELS:
        model = M.get_model(name); model.eval()
        gc = GradCAMExplainer(model=model)
        rs = []
        for f in sample:
            t, _ = P.preprocess(f)
            cls, _, _, _ = M.predict(t, name)
            cat = 1 if cls == "PNEUMONIA" else 0
            occ = occlusion_map(model, t, cat)
            cam = normalize_heatmap(gc.generate_heatmap(t, cat))
            cam = torch.nn.functional.interpolate(
                torch.from_numpy(cam)[None, None], size=occ.shape,
                mode="bilinear", align_corners=False)[0, 0].numpy()
            r = spearmanr(cam.ravel(), occ.ravel()).statistic
            if np.isfinite(r):
                rs.append(float(r))
        out["models"][name] = {"mean_r": round(float(np.mean(rs)), 3),
                               "sd": round(float(np.std(rs)), 3),
                               "n": len(rs)}
        print(f"{name:16s} r = {out['models'][name]['mean_r']:+.3f} "
              f"(sd {out['models'][name]['sd']:.3f}, n={len(rs)})", flush=True)

    open(args.out, "w").write(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
