"""Quantify the gap between the training/eval preprocessing and the service path.

Both apply the same CLAHE settings and resize to the same 224x224 target, but the
training pipeline resizes through torchvision/PIL while the service path uses
cv2.INTER_AREA. Table I is computed through the former and the explanation and
review measurements through the latter, so the size of the gap belongs on record.
"""
import sys, json, pathlib, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'src')
from rad_intel.api.dependencies import model_manager as M
from rad_intel.preprocessing.transforms import default_preprocessor as P
from rad_intel.training.dataset import CXRDataset, get_transforms, load_dataset_splits
from sklearn.metrics import roc_auc_score
from PIL import Image as PILImage

samples = load_dataset_splits(data_dir="Rad-Intel paper/chest_xray")[2]
name = "hybrid"
model = M.get_model(name); model.eval()
thr = M.get_threshold(name)
tf = get_transforms(is_train=False)

pe, ps, ys = [], [], []
for path, y in samples:
    te = tf(PILImage.open(path).convert("RGB"))[None]
    ts, _ = P.preprocess(path)
    with torch.no_grad():
        pe.append(F.softmax(model(te), -1)[0, 1].item())
        ps.append(F.softmax(model(ts), -1)[0, 1].item())
    ys.append(y)

pe, ps, ys = np.array(pe), np.array(ps), np.array(ys)
de, ds = (pe >= thr).astype(int), (ps >= thr).astype(int)
print(f"threshold {thr:.4f}  n={len(ys)}")
print(f"mean |dP|            {np.abs(pe-ps).mean():.4f}   max {np.abs(pe-ps).max():.4f}")
print(f"decisions that flip  {int((de!=ds).sum())} of {len(ys)}  ({100*(de!=ds).mean():.1f}%)")
print(f"accuracy  eval pipeline {(de==ys).mean():.4f}   serving pipeline {(ds==ys).mean():.4f}")
print(f"AUROC     eval pipeline {roc_auc_score(ys,pe):.4f}   serving pipeline {roc_auc_score(ys,ps):.4f}")
