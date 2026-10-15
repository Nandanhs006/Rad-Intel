"""Replay build_review_packet's deterministic sampling to recover which model
and which 30 files the clinician actually saw."""
import sys, glob, random, csv, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,'src')
from pathlib import Path
from rad_intel.api.dependencies import model_manager as M
from rad_intel.preprocessing.transforms import default_preprocessor as P

rows=list(csv.DictReader(open('results/rad_intel_clinical_review.csv')))
target=[(r['truth'], r['predicted'], round(float(r['p_pneumonia']),3)) for r in rows]

def replay(model_name, n=30, seed=2026):
    rng=random.Random(seed)
    root=Path("Rad-Intel paper/chest_xray")/"test"
    pool={"PNEUMONIA":sorted(glob.glob(str(root/"PNEUMONIA"/"*.jpeg"))),
          "NORMAL":sorted(glob.glob(str(root/"NORMAL"/"*.jpeg")))}
    M.get_model(model_name)
    scanned=[]
    for truth, files in pool.items():
        for f in rng.sample(files, min(len(files), n*3)):
            t,_=P.preprocess(f)
            cls,_,probs,_=M.predict(t, model_name)
            scanned.append({"file":f,"truth":truth,"pred":cls,
                            "p":probs["PNEUMONIA"],"correct":cls==truth})
            if len([s for s in scanned if s["truth"]==truth])>=n*2: break
    strata={k:[s for s in scanned if (s["truth"],s["correct"])==k]
            for k in [("PNEUMONIA",True),("PNEUMONIA",False),
                      ("NORMAL",True),("NORMAL",False)]}
    want=max(1,n//4); chosen=[]
    for k,v in strata.items():
        rng.shuffle(v); chosen+=v[:want]
    leftover=[s for s in scanned if s not in chosen]
    rng.shuffle(leftover); chosen+=leftover[:max(0,n-len(chosen))]
    chosen=chosen[:n]; rng.shuffle(chosen)
    return chosen

for name in ["densenet121","hybrid"]:
    c=replay(name)
    got=[(x["truth"],x["pred"],round(x["p"],3)) for x in c]
    hits=sum(1 for a,b in zip(got,target) if a==b)
    print(f"{name:14s} exact positional matches {hits}/30")
    if hits>=28:
        with open('/private/tmp/claude-501/-Users-poorabng-Documents-GitHub-Rad-Intel/46389b50-a553-4b39-802c-64b4189379a6/scratchpad/review_files.txt','w') as fh:
            for x in c: fh.write(f"{x['file']}\t{x['truth']}\t{x['pred']}\t{x['p']:.6f}\n")
        print("  -> wrote review_files.txt")
