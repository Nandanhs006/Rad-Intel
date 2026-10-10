"""Measure Table VII: behaviour of the constrained summarization step."""
from __future__ import annotations
import argparse, glob, json, random, re, time, warnings
warnings.filterwarnings("ignore")

from rad_intel.api.dependencies import model_manager as M
from rad_intel.preprocessing.transforms import default_preprocessor as P
from rad_intel.reporting.report_generator import default_report_generator as G
from rad_intel.xai.gradcam import GradCAMExplainer

REQUIRED = ["examination", "clinical_indication", "technique",
            "findings", "impression", "recommendations"]
# Terms the output may not introduce: specific lobar anatomy the serializer
# never computes (it works in quadrants), and treatment instruction.
FORBIDDEN = re.compile(
    r"\b(left|right)\s+(upper|middle|lower)\s+lobe\b|\blingula\b|"
    r"\b(prescrib|administer|initiate|commence)\w*\s+(antibiotic|antimicrobial)|"
    r"\bstart\s+(antibiotic|antimicrobial)", re.I)


def numbers_in(text):
    return {round(float(x), 1) for x in re.findall(r"\d+\.\d+|\b\d{1,3}\b", text or "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--data-dir", default="Rad-Intel paper/chest_xray")
    args = ap.parse_args()

    rng = random.Random(7)
    files = rng.sample(sorted(glob.glob(f"{args.data_dir}/test/*/*.jpeg")), args.n)
    model = M.get_model(args.model); ex = GradCAMExplainer(model=model)

    stats = {"n": args.n, "schema_ok": 0, "forbidden": 0, "evidence_ok": 0,
             "fallback": 0, "engine": None, "latencies": []}
    repeat_src = None
    for i, f in enumerate(files):
        t, rgb = P.preprocess(f)
        cls, conf, probs, _ = M.predict(t, args.model)
        res = ex.explain(t, rgb, 1 if cls == "PNEUMONIA" else 0)
        t0 = time.perf_counter()
        rep = G.generate(prediction_class=cls, confidence=conf, probabilities=probs,
                         localization=res["localization"])
        stats["latencies"].append(time.perf_counter() - t0)
        sec = rep.get("sections", {})
        stats["engine"] = rep.get("engine")
        if rep.get("engine", "").startswith("rule-based"):
            stats["fallback"] += 1
        body = " ".join(str(sec.get(k, "")) for k in REQUIRED)
        if all(sec.get(k) for k in REQUIRED):
            stats["schema_ok"] += 1
        if not FORBIDDEN.search(body):
            stats["forbidden"] += 0
        else:
            stats["forbidden"] += 1
        # Evidence agreement: the stated confidence must match the classifier's.
        pct = round(conf * 100, 1)
        if any(abs(n - pct) < 0.15 for n in numbers_in(body)) or f"{pct:.1f}" in body:
            stats["evidence_ok"] += 1
        if i == 0:
            repeat_src = (cls, conf, probs, res["localization"])
        print(f"  {i+1}/{args.n}", end="\r", flush=True)
    print()

    # Repeatability: identical evidence, three requests at temperature 0.
    cls, conf, probs, loc = repeat_src
    outs = [G.generate(prediction_class=cls, confidence=conf, probabilities=probs,
                       localization=loc).get("sections", {}).get("findings", "")
            for _ in range(3)]
    stats["repeatable"] = len(set(outs)) == 1

    n = stats["n"]
    res = {
        "engine": stats["engine"],
        "schema_valid_pct": round(100 * stats["schema_ok"] / n, 1),
        "evidence_agreement_pct": round(100 * stats["evidence_ok"] / n, 1),
        "forbidden_claim_pct": round(100 * stats["forbidden"] / n, 1),
        "fallback_pct": round(100 * stats["fallback"] / n, 1),
        "repeatable": stats["repeatable"],
        "mean_latency_s": round(sum(stats["latencies"]) / n, 2),
    }
    print(json.dumps(res, indent=2))
    open("results/table7_reporting.json", "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
