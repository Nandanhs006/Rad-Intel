"""
Build a self-contained clinical review packet.

Produces one HTML file holding a stratified sample of test cases. Each case
shows the radiograph, the Grad-CAM overlay and the generated report, with a
four-item rubric the reviewer scores in the browser. Scores are kept in the
page and exported as CSV with one button; nothing is uploaded anywhere and no
software needs installing.

Sampling is stratified over the two classes AND over whether the model was
right, because the incorrect cases are the informative ones: a system that
produces confident, plausible explanations for wrong predictions is the
failure mode the review exists to detect. Case order is shuffled and the
model's verdict is hidden until the reviewer reveals it, so the rubric is
scored on the image and overlay rather than anchored on the prediction.

Usage:
    python scripts/build_review_packet.py --n 30 --model densenet121
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import random
from pathlib import Path

import numpy as np

from rad_intel.api.dependencies import model_manager
from rad_intel.preprocessing.transforms import default_preprocessor
from rad_intel.reporting.report_generator import default_report_generator
from rad_intel.xai.gradcam import GradCAMExplainer


def b64(path_or_arr):
    from rad_intel.xai.visualizer import image_to_base64
    return image_to_base64(path_or_arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="Rad-Intel paper/chest_xray")
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--n", type=int, default=30, help="total cases")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/clinical_review_packet.html")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.data_dir) / "test"
    pool = {
        "PNEUMONIA": sorted(glob.glob(str(root / "PNEUMONIA" / "*.jpeg"))),
        "NORMAL": sorted(glob.glob(str(root / "NORMAL" / "*.jpeg"))),
    }

    model = model_manager.get_model(args.model)
    thr = model_manager.get_threshold(args.model)
    explainer = GradCAMExplainer(model=model)
    print(f"model={args.model}  threshold={thr:.4f}")

    # Oversample, classify, then stratify by (truth, correct) so incorrect
    # cases are guaranteed representation rather than left to chance.
    scanned = []
    for truth, files in pool.items():
        for f in rng.sample(files, min(len(files), args.n * 3)):
            t, rgb = default_preprocessor.preprocess(f)
            cls, conf, probs, _ = model_manager.predict(t, args.model)
            scanned.append(
                {"file": f, "truth": truth, "pred": cls, "p": probs["PNEUMONIA"],
                 "correct": cls == truth, "tensor": t, "rgb": rgb, "conf": conf,
                 "probs": probs}
            )
            if len([s for s in scanned if s["truth"] == truth]) >= args.n * 2:
                break

    strata = {k: [s for s in scanned if (s["truth"], s["correct"]) == k]
              for k in [("PNEUMONIA", True), ("PNEUMONIA", False),
                        ("NORMAL", True), ("NORMAL", False)]}
    for k, v in strata.items():
        print(f"  available {k}: {len(v)}")

    # Aim for a quarter of the sample in each stratum, topping up from the
    # correct cases when there are not enough errors to go round.
    want = max(1, args.n // 4)
    chosen = []
    for k, v in strata.items():
        rng.shuffle(v)
        chosen += v[:want]
    leftover = [s for s in scanned if s not in chosen]
    rng.shuffle(leftover)
    chosen += leftover[: max(0, args.n - len(chosen))]
    chosen = chosen[: args.n]
    rng.shuffle(chosen)
    print(f"selected {len(chosen)} cases "
          f"({sum(1 for c in chosen if not c['correct'])} incorrect)")

    cases = []
    for i, c in enumerate(chosen, 1):
        res = explainer.explain(c["tensor"], c["rgb"], 1 if c["pred"] == "PNEUMONIA" else 0)
        rep = default_report_generator.generate(
            prediction_class=c["pred"], confidence=c["conf"],
            probabilities=c["probs"], localization=res["localization"],
            patient_metadata={"patient_id": f"CASE-{i:02d}"},
        )
        sec = rep.get("sections", {})
        cases.append({
            "id": i,
            "original": b64(c["rgb"]),
            "overlay": res["overlay_base64"],
            "pred": c["pred"],
            "p": round(c["p"], 3),
            "truth": c["truth"],
            "correct": c["correct"],
            "zone": res["localization"].get("dominant_zone_description", ""),
            "findings": sec.get("findings", ""),
            "impression": sec.get("impression", ""),
        })
        print(f"  case {i}/{len(chosen)}", end="\r", flush=True)
    print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cases, args.model, thr), encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("Send this single file to the reviewer. No install needed.")


def render(cases, model_name, thr):
    data = json.dumps(cases)
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rad-Intel &mdash; Clinical Review</title>
<style>
 :root{--bg:#f7f7f9;--card:#fff;--ink:#1a1a1e;--mut:#6b6b76;--line:#e3e3e8;--accent:#c2410c}
 @media(prefers-color-scheme:dark){:root{--bg:#141417;--card:#1d1d21;--ink:#ececf0;--mut:#9a9aa4;--line:#2e2e35}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 .wrap{max-width:860px;margin:0 auto;padding:16px}
 h1{font-size:1.35rem;margin:.2em 0}
 .intro{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:18px}
 .case{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:18px}
 .imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 @media(max-width:600px){.imgs{grid-template-columns:1fr}}
 .imgs figure{margin:0}.imgs img{width:100%;border-radius:8px;display:block;background:#000}
 figcaption{font-size:.78rem;color:var(--mut);margin-bottom:4px}
 .rep{font-size:.86rem;background:rgba(127,127,140,.08);border-radius:8px;padding:10px;margin-top:10px;white-space:pre-wrap}
 .q{margin:14px 0 4px;font-weight:600;font-size:.92rem}
 .scale{display:flex;gap:6px;flex-wrap:wrap}
 .scale label{flex:1;min-width:58px;text-align:center;border:1px solid var(--line);border-radius:8px;padding:8px 4px;cursor:pointer;font-size:.85rem}
 .scale input{display:none}
 .scale input:checked+span{font-weight:700;color:var(--accent)}
 .scale label:has(input:checked){border-color:var(--accent);background:rgba(194,65,12,.08)}
 .hint{font-size:.78rem;color:var(--mut);margin-bottom:6px}
 .verdict{margin-top:12px;font-size:.85rem;color:var(--mut)}
 button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:11px 18px;font-size:.95rem;cursor:pointer}
 .ghost{background:transparent;color:var(--accent);border:1px solid var(--accent)}
 textarea{width:100%;border:1px solid var(--line);border-radius:8px;padding:8px;background:transparent;color:var(--ink);font:inherit;font-size:.86rem}
 .bar{position:sticky;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:12px 16px;display:flex;gap:10px;align-items:center;justify-content:space-between}
 code{background:rgba(127,127,140,.14);padding:1px 5px;border-radius:4px}
</style></head><body>
<div class="wrap">
<h1>Rad-Intel &mdash; Clinical Review</h1>
<div class="intro">
<p><strong>What this is.</strong> An automated system classifies paediatric chest
radiographs as normal or pneumonia, highlights the region that drove its
decision, and writes a short report from those outputs. We are asking you to
judge whether what it displays is <em>defensible</em> &mdash; not to re-diagnose
the patients.</p>
<p><strong>What we are not asking.</strong> You are not establishing ground truth,
and you are not being asked to agree or disagree with the diagnosis. Please
score each case on the four items as shown.</p>
<p><strong>Time.</strong> About 30&ndash;45 minutes. The model's verdict is hidden until
you choose to reveal it, so please look at the image and overlay first.</p>
<p>When finished, press <strong>Export CSV</strong> at the bottom and send us the file.
Nothing is uploaded; everything stays in your browser.</p>
<p class="hint">Scale: 1 = strongly disagree &nbsp;&middot;&nbsp; 3 = neutral &nbsp;&middot;&nbsp; 5 = strongly agree</p>
</div>
<div class="intro">
<label>Reviewer name (as it should appear in the acknowledgment)<br><input id="rev-name" style="width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--ink)"></label>
<label style="display:block;margin-top:10px">Qualification / affiliation<br><input id="rev-qual" style="width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--ink)"></label>
<label style="display:block;margin-top:10px">Years of reporting experience<br><input id="rev-yrs" type="number" style="width:120px;padding:8px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--ink)"></label>
</div>
<div id="cases"></div>
</div>
<div class="bar"><span id="prog">0 scored</span><button onclick="exportCsv()">Export CSV</button></div>
<script>
const CASES = __DATA__;
const QS = [
 ["q1","The highlighted region is plausible given this radiograph."],
 ["q2","The written report is consistent with what is displayed."],
 ["q3","The report avoids clinical claims the system cannot support."],
 ["q4","Overall, this output is acceptable for research and teaching use."]
];
const box = document.getElementById('cases');
CASES.forEach(c=>{
  const d=document.createElement('div'); d.className='case';
  d.innerHTML = `<strong>Case ${c.id}</strong>
   <div class="imgs">
     <figure><figcaption>Radiograph</figcaption><img src="${c.original}"></figure>
     <figure><figcaption>Model saliency overlay</figcaption><img src="${c.overlay}"></figure>
   </div>
   <div class="rep"><strong>Generated report</strong>\\n${esc(c.findings)}\\n\\n${esc(c.impression)}</div>
   ${QS.map(([k,q])=>`<div class="q">${q}</div><div class="scale">${
       [1,2,3,4,5].map(v=>`<label><input type="radio" name="${k}_${c.id}" value="${v}"><span>${v}</span></label>`).join('')
   }</div>`).join('')}
   <div class="q">Does any sentence read as a definitive diagnosis?</div>
   <div class="scale">${['No','Yes'].map(v=>`<label><input type="radio" name="dx_${c.id}" value="${v}"><span>${v}</span></label>`).join('')}</div>
   <div class="q">Comments (optional)</div>
   <textarea id="cm_${c.id}" rows="2"></textarea>
   <div class="verdict"><button class="ghost" onclick="rev(${c.id},this)">Reveal model verdict</button>
     <span id="v_${c.id}"></span></div>`;
  box.appendChild(d);
});
function esc(s){return (s||'').replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
function rev(id,btn){
  const c=CASES.find(x=>x.id===id);
  document.getElementById('v_'+id).textContent =
    ` Model: ${c.pred} (P(pneumonia)=${c.p}) · zone: ${c.zone}`;
  btn.style.display='none';
}
document.addEventListener('change',()=>{
  const n=CASES.filter(c=>document.querySelector(`input[name=q4_${c.id}]:checked`)).length;
  document.getElementById('prog').textContent=`${n} of ${CASES.length} scored`;
});
function exportCsv(){
  const g=(n)=>{const e=document.querySelector(`input[name=${n}]:checked`);return e?e.value:''};
  let rows=[['reviewer','qualification','years','case','truth','predicted','p_pneumonia','correct','q1_region_plausible','q2_consistent','q3_no_unsupported_claims','q4_acceptable','reads_as_diagnosis','comment'].join(',')];
  const nm=(document.getElementById('rev-name').value||'').replace(/,/g,' ');
  const ql=(document.getElementById('rev-qual').value||'').replace(/,/g,' ');
  const yr=document.getElementById('rev-yrs').value||'';
  CASES.forEach(c=>{
    const cm=(document.getElementById('cm_'+c.id).value||'').replace(/[,\\n]/g,' ');
    rows.push([nm,ql,yr,c.id,c.truth,c.pred,c.p,c.correct,g('q1_'+c.id),g('q2_'+c.id),g('q3_'+c.id),g('q4_'+c.id),g('dx_'+c.id),cm].join(','));
  });
  const b=new Blob([rows.join('\\n')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='rad_intel_clinical_review.csv';a.click();
}
</script></body></html>""".replace("__DATA__", data)


if __name__ == "__main__":
    main()
