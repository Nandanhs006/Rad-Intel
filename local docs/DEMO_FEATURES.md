# Demo features: assistant, UI sections, clinician feedback

Added on top of the finalised system for presentation purposes. The paper, its
figures, the checkpoints, the dataset split and every evaluation output are
untouched — see "What was not changed" at the end.

---

## Setup

No new dependencies. The assistant reuses the Gemini client already in the
project, and the feedback store reuses the existing SQLite database.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | No | Phrases assistant answers and generates reports. Without it both fall back to deterministic text. |
| `GEMINI_MODEL` | No | Defaults to `gemini-3.5-flash-lite`. |
| `DEFAULT_MODEL` | No | Defaults to `hybrid`. |

The key is read server-side from `.env` via `rad_intel.config.settings` and is
never sent to the browser. `.env` is gitignored; confirm with
`git check-ignore -v .env` before committing.

### Run

```bash
.venv/bin/python -m uvicorn rad_intel.api.main:app --host 127.0.0.1 --port 8000
```

New directories are created on first use:

```
feedback_store/images/    reviewed radiographs, named by SHA-256
feedback_store/exports/   versioned JSONL training manifests
```

Both sit outside the Kaggle dataset directory. Add `feedback_store/` to
`.gitignore` if reviewed images should not be committed.

---

## 1. Ask & Learn — educational assistant

General questions about lung health, pneumonia, chest radiography, the terms
in the interface, and how the project works.

**Grounding.** Answers come from a curated knowledge base
(`src/rad_intel/assistant/knowledge_base.py`), not from the model's own recall.
Each entry carries a source (WHO, CDC, NIH NHLBI, or the project itself) and
the interface cites it. The language model is asked to phrase an answer *from
the retrieved extracts only*; with no key, or if the service fails, the
extracts are returned directly.

**Safety**, in `src/rad_intel/assistant/safety.py`, applied in order:

1. **Emergency language** — difficulty breathing, severe chest pain, blue
   lips, confusion, coughing blood, rapid worsening — short-circuits to an
   urgent-care instruction. This outranks every other rule.
2. **Personal medical requests** ("do I have pneumonia", "what antibiotic
   should I take") are declined without a verdict.
3. **Out-of-scope** questions are declined before any model call.
4. **Generated text is screened** before display; a diagnostic or prescriptive
   phrase causes the deterministic answer to be used instead.

**Isolation.** `POST /api/v1/assistant/chat` accepts JSON only and has no file
parameter, so a radiograph cannot reach the language service through it
(posting one returns 422). The on-screen result is sent only when the user
ticks the opt-in box, and then only as a short string — measured at 171 bytes,
containing the predicted label and probability, never image data.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/assistant/status` | GET | Whether the language service is configured |
| `/api/v1/assistant/chat` | POST | Answer a question |

---

## 2. Interface sections

Four tabs: **Analyse X-ray**, **Ask & Learn**, **Clinician Feedback**,
**About & Limitations**.

The analysis workflow is unchanged. Its markup was moved inside a tab panel
and `app.js` was not edited at all; the new behaviour lives in a separate
`static/extras.js`. Accent tokens moved from amber to a medical blue-teal by
changing six CSS variable *values* — no component rule was touched, so the
change reverts with `git checkout src/rad_intel/static/style.css`.

The About tab states the measured results and the known limitations as the
paper reports them, including the negative CBAM result and the shortcut-learning
finding.

---

## 3. Clinician Feedback

A review queue. **Nothing here retrains anything.**

A reviewer enters their name, picks a model version, uploads a radiograph, and
runs the normal prediction through the same preprocessing and the same
calibrated threshold as the rest of the service. They then mark the prediction
correct or incorrect, supply a corrected binary label when marking it
incorrect, and optionally add a note.

**Recorded per review:** model version, preprocessing version, UTC timestamps,
image SHA-256, stored image path, predicted label, probability, threshold,
verdict, correction, note, status, export tag.

**Validation:** JPEG or PNG only, 20 MB limit, non-empty and decodable,
verdict restricted to `correct`/`incorrect`, corrected label restricted to
`NORMAL`/`PNEUMONIA` and *required* when marking a prediction incorrect.

**Export.** "Create reviewed training export" writes
`feedback_store/exports/reviewed-export-<UTC timestamp>.jsonl` and marks those
records `exported`. Exported records are frozen: editing one returns 409. Each
line carries the resolved label and whether it came from a reviewer correction
or from the model being agreed with.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/feedback/warning` | GET | Upload notice and preprocessing version |
| `/api/v1/feedback/submit` | POST | Predict, then record the review |
| `/api/v1/feedback` | GET | List the queue with counts |
| `/api/v1/feedback/{id}` | PATCH | Edit a pending review |
| `/api/v1/feedback/export` | POST | Write a versioned manifest and freeze |
| `/api/v1/feedback/exports` | GET | List manifests |

### Why there is no "train" button

Fine-tuning from reviewer feedback is viable but is not safe to trigger from a
web request, so it was not built. Doing it properly would require, in order:

1. Take an export manifest as the only input.
2. Hold out the original 624-image Kaggle test partition untouched, so the new
   model stays comparable to the paper's numbers.
3. Split reviewed cases into their own train/validation sets; never merge them
   into the original split.
4. Fine-tune from an existing checkpoint into a **new** version
   (`weights/best_hybrid_v2_model.pt`), leaving the original in place.
5. Re-run `scripts/evaluate_all.py` on the untouched test partition and
   compare against the recorded baseline.
6. Require explicit human approval before the new version is selectable.

Steps 1–3 are satisfied by the export format. Steps 4–6 belong in an offline
script run deliberately, not in the serving process.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # 64 passed, 1 skipped
```

`tests/test_assistant.py` (29) covers emergency detection, personal-request
refusal, scope gating, emergency-outranks-scope, the no-provider fallback,
output screening, and source coverage. `tests/test_feedback.py` (8) covers
provenance recording, correction storage, editing, export tagging, queue
transitions, and isolation from the analyses table.

---

## What was not changed

- `Rad-Intel paper/` — LaTeX, figures, bibliography: untouched.
- `results/` — all measured metrics and probabilities: untouched.
- `weights/` — all five checkpoints: untouched.
- `Rad-Intel paper/chest_xray/` — verified at 1341/3875, 8/8, 234/390 after the work.
- Preprocessing: one path only. `tests/test_serving_contract.py` still asserts
  the serving and training pipelines build identical tensors.
- `app.js`, `explain.py`, `gradcam.py`, `lime_explainer.py`, the model factory
  and the trainer: not edited.

Verified after the changes: `/api/v1/predict` returns P(pneumonia) = 0.872739
on the reference radiograph, identical to the value recorded before the work.

## Known limitations

- The feedback workspace has no authentication. It is role-separated by
  interface only. Any deployment beyond a local demo needs real access control.
- The knowledge base is hand-written and small; out-of-scope questions are
  declined rather than answered from general knowledge, which is deliberate.
- Emergency detection is keyword-based. It is tuned to over-trigger, because a
  needless redirect to a clinician is the safe failure.
- Assistant latency follows the Gemini service and has been observed between
  2 and 90 seconds.
- LIME is exposed at `/api/v1/explain?method=lime` but returns no quadrant
  statistics; that is pre-existing behaviour and was left alone.
