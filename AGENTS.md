# Rad-Intel — context for an AI coding agent

Read this before touching anything. It is written for an assistant picking
the project up cold, and it is mostly about **not breaking things that are
already correct**.

---

## What this is

A final-year major project: pneumonia detection from chest radiographs, by
**Poorab N G**, Malnad College of Engineering. A DenseNet121 + Swin-T fusion
model with CBAM attention, served behind a FastAPI web app that has:

- a **public tool** — upload a radiograph, get a prediction, a Grad-CAM
  overlay and a written report
- a **closed clinician workspace** — named clinicians review cases against
  the model, and their reviews become training material
- an **administrator console** — accounts, activity log, training, and
  releasing a model to the public

There is an accompanying IEEE paper whose reported numbers come from this
code.

**The site is mid-deployment.** The code is finished and tested; the server
is being set up. The two documents that matter are `DEPLOY.md` (Azure, now)
and `MIGRATE.md` (the fallback, later).

---

## Hard rules — do not break these

These are the owner's standing instructions. They override convenience.

1. **Do not touch the paper.** No edits to the LaTeX, figures, bibliography,
   reported metrics, checkpoints, dataset split or evaluation outputs. The
   paper is submitted and locked.
2. **One inference path.** Serving and training use the same preprocessing,
   imported, never reimplemented. Do not add a second way to run the model
   — not a quantised copy, not an ONNX export, not a different torch
   version. The deployed model must produce what the paper reports.
3. **Never modify the Kaggle dataset or its split.** The 624-image test
   partition is the only thing making new models comparable to the reported
   ones. It is never trained on.
4. **`weights/best_hybrid_model.pt` is sacred.** It is the checkpoint the
   paper describes. Nothing writes to it. New models go to new filenames.
5. **Do not auto-retrain when feedback is submitted.** Training is always a
   deliberate, separate act.
6. **Secrets come from environment variables only.** Never in code, logs,
   screenshots or commits.
7. **No AI attribution in commits or PRs.** No `Co-Authored-By` trailers, no
   mention of any AI assistant in commit messages, file names or file
   contents. This was cleaned up once already at some cost; do not reintroduce
   it.
8. **Stop any dev server you start.** The owner often has one running on port
   8000; leave it alone.

---

## Layout

```
src/rad_intel/
  api/           FastAPI app, routes, rate limiting, model manager
    routes/      analyze, predict, explain, report, feedback, admin, auth,
                 health, history, assistant
    dependencies.py   ModelManager: loading, caching, thresholds
    ratelimit.py      per-identity and global limits
  auth/          accounts, sessions, one-time codes, crypto, email, activity
  models/        factory.py (architectures), registry.py (released models)
  preprocessing/ the single CLAHE + resize + normalise path
  training/      trainer.py, dataset.py — the paper's own recipe
  xai/           Grad-CAM, LIME, anatomical zones
  reporting/     report generation, Gemini client with offline fallback
  static/        the whole front end: index.html, app.js, workspace.js,
                 extras.js, style.css — vanilla JS, no build step
scripts/         preflight.py, manage_clinicians.py, connect_drive.py,
                 train_community_model.py, finetune_from_feedback.py
deploy/          radintel.service, Caddyfile, run-rad-intel.sh,
                 com.radintel.server.plist
tests/           ~325 tests; they are the specification
```

**Run the tests before and after any change:**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expect **325 passed, 2 skipped**. The skips are environmental (no Kaggle
dataset present locally).

---

## Facts you will need, measured not guessed

| | |
|---|---|
| Python | 3.11+ required, `.python-version` says 3.12 |
| PyTorch | **2.14.1+cpu** — the checkpoint was trained with it |
| Memory, server at rest | **871 MB** |
| Memory, after cycling every model | **~1.17 GB and it plateaus** |
| Grad-CAM path | loads a lung-segmentation model, **+380 MB** |
| Realistic ceiling | **~1.55 GB** → 2 GB host is right, 1 GB is not |
| Model files | 5, **494 MB** total |
| HTTP routes | 61 — 8 public, 9 optional-auth, 12 clinician, 32 admin |

**Model files** (`weights/`). The chooser offers only models whose file is
present:

| File | Size |
|---|---|
| `best_hybrid_model.pt` | 136 MB — the default, required |
| `best_hybrid_no_cbam_model.pt` | 136 MB |
| `best_swin_t_model.pt` | 105 MB |
| `best_resnet50_model.pt` | 90 MB |
| `best_densenet121_model.pt` | 27 MB — small enough to be in git |

Only **one** non-default model is held in memory at a time
(`RAD_INTEL_MODEL_CACHE`, default 1). Switching costs 200–700 ms and is
rate limited: 15 switches per person per 5 min, 40 overall per minute. The
default model is exempt by name and never rationed.

---

## Settings

All read from the process environment; `.env` beside the code is loaded by
`deploy/run-rad-intel.sh`. `.env.example` documents every one.

**Must be set in production:**

```
RAD_INTEL_SECRET_KEY      64 hex chars; losing it makes stored phone
                          numbers permanently unreadable
RAD_INTEL_COOKIE_SECURE=1 whenever the public address is HTTPS
RAD_INTEL_TRUSTED_PROXY   1 behind Caddy, empty behind nothing — see below
SMTP_*                    Gmail app password; mail is the second factor
GEMINI_API_KEY            optional; falls back to a rule-based writer
```

**Must stay empty in production:** `RAD_INTEL_ENABLE_DOCS`,
`RAD_INTEL_CORS_ORIGINS`, `RAD_INTEL_OTP_CHANNEL`, `RAD_INTEL_OTP_ECHO`.

**`RAD_INTEL_TRUSTED_PROXY` is not a guess.** Set it to 1 only when something
you control writes `X-Forwarded-For`. Caddy does, and ignores what clients
claim, so 1 is correct on Azure. Behind a tunnel, test first: open the site
from a phone on mobile data, sign in, and read the address in the console's
activity log. A real public address means set it; `127.0.0.1` means leave it
empty. Wrong either way is quiet — either anyone can forge an address and the
rate limits mean nothing, or every visitor counts as one person and one of
them locks out the rest.

**Deployment state is not in git:** `.env`, `.radintel_key`,
`weights/active_models.json`, `weights/released_models.json`.

---

## Deploying — Azure, now

Full instructions: **`DEPLOY.md`**. Shape of it:

- **Ubuntu 24.04** on an Azure **B1ms** (1 vCPU, 2 GB), Standard SSD,
  ports 22/80/443, paid for with the **Azure for Students** credit ($100,
  no credit card, ~5 months; the subscription is *disabled* when the credit
  runs out, never charged).
- `pip install -r requirements-deploy.txt` — this pins the **CPU** PyTorch
  wheels. The default would drag in 2–3 GB of CUDA libraries that cannot run.
- **systemd** (`deploy/radintel.service`) keeps it up. Install it with the
  `sed` that fills in `__USER__` and `__HOME__`, so the same file works on
  any machine.
- **Caddy** (`deploy/Caddyfile`) holds a free Let's Encrypt certificate for
  the DNS name Azure gives the VM. Rad-Intel itself listens on **127.0.0.1
  only** — Caddy is the one way in.
- `scripts/preflight.py` checks everything that can quietly be wrong and
  prints OK / FIX / CHECK. **Run it after any settings change.** It exits
  non-zero on any FIX.
- `scripts/connect_drive.py` connects Google Drive so reviewed radiographs
  survive the server. One scope, `drive.file` — per-file access to files the
  app created, enforced by Google, so it cannot see anything else in that
  Drive.

**Things that have already gone wrong here, so check them first:**

- The DNS name must be **copied from the portal**, never typed. It contains
  the region, and the wrong region is the commonest failure.
- The SSH key downloads to `~/Downloads/radintel_key.pem` and is offered
  once. `DEPLOY.md` Step 2 sets up an SSH config entry so everything
  afterwards is just `ssh radintel` / `scp … radintel:…`.
- Caddy failing to get a certificate is almost always port 80 closed in the
  Azure network rules.

---

## Migrating — the 2013 Intel Mac, later

Full instructions: **`MIGRATE.md`**. The one thing to know:

**That Mac must run Ubuntu, not macOS.** PyTorch has shipped no macOS Intel
build since 2.2; this project runs 2.14.1 and the checkpoint was trained with
it. Pinning torch back to 2.2 would drag `torchvision` and the rest down and
put the deployed model on a different numerical stack from the paper — which
rule 2 forbids. Ubuntu on that Mac gets the identical `manylinux_2_28_x86_64`
wheel the Azure server runs.

Because Azure is already Ubuntu, almost everything transfers: same torch,
same `deploy/radintel.service`, same `.env` but for one line. Only two things
differ — **Tailscale Funnel** replaces Caddy (a home connection has no public
address for Let's Encrypt to verify), and `RAD_INTEL_TRUSTED_PROXY` goes back
to empty until tested.

What moves: `rad_intel.db`, `.radintel_key`, `.env`, and the `.pt` files.
What does not: `feedback_store/` — every reviewed radiograph is in Google
Drive and is fetched back on demand.

---

## The two model lineages

Do not confuse them.

- **`best_hybrid_model.pt`** — the paper's model. Never written to.
- **`best_community_hybrid_v{n}_model.pt`** — a separate lineage, retrained
  from scratch on the public dataset *plus every reviewed case*, every run.
  Trained **off-server** on a free GPU (`COMMUNITY_MODEL.md`), uploaded back
  through the admin console, and only offered publicly when deliberately
  released. See `scripts/train_community_model.py`.

Training is **administrator-only**. Clinicians upload, correct and export;
export is frozen for everybody while a run is in progress.

---

## Conventions

- Comments explain **why**, not what. Match the surrounding density.
- Tests are prose about behaviour, not mechanical assertions. Read a few
  before writing one.
- Error messages are read by clinicians. Say what to do, not which check
  failed.
- Front end is vanilla JS with no build step. Bump the `?v=` query on
  `index.html`'s asset links when changing CSS or JS, or browsers cache it.
- Verify in a real browser or against a running server, not by reasoning.
  Several bugs in this project were found only that way and would have
  shipped otherwise.
