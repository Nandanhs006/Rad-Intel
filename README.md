# Rad-Intel: Clinical Decision Support System for Automated Pneumonia Detection

> **Data-Driven Clinical Decision Support Framework for Automated Pneumonia Detection from Chest Radiographs**  
> Built on **Python 3.14**, PyTorch, torchvision, timm, FastAPI, ReportLab, and Google GenAI (Gemini Flash).

---

## 1. System Architecture Overview

```
                      +-----------------------------+
                      |   Chest Radiograph (CXR)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Preprocessing (CLAHE, 224p) |
                      +--------------+--------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
      +-------------------------+         +-------------------------+
      |  DenseNet121 Extractor  |         |  Swin-T Vision Transf.  |
      |   (Local Feature Maps)  |         |   (Global Context)      |
      |    (B, 1024, 7, 7)      |         |     (B, 768, 7, 7)      |
      +------------+------------+         +------------+------------+
                   |                                   |
                   +-----------------+-----------------+
                                     | (Channel Concat: 1792)
                                     v
                      +-----------------------------+
                      |  1x1 Projection Conv (512)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   CBAM Attention Module     |
                      | (Channel + Spatial Refine)  |
                      +--------------+--------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
      +-------------------------+         +-------------------------+
      |   Classification Head   |         |   Explainability Engine |
      |   (Pool + Dense Logits) |         | (Grad-CAM & LIME Zones) |
      +------------+------------+         +------------+------------+
                   |                                   |
                   +-----------------+-----------------+
                                     |
                                     v
                      +-----------------------------+
                      | Strict LaTeX LLM Generator  |
                      | (Gemini Flash / ACR Rules)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Publication-Grade PDF Engine|
                      | (ReportLab / pdflatex)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  FastAPI Endpoints + Web UI |
                      +-----------------------------+
```

---

## 2. Strict LaTeX Prompting & Clinical PDF Reporting

Rad-Intel synthesizes formal medical reports following the American College of Radiology (ACR) and RSNA structured reporting standards. Instead of loose markdown, reports are generated through **strict prompt rules for compilable LaTeX content**, which are then compiled into **publication-grade clinical PDF files**.

### Strict Prompt Rules (`prompts/latex_clinical_report_prompt.txt` / `src/rad_intel/reporting/latex_prompts.py`)
- **Strict LaTeX Syntax**: Rigid character escaping (`\%`, `\&`, `\_`, `\#`, `\$`), standard packages (`geometry`, `tcolorbox`, `tabularx`, `booktabs`), zero extraneous conversational commentary.
- **ACR Clinical Hierarchy**:
  1. Institution & Study Header (Rad-Intel CDS, Accession ID, Timestamp)
  2. Patient Demographics & Modality protocol
  3. AI Diagnostic Verdict & Confidence %
  4. Explainable AI (Grad-CAM) 4-Quadrant Saliency Distribution (RUZ, RLZ, LUZ, LLZ)
  5. `\section*{EXAMINATION}`
  6. `\section*{CLINICAL INDICATION}`
  7. `\section*{TECHNIQUE}`
  8. `\section*{FINDINGS}` (Airspaces, Pleura, Cardiomediastinum, Osseous structures)
  9. `\section*{IMPRESSION}` (Definitive clinical conclusions)
  10. `\section*{RECOMMENDATIONS}` (Actionable follow-up & cultures)
  11. Regulatory Medicolegal Verification Notice (`tcolorbox`)
  12. Attending Radiologist Attestation block
- **Anti-Hallucination Guardrails**:
  - Grounds every statement exclusively on provided quantitative predictions and Grad-CAM lung zone metrics.
  - Strictly prohibits fabricating prior imaging, unmentioned patient history, or non-existent laboratory values.
  - Normal radiographs strictly report clear lung parenchyma without consolidation or effusion.
  - Pneumonia findings accurately pinpoint consolidation to the detected dominant zone(s).

### Dual-Engine PDF Compiler (`src/rad_intel/reporting/pdf_generator.py`)
- **Native TeX Compiler**: Invokes system `pdflatex` or `xelatex` if present on the host environment.
- **ReportLab Medical Document Engine**: Self-contained, zero-dependency medical PDF renderer delivering clean hospital branding, tabular demographic layout, color-coded diagnostic badges (Emerald for Normal, Crimson for Pneumonia), Grad-CAM heat-intensity tables, itemized clinical findings, and physician signature fields.

---

## 3. Testing & Verification

You can verify each stage individually from the project root using the provided virtual environment:

### Stage 1: Environment, Dependencies & Configuration
```powershell
.\.venv\Scripts\python.exe scripts/test_stage1_env.py
```

### Stage 2: Deep Learning Model Architecture
```powershell
.\.venv\Scripts\python.exe scripts/test_stage2_model.py
```

### Stage 3: Medical Preprocessing & Explainability (XAI)
```powershell
.\.venv\Scripts\python.exe scripts/test_stage3_xai.py
```

### Stage 4: LLM Clinical LaTeX & PDF Report Generation
```powershell
.\.venv\Scripts\python.exe scripts/test_stage4_report.py
```

### Stage 5: FastAPI End-to-End API Integration
```powershell
.\.venv\Scripts\python.exe scripts/test_stage5_api.py
```

### Running the Full Automated Pytest Suite (20 Tests)
```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 4. Running the Web Application & REST API

To launch the FastAPI development server:
```powershell
.\.venv\Scripts\uvicorn.exe src.rad_intel.api.main:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive Web Interface**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 5. REST API Endpoint Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` or `/api/v1/health` | Service health, PyTorch version, active hardware device |
| `GET` | `/api/v1/models` | List supported models, ablation variants, and parameter counts |
| `POST` | `/api/v1/predict` | Upload CXR image -> returns `NORMAL` or `PNEUMONIA` + confidence |
| `POST` | `/api/v1/explain` | Upload CXR image -> returns Grad-CAM / LIME overlay + quadrant metrics |
| `POST` | `/api/v1/report` | Translates model metrics + XAI into structured LaTeX report & base64 PDF |
| `POST` | `/api/v1/report/pdf` | Streams official binary clinical PDF report (`application/pdf`) |
| `POST` | `/api/v1/analyze` | Flagship all-in-one pipeline (Upload -> Predict -> Explain -> LaTeX -> PDF -> Persist) |
| `GET` | `/api/v1/analyze/{id}/pdf` | Downloads the generated clinical PDF report for a completed analysis |
| `GET` | `/api/v1/history` | Paginated query history of patient analyses from SQLite database |
| `GET` | `/api/v1/history/{id}` | Detailed inspection of a specific past diagnostic scan |
| `POST` | `/api/v1/assistant/chat` | Retrieval-grounded educational assistant (public, safety-gated) |
| `POST` | `/api/v1/auth/code` | Clinician: email + Rad-Intel ID -> one-time code to their phone |
| `POST` | `/api/v1/auth/admin/code` | Administrator: email + password -> one-time code |
| `POST` | `/api/v1/auth/login` | Exchanges a one-time code for a session |
| `POST` | `/api/v1/auth/request-access` | Records a request to join. Creates no account |
| `GET` | `/api/v1/feedback` | The review queue. `scope=mine`, `sort=name` |
| `POST` | `/api/v1/feedback/submit` | Runs the shared model and files the reviewer's verdict |
| `PATCH`/`DELETE` | `/api/v1/feedback/{id}` | Correct or remove a case. Its author, or an administrator |
| `POST` | `/api/v1/feedback/export` | Freezes pending cases into a versioned manifest |
| `POST` | `/api/v1/feedback/train` | Starts the one fine-tuning run; `/train/cancel`, `/train/status` |
| `GET` | `/api/v1/feedback/models` | Every version of the shared model and which one is served |
| `GET` | `/api/v1/admin/*` | Accounts, access requests, sessions, activity log. Administrator only |

Everything under `/api/v1/feedback` and `/api/v1/admin` requires a session; the rest is public.

---

## 6. Environment Variables (`.env`)

```ini
# Gemini API Key (optional for offline testing; required for live Google GenAI reports)
GEMINI_API_KEY=

# Hardware Device: "auto", "cpu", or "cuda"
DEVICE=auto

# Default Model: "hybrid", "densenet121", "swin_t", "hybrid_no_cbam", "resnet50"
DEFAULT_MODEL=hybrid

# Port and Host
HOST=127.0.0.1
PORT=8000
```
---

## 7. The Clinician Review Workspace

A closed area of the web interface where named clinicians review cases against
the one shared model and correct it where it is wrong. It is not public, there
is no sign-up page, and an account exists only because an administrator
created it.

### How somebody gets in

1. They open the **Clinician Feedback** tab, read what the study stores and
   who can see it, and submit a request: name, institution, email and a phone
   number. This creates no account.
2. The request is recorded and — when SMTP is configured — emailed to the
   administrator. It also appears in the administrator console.
3. The administrator verifies their medical registration **independently**,
   then creates the account and hands over the generated Rad-Intel ID in
   person or by a channel the clinician already trusts.

If SMTP is not configured, or its password has expired, the email simply does
not go out — the applicant is told the administrator will be in touch either
way, which is true, so the failure cannot surface on their side. The console
is therefore explicit about it: requests that never reached an inbox are
counted in the warning at the top, marked **not emailed** in the list, and can
be sent with one button once mail works. **Send myself a test email** checks
the configuration without inventing an applicant.

### Logging in

Two factors, two requests, every time:

| Who | First factor | Second factor |
|---|---|---|
| Clinician | email + Rad-Intel ID | one-time code to the registered phone |
| Administrator | email + password | one-time code to the registered phone |

A clinician has no password: the Rad-Intel ID their administrator issued is
their first factor, so there is nothing for them to reuse from another site
and nothing to phish out of them with a password form. The code is twelve
characters, single-use, expires in ten minutes, is never issued twice to the
same person, and is stored only as a hash.

The code goes **by email**, to the address on the account. There is no SMS:
India's DLT rules put domestic application-to-person messaging behind weeks
of registration, and it bought nothing email does not already do here. Codes
are never sent over WhatsApp either — the phone number on a record is how the
administrator reaches that clinician personally, and a one-time code does not
belong in a chat account that may be open on a shared laptop.

Pressing *Send verification code* again within a minute does not mint a
second code or count against the allowance; it says to look in the inbox.
Beyond that an account may ask for twelve codes in fifteen minutes.

**Which door you come in by decides what you can do.** An administrator who
signs in with their Rad-Intel ID holds a *clinician* session and sees the
clinician workspace; the console needs a session opened with the
administrator password. A Rad-Intel ID is written down and handed over, so it
must not reach the controls that read and delete everybody's data. The mode
is recorded when the code is issued, where the first factor was checked, so
submitting the code cannot upgrade the session.

If mail breaks, nobody can log in — including the administrator. Set
`RAD_INTEL_OTP_CHANNEL=log` to print codes to the server's own terminal, or
use the console command below.

**Send myself a test email** in the console reports the mail server's own
words on failure. Those words usually name the exact remedy — a Gmail
rejection says in as many terms that it wants a 16-character App Password —
and they never reach the login form: the study's mail configuration is not
the business of the clinician trying to log in.

### Locked out

If mail is not working, nobody can log in — including the administrator.
Rather than leaving that to be discovered, there is a way back:

```bash
python scripts/manage_clinicians.py code --email you@example.org
python scripts/manage_clinicians.py code --email you@example.org --admin
```

It prints an ordinary sign-in code — single use, ten minutes, same
verification as any other — and sends it nowhere. In the login form, press
**I already have a code** instead of **Send verification code**, and type it
in. Running it needs a shell on the machine the service runs on, and anybody
with that can already read the database; what it does not need is a working
provider. It is written to the activity log, and it is the one path that
ignores the per-account code-request limit, because a code that is delivered
to nobody cannot flood anybody's phone.

`RAD_INTEL_OTP_CHANNEL=log` does the same thing through the web interface,
printing codes to the server's terminal.

If a code cannot be delivered the sign-in is refused. It is never shown on
screen: a system that falls back to displaying the code when its provider is
down has a second factor in name only.

### Administering it

```bash
# Create an account and print the Rad-Intel ID to hand over
python scripts/manage_clinicians.py add --email a@b.org --name "Dr A" \
    --institution "X Hospital" --phone "+91 98765 43210"

python scripts/manage_clinicians.py list          # everyone
python scripts/manage_clinicians.py show   --email a@b.org
python scripts/manage_clinicians.py requests      # people waiting
python scripts/manage_clinicians.py new-id --email a@b.org   # lost ID
python scripts/manage_clinicians.py phone  --email a@b.org --phone "+91 ..."
python scripts/manage_clinicians.py disable --email a@b.org  # takes effect at once
python scripts/manage_clinicians.py delete  --email a@b.org
python scripts/manage_clinicians.py migrate       # IDs for pre-existing accounts
python scripts/manage_clinicians.py delivery      # is anything actually configured?
```

Everything above is also in the administrator console inside the workspace,
along with the activity log, the open sessions and the access requests.

### What each participant can do

| | Clinician | Administrator |
|---|---|---|
| See every case from every site | yes | yes |
| Edit or delete **their own** cases | yes | yes |
| Edit or delete **anybody's** cases | no | yes |
| Change the image on a filed case | no | no — delete and re-upload |
| Create an export, start a fine-tuning run | yes | yes |
| Cancel a run | only one they started | any |
| Create, edit, disable or delete accounts | no | yes |
| Read the activity log and open sessions | no | yes |
| Choose which model version is served | no | yes |

Other clinicians see only a participant's **name and institution**. Email
addresses, phone numbers and Rad-Intel IDs are not in that response at all.

### Training on reviewed cases

Submitting a review changes nothing. Reviews are exported to a versioned
manifest, which freezes them, and a fine-tuning run reads that manifest.

There is **one shared model**, so there is one run at a time. While a run is
going, everybody can keep uploading and correcting cases, but exporting and
training are held and the interface says who is holding them. The run can be
cancelled by the clinician who started it or by the administrator, and it
cancels itself if the browser that started it stops reporting in for three
minutes — training happens on the server, so a clinician's connection cannot
interrupt the computation, but a run nobody is watching should not keep the
queue locked.

A run writes `weights/best_hybrid_v<N>_model.pt` and never overwrites the base
checkpoint. **Nothing is served until an administrator activates it** from the
"shared model" panel, which writes `weights/active_models.json` rather than
copying over the base file — so the version the project's reported results
describe stays byte-identical on disk and switching back cannot fail.

To try the shared model without filing a review, use the **Analyse X-ray**
tab: it runs whichever version is currently active.

### Where images live, and where they go when this is hosted

`feedback_store/images/<sha256>.jpg`, one file per distinct radiograph, so
the same image submitted twice costs nothing extra. The database holds only
the hash and the path. They are served to logged-in clinicians through an
authenticated endpoint that takes a record UUID, not a path, and the resolved
path is checked against the store before anything is read — a URL alone is
useless without a session. Deleting a case removes its image unless another
case still refers to the same file.

**`RAD_INTEL_FEEDBACK_DIR` moves the whole store**, which is what hosting
needs. Most small hosts give a container a filesystem that is wiped on every
redeploy: the code comes back, the radiographs do not. Point this at
something that survives:

```ini
RAD_INTEL_FEEDBACK_DIR=/data/rad-intel/feedback_store
```

Images and export manifests both live under it. Paths recorded before the
move stay valid, because a path inside the project is still stored relative
to it; anything written afterwards is stored absolute.

What to point it at, cheapest first:

| | Good for | Watch out for |
|---|---|---|
| **A mounted disk** on the host (Railway/Render/Fly volume, or a VPS) | the obvious answer: no extra account, no API, ordinary file I/O | costs a little per GB; a free tier may not offer one |
| **A Google Drive folder mounted with `rclone`** | free up to the Drive quota, and it is the one account already in use here | it is a network filesystem — slower, and a dropped mount looks like missing images, so training should not run against it directly |
| **Object storage** (S3, R2, GCS) | the right answer at scale | needs an SDK and a rewrite of the store; not worth it for a study |

A Drive mount works **without any code change**: `rclone mount` it, point
`RAD_INTEL_FEEDBACK_DIR` at the mount, and the service cannot tell the
difference. Keep the folder private and shared with nobody — Drive sharing
is the whole access-control story at that point, and this holds radiographs.
Sizing is not the constraint: the Kaggle set is a few GB and reviewed cases
accumulate at a handful of megabytes a week, so 8 GB is years of room.

The **database** (`rad_intel.db`) is a separate question with the same
answer: it holds accounts, sessions and the review queue, and it must sit on
something that survives a redeploy too. It is small. Do not put SQLite on a
network mount — put it on a real disk, and the images wherever is convenient.

### Security, stated plainly

What is in place:

- Passwords and one-time codes are hashed (scrypt and a keyed digest); neither
  is recoverable from the database.
- Rad-Intel IDs and phone numbers are AES-256-GCM encrypted at rest, with a
  keyed blind index so exact lookup still works without decrypting the table.
- Sessions are server-side. The cookie carries a random token, `HttpOnly`,
  `SameSite=Lax`, and **no Max-Age**, so it goes when the browser closes; the
  server keeps a twelve-hour ceiling for browsers that restore their tabs. An
  account is signed in on **one device at a time**: logging in anywhere closes
  the session open elsewhere, the person is told, and it is logged. Logging
  out, being disabled, or having a Rad-Intel ID rotated all end a session
  immediately and require a fresh code.
- Every action against shared data is logged against a name and readable only
  by the administrator.
- Rate limits on sign-in, code requests, access requests, submissions,
  exports, training runs and the public analysis endpoints.

What is **not** in place, and should be before this is anything but a study:

- No TLS by default — set `RAD_INTEL_COOKIE_SECURE=1` once there is.
- No encryption of the stored radiographs themselves, and no automatic
  de-identification. That is the uploading clinician's responsibility, and the
  consent panel says so.
- **No second factor beyond the mailbox.** A code to an account's email is
  only as strong as that mailbox; somebody who can read it, and who has the
  Rad-Intel ID, is in. The one-device rule makes that visible — the real
  owner gets logged out — but it does not prevent it.
- No hardware tokens, no IP allow-list, no intrusion detection.
- Rate-limit counters live in the process, which is correct for one Uvicorn
  worker and would need shared storage before running several.

Set `RAD_INTEL_SECRET_KEY` in any real deployment. Without it a key is
generated into `.radintel_key`; that works, but it is one more file to keep
and back up, and losing it makes stored phone numbers unreadable.

---

## 8. Environment Variables for the Workspace

These are read from the process environment rather than through the settings
object, so no endpoint or log can return them. See `.env.example` for the full
list with comments.

```ini
RAD_INTEL_SECRET_KEY=        # signs sessions, encrypts phone numbers
RAD_INTEL_COOKIE_SECURE=     # 1 behind HTTPS

# one-time codes and notifications: email, through one mail server.
RAD_INTEL_ADMIN_EMAIL=       # where access requests are sent
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=               # a 16-character Gmail App Password
SMTP_FROM=

# "log" prints codes to the server's own terminal and sends them nowhere:
# for a demonstration, and for getting back in when mail has broken.
RAD_INTEL_OTP_CHANNEL=

# LOCAL TESTING ONLY: with nothing else set, write codes to the server's log
RAD_INTEL_OTP_ECHO=

# where access requests are emailed
RAD_INTEL_ADMIN_EMAIL=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=
```
