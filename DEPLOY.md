# Putting Rad-Intel online

Follow this top to bottom. Every step says what to do and how to tell it
worked. Nothing here needs you to understand the code.

Budget about **two hours** the first time.

---

## Before you start, understand one thing

Your project is three pieces that must all end up on the server:

| Piece | What it is | Size |
|---|---|---|
| **The code** | what you push to GitHub | small |
| **The model file** | `weights/best_hybrid_model.pt` | **136 MB** |
| **The data** | `rad_intel.db` + `feedback_store/` | small, but must survive |

The code goes up with git. **The model file does not** — it is in
`.gitignore`, because GitHub refuses files that big. If you forget it, your
site will load perfectly and every single analysis will fail.

That is the one mistake everybody makes. Step 3 is how you avoid it.

---

## Step 1 — Pick a host

You need a host that gives you:

- **2 GB of memory** (the model needs roughly 1–1.5 GB just to load)
- **a disk that keeps files between deploys** (called a *volume*)
- **HTTPS** (every host below gives this free)

Free tiers usually give 512 MB of memory and **no** permanent disk. Your app
will crash on startup, or lose every uploaded radiograph on each redeploy.
This needs a paid tier — roughly **₹400–900 a month**.

Reasonable choices:

- **Railway** — simplest. Connect GitHub, add a volume, done.
- **Render** — similar, slightly more clicking.
- **A small VPS** (DigitalOcean, Hetzner) — cheapest and most control, but
  you install things yourself.

The rest of this guide assumes Railway. Render is nearly identical.

---

## Step 2 — Push your code

```bash
cd ~/Documents/GitHub/Rad-Intel
git push origin poorab
```

On Railway: **New Project → Deploy from GitHub → pick Rad-Intel → branch
`poorab`**.

It will try to build and probably fail or crash. **That is expected.** It has
no settings and no model yet. Carry on.

---

## Step 3 — Add a disk, and put the model on it

In Railway: your service → **Variables/Settings → Volumes → New Volume**.
Mount it at:

```
/data
```

Give it **2 GB**.

Now get the model file onto that disk. Pick whichever you can manage:

**Option A — upload it directly (simplest)**

Railway gives you a shell (`railway run bash`, or the web terminal). From
your own machine:

```bash
railway run bash -c "mkdir -p /data/weights"
# then copy the file up using railway's file transfer, or option B
```

**Option B — put it in Google Drive and download it on the server**

1. Upload `weights/best_hybrid_model.pt` to your Drive.
2. Right-click → Share → **Anyone with the link** → copy the link.
3. Pull out the long id from the link.
4. On the server, once:

```bash
pip install gdown
gdown <THE_ID> -O /data/weights/best_hybrid_model.pt
```

5. **Set the Drive file back to private** when the download is done.

**How to tell it worked**

```bash
ls -lh /data/weights/
```

You should see `best_hybrid_model.pt` at about **136M**. Only this one file
is needed. The other four are for your paper, not for running the site.

---

## Step 4 — Fill in the settings

Railway → your service → **Variables**. Add each of these.

**Where your data lives** (so it survives a redeploy):

```
RAD_INTEL_FEEDBACK_DIR = /data/feedback_store
DATABASE_URL           = sqlite+aiosqlite:////data/rad_intel.db
MODEL_WEIGHTS_PATH     = /data/weights/best_hybrid_model.pt
```

**Security** — generate the key by running this on your own machine and
pasting the result:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

```
RAD_INTEL_SECRET_KEY    = <the long string you just generated>
RAD_INTEL_COOKIE_SECURE = 1
RAD_INTEL_TRUSTED_PROXY = 1
```

> `RAD_INTEL_TRUSTED_PROXY=1` matters on Railway, Render, Cloudflare or any
> load balancer. Without it every request looks like it came from the proxy,
> so **one person can use up everybody's rate limit** and the activity log
> records the wrong address. Leave it empty only if people reach the machine
> directly.

> Write that key down somewhere safe. If you lose it, every stored phone
> number becomes permanently unreadable.

**Email** — copy these from your own `.env` file:

```
RAD_INTEL_ADMIN_EMAIL = poorabng@gmail.com
SMTP_HOST             = smtp.gmail.com
SMTP_PORT             = 587
SMTP_USER             = poorabng@gmail.com
SMTP_PASSWORD         = <your 16-character Gmail App Password>
SMTP_FROM             = poorabng@gmail.com
```

**Reports:**

```
GEMINI_API_KEY = <copy from your .env>
```

**Google Drive** — where reviewed radiographs go so they outlive the server.
See Step 4b below for how to get these three.

```
GOOGLE_DRIVE_CLIENT_ID     = <from Step 4b>
GOOGLE_DRIVE_CLIENT_SECRET = <from Step 4b>
GOOGLE_DRIVE_REFRESH_TOKEN = <from Step 4b>
```

**Leave these empty.** They are only for development:

```
RAD_INTEL_ENABLE_DOCS   (empty)
RAD_INTEL_CORS_ORIGINS  (empty)
RAD_INTEL_OTP_CHANNEL   (empty)
RAD_INTEL_OTP_ECHO      (empty)
```

---

## Step 4b — Put the radiographs somewhere permanent

**Skip this and the site still works.** Reviewed radiographs are kept on the
server's disk. The problem is that the disk is rented by the month, most hosts
wipe it on every redeploy, and a radiograph a clinician uploaded is the one
thing here that **cannot be made again**. The model can be retrained. The
database can be rebuilt. A lost radiograph is lost.

So each one is also copied into **your** Google Drive, and fetched back from
there if the server ever comes up with an empty disk.

### What it can see

One permission: `drive.file` — access to files the app created itself.

That is not a promise to behave. Google's API **will not return** anything
else. Your documents, your photos, the `.pt` checkpoints you keep in the same
Drive: this connection cannot read them, cannot list them, cannot tell they
exist. It makes one folder, *Rad-Intel reviewed radiographs*, and that folder
is the entire world it has.

### Setting it up, once

1. **console.cloud.google.com** → new project, any name.
2. **APIs & Services → Library** → search *Google Drive API* → **Enable**.
3. **APIs & Services → OAuth consent screen**:
   - User type **External**
   - App name, your email, your email again → **Save**
   - **Publish app** → Confirm.

   > **Publishing matters.** An app left in *Testing* has its token expired by
   > Google after **seven days**, so uploads start failing a week after you
   > deploy. `drive.file` is not a sensitive scope, so publishing needs no
   > review and no verification — it is one button.
4. **Credentials → Create credentials → OAuth client ID → Desktop app.**
   Copy the client ID and client secret.
5. On your own machine:

```bash
python scripts/connect_drive.py --client-id YOUR_ID --client-secret YOUR_SECRET
```

A browser opens. Sign in as the account whose Drive should hold the
radiographs, allow the one permission, and it prints three lines. Put them in
the host's Variables, as in Step 4.

**How to tell it worked:** the console gets a card, *Where reviewed radiographs
are kept*, saying they are copied to your Drive, with a link to the folder.

**If you connect it after people have already reviewed cases:** that card
offers **Copy what is missing**. Press it once.

**To disconnect:** myaccount.google.com/permissions. The files stay; they are
yours.

---

## Step 5 — Tell it how to start

Railway → **Settings → Build Command**:

```
pip install -r requirements-deploy.txt
```

That one line saves two to three gigabytes. By default pip installs the
**CUDA** build of PyTorch — NVIDIA libraries that will never run on a host
with no graphics card, but still have to be downloaded, stored and shipped on
every deploy. `requirements-deploy.txt` points it at the CPU wheels instead.
The site behaves identically.

Railway → **Settings → Start Command**:

```
uvicorn rad_intel.api.main:app --host 0.0.0.0 --port $PORT --workers 1
```

`--workers 1` matters. The limits that stop people hammering the site are
counted inside one process; two processes means two separate budgets.

Redeploy. Watch the logs. **The first start takes a minute or two** while the
model loads — that is normal, not a hang.

---

## Step 6 — Check it actually works

**First, one command.** On the server shell:

```bash
python scripts/preflight.py
```

It goes through everything that can quietly be wrong at the same time — the
key, the cookie, the proxy, the model file, the mail settings, where the
review store lives, whether an administrator exists, whether Drive is
connected — and prints each as **OK**, **FIX** or **CHECK**, with what to do
about it. Nothing says FIX, you are ready. Add `--email` and it sends you a
real test message too.

Then open your URL and go through these in order:

**1. Is it alive?** Visit `your-url/api/v1/health`. You want:

```json
{"status": "ok", "gemini_reporting_live": true}
```

**2. Does the model work?** On the site: **Analyse X-ray → Load Pneumonia
CXR → Analyze**. You should get a verdict, a heat map and a written report.

> If you get *"Model has no trained checkpoint"* — the file from Step 3 is
> missing or in the wrong place. Go back and check `ls -lh /data/weights/`.

**3. Can you log in?** **Clinician Feedback → Administrator →** your email and
password → a code arrives in your Gmail → paste it in.

> If no code arrives, your SMTP settings are wrong. Check your spam folder
> first.

**4. Does mail work from the server?** In the console: **Send myself a test
email**. It should say it was sent.

**5. Is it private?** Visit `your-url/docs` — you should get **404**. If you
see a page of API documentation, `RAD_INTEL_ENABLE_DOCS` is set; empty it.

---

## Step 7 — Make yourself the administrator on the server

Your admin account exists in your laptop's database, not the server's. Open
the server shell and create it there:

```bash
python scripts/manage_clinicians.py add \
  --email poorabng@gmail.com \
  --name "Poorab N G" \
  --institution "Malnad College of Engineering" \
  --phone "+91 9844407409" \
  --role admin
```

It prints a password and a Rad-Intel ID. **Copy both.** Log in with the
password through the **Administrator** door.

If anything goes wrong and you cannot log in, this always works from the
server shell:

```bash
python scripts/manage_clinicians.py code --email poorabng@gmail.com --admin
```

It prints a code. On the login page press **"I already have a code"**, paste
it in. No email needed.

---

## Step 8 — Add your first clinician

In the console: **Add a clinician** → fill in their details → **Generate** →
**Create account**.

A Rad-Intel ID appears in an orange box. **Give it to them in person or over
WhatsApp — not in the same message as the website link.** The ID and their
email are the two halves of their login; keeping them in separate places is
the whole point.

They then log in with: their email + that ID + a code emailed to them.

---

## That's it. You are live.

---

# Day-to-day, afterwards

## Someone asks for access

They press **Request clinician access** on the site. It emails you and
appears in your console. You check they are who they say, then **Use these
details → Generate → Create account**, and hand over the ID.

## A clinician loses their Rad-Intel ID

Console → find them → **New ID**. The old one dies immediately and they are
logged out. Give them the new one.

## Training a better model

One way, three steps, all in the console.

1. Clinicians review cases (upload, mark right or wrong) and press **Create
   training export** when a batch is worth keeping. That is their save point,
   and it is the only part of this they do.
2. **You** press **Build training bundle** — one zip with every reviewed
   radiograph and its label.
3. You train it on a free GPU, then paste the resulting file's share link
   back into the console.

This produces the **community model**: a second model kept completely apart
from the one your paper reports. Each run throws the previous community
version away and retrains the same architecture *from scratch* on the public
dataset plus **every** case collected so far, so a version is built from all
the evidence rather than carrying the drift of every run before it.

**[COMMUNITY_MODEL.md](COMMUNITY_MODEL.md)** walks through all of it in plain
language, with the notebook cells to paste.

> It does not take the site down. The training happens on a borrowed GPU,
> nowhere near your server, and the server keeps serving throughout.

## Releasing a model to the public

New in the console: **Release a model to the public**.

It lists every version you have produced — fine-tuned ones and community
ones — with its scores: accuracy, AUROC, precision, sensitivity, specificity,
F1, measured on the same 624 held-out radiographs as the models in your paper,
so you can compare like with like.

Press **Release to public**. It asks for your **password, every single
time**. Then that version appears in the model dropdown on **Analyse X-ray**
for everyone, labelled *"released from reviewed cases"* with its scores.

Two things that do not happen:

- **The default does not change.** Visitors still get the complete classifier
  unless they deliberately pick the new one.
- **Nobody's radiographs are published.** Only a model whose weights were
  shaped by them.

**Withdraw from public** takes it back off the list. The file stays on disk.

One thing the release panel deliberately cannot do: make a community model the
*default*. It reaches people one way only — released by name, chosen
deliberately on the Analyse tab. The model a visitor gets without choosing
anything is always the one your paper describes.

---

# Things to know, not things to do

- **If Gmail breaks, nobody can log in.** The command in Step 7 is your way
  back in. Keep it somewhere you can find it.
- **The mailbox is the second factor.** Someone with a clinician's Rad-Intel
  ID *and* access to their email is in. That is the design; it is reasonable
  for a study.
- **A session ends when the browser closes**, or after 7 days. One device at
  a time — logging in somewhere else logs out the old one.
- **There is no idle timeout.** An unlocked laptop is an open account until
  the browser closes.
- **Run preflight after any settings change.** `python scripts/preflight.py`.
  It is the quickest way to find out that something you changed last week
  broke sign-in for everybody.
- **Back up `/data`.** It holds every account and every reviewed radiograph.
  A monthly copy is enough at this scale. With Step 4b done, the radiographs
  are already safe in Drive — what a backup protects is the accounts and the
  reviews themselves.
