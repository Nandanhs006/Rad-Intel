# Putting Rad-Intel online

You are hosting this yourself, on the spare MacBook, and reaching it through
a free tunnel. No card, no monthly bill, no server to administer.

Follow this top to bottom. Every step says what to do and how to tell it
worked. Nothing here needs you to understand the code.

Budget about **two hours** the first time.

---

## Why this way

The free tiers that could have carried this have mostly closed. Render's
free plan gives 512 MB of memory and this needs about 870 MB. Fly.io and
Koyeb stopped offering free plans to new accounts. Hugging Face now requires
a paid plan for the kind of Space this would be. What is left is a machine
you already own.

That turns out to be a decent answer rather than a consolation. The disk is
permanent, so nothing is wiped on a redeploy and there is no volume to
configure. There is no monthly cost to forget about. And the model files
never have to be uploaded anywhere — they are already on the machine.

**The one real cost:** the site is up only while that MacBook is on, awake
and online. For a study with a handful of clinicians, that is usually fine.
It is not a service other people can depend on at three in the morning, and
you should tell them so.

---

## What you need

- The **spare Intel MacBook**, on its charger, on your wifi.
- About **3 GB free disk**, and the project copied onto it.
- A free **Tailscale** account — no card.

### A word about 4 GB

Measured: Rad-Intel sits at about **870 MB** once started. macOS itself wants
roughly 2 GB. That leaves the machine working, with some swapping, and it
will hold.

Two things keep it there:

- **Put only `best_hybrid_model.pt` on it.** Models without a checkpoint are
  simply not offered, so the chooser shows one model and nothing ever gets
  loaded and unloaded. Step 2 says where to put it.
- **Leave `RAD_INTEL_MODEL_CACHE` empty.** The default already holds one
  model at a time.

Analyses will take a few seconds rather than under one, because the
processor is old. That is the trade, and it is a fair one.

---

## Step 1 — Get the project onto that MacBook

On the spare machine, install the tools, then the project:

```bash
xcode-select --install
```

Install Python 3.12 or newer from [python.org](https://www.python.org/downloads/macos/)
if `python3 --version` shows anything older. Then:

```bash
mkdir -p ~/Documents/GitHub && cd ~/Documents/GitHub
git clone https://github.com/YOUR_USERNAME/Rad-Intel.git
cd Rad-Intel
python3 -m venv .venv
.venv/bin/pip install -r requirements-deploy.txt
```

That last line takes ten to twenty minutes and downloads about a gigabyte.
It installs the **CPU** build of PyTorch, which is the right one here and
several gigabytes smaller than the default.

**How to tell it worked**

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
```

---

## Step 2 — Put the model file in place

The model is not in git — it is too big for GitHub. Copy
`weights/best_hybrid_model.pt` across from your main machine: AirDrop, a USB
stick, or your Google Drive. It goes in `~/Documents/GitHub/Rad-Intel/weights/`.

**How to tell it worked**

```bash
ls -lh ~/Documents/GitHub/Rad-Intel/weights/
```

You want `best_hybrid_model.pt` at about **136M**.

> You can copy the other four `.pt` files too, and the Analyse tab will then
> offer those models as well. On 4 GB, start with just the one. You can
> always add them later.

---

## Step 3 — Give it a public address

[Tailscale](https://tailscale.com) will carry traffic from the internet to
this machine without opening a single port on your router — which also means
it works on a home connection that has no public address of its own.

1. Make a free account at **tailscale.com** and install Tailscale on the
   MacBook. Sign in.
2. In the Tailscale admin console → **DNS**, note your tailnet name. It looks
   like `tail1234.ts.net`.
3. Still in the admin console → **Access controls**, make sure `funnel` is
   enabled for your devices. On a new tailnet it usually already is.
4. On the MacBook:

```bash
tailscale funnel --bg 8000
```

> **If `tailscale` is not found**, the Mac app keeps its command somewhere
> the shell does not look. Add it once:
>
> ```bash
> echo 'alias tailscale="/Applications/Tailscale.app/Contents/MacOS/Tailscale"' \
>   >> ~/.zshrc
> source ~/.zshrc
> ```

It prints your public address:

```
https://that-macbook.tail1234.ts.net
```

**That address is yours and it does not change.** Write it down — it is what
you give clinicians.

**How to tell it worked:** open that address on your *phone, with wifi off*.
You should reach the site. (Nothing is running yet, so expect an error page —
what matters is that it is *your* machine answering, not a timeout.)

> Tailscale Funnel is free, needs no domain and no card. Two things to know:
> it only carries ports 443, 8443 and 10000 on the public side, which is
> handled for you, and its bandwidth is capped at a level that is generous
> for this and not configurable.

---

## Step 4 — Fill in the settings

Settings live in a file called `.env`, beside the code. Start from the
example and edit it:

```bash
cd ~/Documents/GitHub/Rad-Intel
cp .env.example .env
nano .env
```

`nano` saves with **Ctrl+O**, Enter, then **Ctrl+X**.

**Where your data lives.** Leave `RAD_INTEL_FEEDBACK_DIR` and `DATABASE_URL`
**empty**. On your own machine the defaults — a folder beside the code and
`rad_intel.db` next to it — are exactly right, and nothing wipes them.

**Security.** Generate a key:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

and put it in, along with the two below:

```
RAD_INTEL_SECRET_KEY=<the long string you just generated>
RAD_INTEL_COOKIE_SECURE=1
RAD_INTEL_TRUSTED_PROXY=
```

> **Write that key down somewhere safe.** If you lose it, every stored phone
> number becomes permanently unreadable.

> `RAD_INTEL_COOKIE_SECURE=1` is right even though this process speaks plain
> HTTP: the address people actually use is HTTPS, and that is what the
> browser decides on.

> Leave `RAD_INTEL_TRUSTED_PROXY` **empty for now**. It decides whether the
> `X-Forwarded-For` header is believed. Believe it when nothing sets it and
> anybody can claim any address, so the rate limits stop meaning anything;
> ignore it when a tunnel *does* set it and every visitor looks like the same
> one person. Step 7 has a one-minute test that tells you which you have.

**Email** — the same values you already use. For Gmail, `SMTP_PASSWORD` is a
16-character App Password, not your account password:

```
RAD_INTEL_ADMIN_EMAIL=poorabng@gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=poorabng@gmail.com
SMTP_PASSWORD=<your 16-character Gmail App Password>
SMTP_FROM=poorabng@gmail.com
```

**Reports:**

```
GEMINI_API_KEY=<copy from your main machine's .env>
```

**Google Drive** — where reviewed radiographs go so they outlive this
machine. See Step 4b below for how to get these three.

```
GOOGLE_DRIVE_CLIENT_ID=
GOOGLE_DRIVE_CLIENT_SECRET=
GOOGLE_DRIVE_REFRESH_TOKEN=
```

**Leave these empty.** They are for development only, and on a machine the
public can reach they are actively harmful:

```
RAD_INTEL_ENABLE_DOCS=
RAD_INTEL_CORS_ORIGINS=
RAD_INTEL_OTP_CHANNEL=
RAD_INTEL_OTP_ECHO=
RAD_INTEL_MODEL_CACHE=
```

> `.env` is in `.gitignore`, so none of this is ever pushed to GitHub. Keep
> it that way.

---

## Step 4b — Put the radiographs somewhere permanent

**Skip this and the site still works. Do not skip it.** Reviewed radiographs
are kept on this MacBook's disk. The disk is permanent — but the MacBook is
not. It can be dropped, stolen, or simply stop one morning, and a radiograph
a clinician uploaded is the one thing here that **cannot be made again**. The
model can be retrained. The database can be rebuilt. A lost radiograph is
lost.

Self-hosting makes this step *more* important than it would be on a rented
server, not less: there is no datacentre keeping a copy for you.

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
5. On the MacBook, in the project folder:

```bash
.venv/bin/python scripts/connect_drive.py --client-id YOUR_ID --client-secret YOUR_SECRET
```

A browser opens on that machine. Sign in as the account whose Drive should
hold the radiographs, allow the one permission, and it prints three lines.
Paste them into `.env`, as in Step 4, and restart:

```bash
launchctl kickstart -k gui/$(id -u)/com.radintel.server
```

**How to tell it worked:** the console gets a card, *Where reviewed radiographs
are kept*, saying they are copied to your Drive, with a link to the folder.

**If you connect it after people have already reviewed cases:** that card
offers **Copy what is missing**. Press it once.

**To disconnect:** myaccount.google.com/permissions. The files stay; they are
yours.

---

## Step 5 — Make it start by itself, and stay up

Two things have to be true: it starts again after a reboot or a power cut,
and the MacBook does not fall asleep underneath it.

Both are handled by a small service. Install it:

```bash
cd ~/Documents/GitHub/Rad-Intel
sed "s|YOUR_USERNAME|$USER|g" deploy/com.radintel.server.plist \
  > ~/Library/LaunchAgents/com.radintel.server.plist
launchctl load -w ~/Library/LaunchAgents/com.radintel.server.plist
```

It runs `deploy/run-rad-intel.sh`, which reads your `.env`, starts the server
on port 8000, and wraps it in `caffeinate` so macOS will not sleep while it
is running. If it ever stops, launchd starts it again.

After editing `.env`, restart it with:

```bash
launchctl kickstart -k gui/$(id -u)/com.radintel.server
```

**One worker, deliberately.** The rate limits and the one-training-run-at-a-
time lock are counted inside a single process. Two workers would be two
separate sets of both, and every limit would be twice as generous as it
reads.

Now set the machine itself:

- **System Settings → Lock Screen** → *Turn display off on power adapter when
  inactive*: set it to a short time. The screen may sleep; the machine must
  not.
- **System Settings → Energy** → turn **on** *Start up automatically after a
  power failure*, and **off** any "put hard disks to sleep".
- Leave it **plugged in**, and leave the **lid open**. A closed lid sleeps a
  MacBook unless a display is attached, and a sleeping MacBook serves nothing.

**How to tell it worked**

```bash
curl -s localhost:8000/api/v1/health
tail -f ~/Library/Logs/rad-intel.log
```

The first start takes a minute or two while the model loads. That is normal,
not a hang.

To stop it:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.radintel.server.plist
```

---

## Step 6 — Make yourself the administrator on the server

Your admin account exists in your main machine's database, not this one's.
On the MacBook:

```bash
.venv/bin/python scripts/manage_clinicians.py add \
  --email poorabng@gmail.com \
  --name "Poorab N G" \
  --institution "Malnad College of Engineering" \
  --phone "+91 9844407409" \
  --role admin
```

It prints a password and a Rad-Intel ID. **Copy both** — you sign in with
the password through the **Administrator** door, which the next step checks.

If anything goes wrong and you cannot log in, this always works on the
MacBook itself:

```bash
.venv/bin/python scripts/manage_clinicians.py code --email poorabng@gmail.com --admin
```

It prints a code. On the login page press **"I already have a code"**, paste
it in. No email needed.

---

## Step 7 — Check it actually works

**First, one command.** On the MacBook:

```bash
.venv/bin/python scripts/preflight.py
```

It goes through everything that can quietly be wrong at the same time — the
key, the cookie, the proxy, the model file, the mail settings, where the
review store lives, whether an administrator exists, whether Drive is
connected — and prints each as **OK**, **FIX** or **CHECK**, with what to do
about it. Nothing says FIX, you are ready. Add `--email` and it sends you a
real test message too.

> Run this *after* Step 6, not before — one of the things it checks is
> whether an administrator account exists on this server, and until you make
> one it will rightly say FIX.

Then open your URL and go through these in order:

**1. Is it alive?** Visit `your-url/api/v1/health`. You want something like
this — the first field is the one that matters:

```json
{"status": "ok", "active_device": "cpu", "default_model": "hybrid",
 "gemini_reporting_live": true}
```

**2. Does the model work?** On the site: **Analyse X-ray → Load Pneumonia
CXR → Analyze Radiograph**. You should get a verdict, a heat map and a
written report.

> If you get *"Model has no trained checkpoint"* — the file from Step 2 is
> missing or misnamed. Check `ls -lh ~/Documents/GitHub/Rad-Intel/weights/`.

**3. Can you log in?** **Clinician Feedback → Administrator →** your email and
password → a code arrives in your Gmail → paste it in.

> If no code arrives, your SMTP settings are wrong. Check your spam folder
> first.

**4. Does mail work from the server?** In the console: **Send myself a test
email**. It should say it was sent.

**5. Is it private?** Visit `your-url/docs` — you should get **404**. If you
see a page of API documentation, `RAD_INTEL_ENABLE_DOCS` is set; empty it.

**6. Does it see who is calling?** This is the `RAD_INTEL_TRUSTED_PROXY`
question from Step 4, and one minute settles it.

On your **phone, with wifi off**, open the site and sign in. Then on the
MacBook, in the console: **Administrator → Activity**. Find that sign-in and
look at the address beside it.

- It shows **your phone's address** (something public, not starting `127.`) —
  the tunnel is passing the caller through. Set `RAD_INTEL_TRUSTED_PROXY=1`
  in `.env` and restart.
- It shows **`127.0.0.1`** — the tunnel is not passing it through. Leave
  `RAD_INTEL_TRUSTED_PROXY` empty.

Getting this wrong is quiet rather than loud, which is why it is worth the
minute: set it when nothing sets the header and anyone can claim any address,
so the rate limits stop meaning anything. Leave it unset when the header *is*
set, and every visitor counts as the same person, so one of them can lock out
all the others.

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
for everyone, with its scores and a note saying what it was built from —
*"retrained on the public dataset plus reviewed cases"* for a community
model, *"fine-tuned on reviewed cases"* for the other kind.

Two things that do not happen:

- **The default does not change.** Visitors still get the complete classifier
  unless they deliberately pick the new one.
- **Nobody's radiographs are published.** Only a model whose weights were
  shaped by them.

**Withdraw from public** takes it back off the list. The file stays on disk.

Separately, in **The shared model**, *Serve this version* changes which
checkpoint everybody gets **by default**, without them choosing anything.
That asks for your password too — it reaches further than a release does, so
it is not a single click on a page left open. It is a pointer, never a copy
over the original, so going back to the base checkpoint cannot fail.

One thing the release panel deliberately cannot do: make a community model the
*default*. It reaches people one way only — released by name, chosen
deliberately on the Analyse tab. The model a visitor gets without choosing
anything is always the one your paper describes.

---

# Things to know, not things to do

- **If Gmail breaks, nobody can log in.** The `manage_clinicians.py code`
  command at the end of Step 6 is your way back in. Keep it somewhere you can
  find it.
- **The mailbox is the second factor.** Someone with a clinician's Rad-Intel
  ID *and* access to their email is in. That is the design; it is reasonable
  for a study.
- **A session ends when the browser closes**, or after 7 days. One device at
  a time — logging in somewhere else logs out the old one.
- **There is no idle timeout.** An unlocked laptop is an open account until
  the browser closes.
- **The MacBook is on your home network.** Tailscale exposes one port and
  nothing else — no router change, no open ports — so the rest of the machine
  stays unreachable. Keep macOS's firewall on anyway, and do not use that
  machine for anything else.
- **Run preflight after any settings change.** `.venv/bin/python scripts/preflight.py`.
  It is the quickest way to find out that something you changed last week
  broke sign-in for everybody.
- **Back up the project folder.** `rad_intel.db` holds every account and
  every review; `.radintel_key` is what makes the stored phone numbers
  readable. With Step 4b done the radiographs themselves are already safe in
  Drive, so this is a small copy. Once a month onto a USB stick or your main
  machine is enough at this scale:

  ```bash
  cd ~/Documents/GitHub/Rad-Intel
  zip -r ~/Desktop/rad-intel-backup-$(date +%Y%m%d).zip \
    rad_intel.db .radintel_key .env feedback_store
  ```

- **The site is up only while that MacBook is.** Power cut, closed lid,
  someone unplugging it, your internet dropping — any of those and the
  address stops answering until it is back. Say so to the clinicians rather
  than letting them discover it.
- **Check on it now and then.** `tail ~/Library/Logs/rad-intel.log` and the
  Activity log in the console will both tell you whether anything has been
  happening.
