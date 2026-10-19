# Putting Rad-Intel online

You are deploying to a small Azure server, paid for with the **Azure for
Students** credit. No credit card, no bill, a real HTTPS address, and it runs
on its own.

Follow this top to bottom. Every step says what to do and how to tell it
worked. Nothing here needs you to understand the code.

Budget about **two and a half hours** the first time.

---

## What the credit buys, honestly

**$100, which must be spent within 12 months.** When it runs out Azure
**disables the subscription** — it does not charge you, and there is no card
on file for it to charge. You cannot wake up to a bill.

The one thing people get wrong: **Azure bills for every hour the machine is
switched on, not per visit.** Three people using it costs exactly what three
hundred would. So the way to stretch the credit is not to use it less — it is
to switch the machine off when nobody needs it, which Step 6 sets up for you.

Roughly, per month:

| | running 24/7 | off 10 hours a night |
|---|---|---|
| The machine (B1ms, 2 GB) | ~$15 | ~$9 |
| Disk and address | ~$6 | ~$6 |
| **Total** | **~$21** | **~$15** |
| **$100 lasts** | **~5 months** | **~7 months** |

Bandwidth is free up to 100 GB a month, which this will not approach.

> **Why 2 GB and not the cheaper 1 GB.** Measured, this needs about 870 MB,
> so 1 GB leaves nothing for the operating system. The smaller machine is
> also much slower in a way that bites exactly here: these are *burstable*
> machines, and the 1 GB one gets about a tenth of a processor once its
> saved-up credit is gone. The 2 GB one banks credit while nobody is using
> it and spends it on the occasional analysis, which is precisely your
> pattern.

**Put a reminder in your phone for four months from today.** That is when you
decide what happens next: renew the student credit if you are still enrolled,
move to the 2013 Mac, or pay roughly ₹350 a month elsewhere.

**[MIGRATE.md](MIGRATE.md)** is the move to the Mac, written out and ready.
It has one thing in it you should read before then rather than on the day:
that Mac has to run Ubuntu, because PyTorch stopped building for Intel macOS
three years ago. Read it at month four and do the dress rehearsal it
describes; the switch is then an afternoon rather than a bad night.

---

## Step 1 — Make the server

1. Go to **portal.azure.com** and sign in with your college email.
2. Search for **Virtual machines** → **Create** → **Azure virtual machine**.
3. Fill in only these; leave everything else alone:

| Field | What to put |
|---|---|
| Resource group | **Create new** → `rad-intel` |
| Virtual machine name | `radintel` |
| Region | **any India region** — Central India, South India, whichever it offers |
| Image | **Ubuntu Server 24.04 LTS - x64 Gen2** |
| Size | **See below** |
| Authentication type | **SSH public key** |
| Username | `azureuser` |
| SSH public key source | **Generate new key pair** |
| Key pair name | `radintel_key` |
| Public inbound ports | **Allow selected ports** |
| Select inbound ports | **SSH (22)**, **HTTP (80)**, **HTTPS (443)** |

> **Whichever region you pick is fine** — but remember which, because it
> becomes part of your web address. Do not assume it is Central India just
> because this guide says so.

For **Size**, click *See all sizes*, search `B1ms`, and pick
**B1ms — 1 vcpu, 2 GiB memory**. It is not in the short list by default.

4. Go to the **Disks** tab. Change **OS disk type** to **Standard SSD**. The
   default is Premium and costs about three times as much for no benefit
   here.

5. **Review + create** → **Create**. A box appears offering a private key
   file — **Download private key and create resource.**

   On a Mac it lands in **`~/Downloads/radintel_key.pem`**. It is the only
   way into the machine and it is **never offered again**. If you lose it you
   delete the VM and start this step over.

**How to tell it worked:** the VM page shows *Running* after a minute or two.

---

## Step 2 — Give it a name, and set up one-word access

### The address

On the VM's **Overview** page, find **DNS name** → click **Configure** →
under *DNS name label* type `radintel-mce` → **Save**.

Go back to **Overview**. There is now a full address next to *DNS name*.
**Copy it from the page — do not type it from memory.** It looks like:

```
radintel-mce.indiasouthcentral.cloudapp.azure.com
```

but the middle part is whatever region you actually chose, and getting it
wrong is the commonest way to lose twenty minutes here.

Check it resolves, on **your own machine**:

```bash
ping -c 2 PASTE-YOUR-ADDRESS-HERE
```

Replies mean the name is right. *"cannot resolve"* means it is not — go back
and copy it again.

### One-word access

Rather than retyping a key path and a long hostname every time, tell SSH
about the machine once. On **your own machine**:

```bash
chmod 600 ~/Downloads/radintel_key.pem
```

```bash
cat >> ~/.ssh/config <<'END'

Host radintel
  HostName PASTE-YOUR-ADDRESS-HERE
  User azureuser
  IdentityFile ~/Downloads/radintel_key.pem
  ServerAliveInterval 60
END
```

Replace `PASTE-YOUR-ADDRESS-HERE` with the address you copied —
`nano ~/.ssh/config` if you need to edit it afterwards.

Now:

```bash
ssh radintel
```

Say `yes` at the fingerprint prompt. You are on the server.

> **If it says `Permission denied (publickey)`** the key file is not where
> SSH is looking. Check it is really there:
> ```bash
> ls -l ~/Downloads/radintel_key.pem
> ```
> *No such file or directory* means it downloaded somewhere else — find it
> with `ls ~/Downloads/*.pem` and put that path in `~/.ssh/config`.

> **`ServerAliveInterval 60`** stops the connection dropping while a long
> install runs. Worth having.

> Every command from here until Step 7 runs **on the server**, inside that
> SSH session. Disconnected? Just `ssh radintel` again.

---

## Step 3 — Set the server up

On the server. Paste these one block at a time.

**The basics, and some swap.** 2 GB is enough but not generous, and swap
means a brief spike slows things down instead of killing the process:

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install python3-venv python3-pip git unzip
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

`free -h` should show 2 GB of swap.

**The project:**

```bash
cd ~
git clone https://github.com/Nandanhs006/Rad-Intel.git
cd Rad-Intel
python3 -m venv .venv
.venv/bin/pip install -r requirements-deploy.txt
```

That takes fifteen to twenty-five minutes and downloads about a gigabyte. It
installs the **CPU** build of PyTorch, which is the right one here and several
gigabytes smaller than the default.

**How to tell it worked:**

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
```

---

## Step 3b — Send the model files

**Send all five.** The model files are not in git — GitHub refuses files that
big — and **the Analyse tab only offers models whose file is actually on the
server.** Send one and your site offers one model; send five and it offers
the five your About page describes.

| File | Size | What it is |
|---|---|---|
| `best_hybrid_model.pt` | 136 MB | **the default** — required |
| `best_hybrid_no_cbam_model.pt` | 136 MB | the ablation |
| `best_swin_t_model.pt` | 105 MB | Swin-T baseline |
| `best_resnet50_model.pt` | 90 MB | ResNet50 baseline |
| `best_densenet121_model.pt` | 27 MB | DenseNet121 baseline |

**494 MB in total** — though `best_densenet121_model.pt` is small enough to
live in the repository, so it arrives with the clone and only about 467 MB
actually travels. Sending all five anyway is simplest; the one already there
is just overwritten with itself.

On **your own machine**, in a second terminal window:

```bash
cd ~/Documents/GitHub/Rad-Intel
scp weights/*.pt radintel:~/Rad-Intel/weights/
```

That is the whole command — the `radintel` shortcut from Step 2 carries the
key and the address. Expect **twenty minutes to an hour** depending on your
upload speed; it shows a progress bar per file.

> **In a hurry, or on a slow connection?** Send the default alone first and
> the site works:
> ```bash
> scp weights/best_hybrid_model.pt radintel:~/Rad-Intel/weights/
> ```
> Send the rest whenever. They appear in the chooser after a restart:
> `sudo systemctl restart radintel`.

**How to tell it worked**, back on the server:

```bash
ls -lh ~/Rad-Intel/weights/
```

Five files, sizes matching the table above.

> **Having all five costs disk, not memory.** A model is loaded only when
> somebody picks it, and the previous choice is dropped to make room — the
> server sits at about 870 MB however many files are there.

---

## Step 4 — Fill in the settings

On the server:

```bash
cd ~/Rad-Intel
cp .env.example .env
nano .env
```

`nano` saves with **Ctrl+O**, Enter, then **Ctrl+X**.

**Where your data lives.** Leave `RAD_INTEL_FEEDBACK_DIR` and `DATABASE_URL`
**empty**. The defaults — a folder beside the code, and `rad_intel.db` next
to it — are right here, and nothing wipes them.

**Security.** Generate a key:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

```
RAD_INTEL_SECRET_KEY=<the long string you just generated>
RAD_INTEL_COOKIE_SECURE=1
RAD_INTEL_TRUSTED_PROXY=1
```

> **Write that key down somewhere off the server.** Lose it and every stored
> phone number becomes permanently unreadable.

> `RAD_INTEL_TRUSTED_PROXY=1` is correct here and I can say so without you
> having to test it. Step 5 puts Caddy in front, Caddy *writes* the
> `X-Forwarded-For` header itself and overwrites anything a visitor claimed,
> and Rad-Intel itself only listens on the loopback — so that header is the
> truth and nothing else can reach the app to lie about it.

**Email** — the values you already use. For Gmail, `SMTP_PASSWORD` is a
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
GEMINI_API_KEY=<copy from your own machine's .env>
```

**Google Drive** — Step 4b fills these in.

```
GOOGLE_DRIVE_CLIENT_ID=
GOOGLE_DRIVE_CLIENT_SECRET=
GOOGLE_DRIVE_REFRESH_TOKEN=
```

**Leave these empty.** They are for development, and on a machine the public
can reach they are actively harmful:

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
are kept on this server's disk. That disk lives as long as the subscription
does — and the subscription has about five months of credit in it. When it
is disabled, the machine and its disk go with it, and a radiograph a
clinician uploaded is the one thing here that **cannot be made again**. The
model can be retrained. The database can be rebuilt. A lost radiograph is
lost.

A credit with a deadline on it makes this step *more* important, not less.
Do it now, not in month four.

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
5. This one runs on **your own machine**, not the server — it opens a
   browser window, and the server has no browser. Clone the project locally
   if you have not already, then:

```bash
.venv/bin/python scripts/connect_drive.py --client-id YOUR_ID --client-secret YOUR_SECRET
```

A browser opens on that machine. Sign in as the account whose Drive should
hold the radiographs, allow the one permission, and it prints three lines.
It prints three lines. Put them into the **server's** `.env`, as in Step 4,
then restart it:

```bash
sudo systemctl restart radintel
```

**How to tell it worked:** the console gets a card, *Where reviewed radiographs
are kept*, saying they are copied to your Drive, with a link to the folder.

**If you connect it after people have already reviewed cases:** that card
offers **Copy what is missing**. Press it once.

**To disconnect:** myaccount.google.com/permissions. The files stay; they are
yours.

---

## Step 5 — Make it run on its own, with a real certificate

Two pieces. **systemd** keeps Rad-Intel running and starts it again if it
stops or the machine reboots. **Caddy** sits in front, holds the HTTPS
certificate, and is the only thing the outside world can reach.

**Rad-Intel as a service:**

```bash
cd ~/Rad-Intel
sed -e "s|__USER__|$USER|g" -e "s|__HOME__|$HOME/Rad-Intel|g" \
  deploy/radintel.service | sudo tee /etc/systemd/system/radintel.service
sudo systemctl daemon-reload
sudo systemctl enable --now radintel
```

> The `sed` fills in your username and folder. The same file therefore works
> unchanged on any other Linux machine you move to later — see
> **[MIGRATE.md](MIGRATE.md)**.

Give it two minutes — the first start loads the model — then:

```bash
curl -s localhost:8000/api/v1/health
```

If that is quiet, read the log: `journalctl -u radintel -n 50`.

**Caddy, and the certificate:**

```bash
sudo apt -y install debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt -y install caddy
```

Now put your own address in its config:

```bash
sudo cp ~/Rad-Intel/deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile
```

Change the **first line** to **your own address from Step 2** — the one you
copied from the portal — keeping the `{` at the end:

```
radintel-mce.centralindia.cloudapp.azure.com {
```

> It must match exactly, region and all. Caddy asks Let's Encrypt for a
> certificate for whatever name is on that line, and a name that is not
> really yours is refused.

Then:

```bash
sudo systemctl restart caddy
```

Caddy fetches a free Let's Encrypt certificate for that name, by itself,
within about thirty seconds — and renews it by itself for ever after. There
is nothing to buy and nothing to remember.

**How to tell it worked:** on your own machine, open

```
https://YOUR-ADDRESS-FROM-STEP-2
```

A padlock and the Rad-Intel page. If the browser complains about the
certificate, wait a minute and reload — it was still being issued.

> If Caddy cannot get a certificate, nine times out of ten port 80 is closed.
> In the portal: your VM → **Networking** → make sure inbound rules allow
> **80** and **443**.

---

## Step 5b — Switch it off at night (optional, and worth it)

This is the difference between the credit lasting about five months and about
seven. Skip it if the clinicians genuinely use it at odd hours.

In the portal: your VM → **Auto-shutdown** (left menu, under *Operations*) →
**On** → pick a time, say **1:00 AM**, set your time zone to **India
Standard Time** → **Save**.

Starting it again is manual: the VM's **Overview** page → **Start**. It is
serving again about two minutes later.

> Switching the machine off stops the compute charge. The disk and the
> address keep costing their ~$6 a month either way.

---

## Step 6 — Make yourself the administrator on the server

Your admin account exists in your own machine's database, not this one's.
On the server:

```bash
cd ~/Rad-Intel
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
server itself:

```bash
cd ~/Rad-Intel
.venv/bin/python scripts/manage_clinicians.py code --email poorabng@gmail.com --admin
```

It prints a code. On the login page press **"I already have a code"**, paste
it in. No email needed.

---

## Step 7 — Check it actually works

**First, one command.** On the server:

```bash
cd ~/Rad-Intel && .venv/bin/python scripts/preflight.py
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
> missing or misnamed. Check `ls -lh ~/Rad-Intel/weights/`.

**3. Can you log in?** **Clinician Feedback → Administrator →** your email and
password → a code arrives in your Gmail → paste it in.

> If no code arrives, your SMTP settings are wrong. Check your spam folder
> first.

**4. Does mail work from the server?** In the console: **Send myself a test
email**. It should say it was sent.

**5. Is it private?** Visit `your-url/docs` — you should get **404**. If you
see a page of API documentation, `RAD_INTEL_ENABLE_DOCS` is set; empty it.

**6. Does it see who is calling?** You set `RAD_INTEL_TRUSTED_PROXY=1` in
Step 4; this confirms it. Open the site **from your phone with wifi off**,
sign in, then in the console: **Administrator → Activity**. Find that sign-in
and look at the address beside it.

It should be **your phone's own address** — something public, not
`127.0.0.1`. That means Caddy is passing the real caller through, which is
what makes the rate limits count people rather than counting everybody as
one.

> If it does say `127.0.0.1`, Caddy is not setting the header. Check you
> copied `deploy/Caddyfile` rather than writing your own, and restart both:
> `sudo systemctl restart caddy radintel`.

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

- **The credit runs out in about five months.** Azure disables the
  subscription rather than charging you, but the machine and its disk go with
  it. Put the reminder in your phone now. A backup (below) is what makes that
  a move rather than a loss.
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
- **Only three ports are open, and only one thing answers.** Rad-Intel itself
  listens on the loopback, so the only way in is through Caddy on 443. Leave
  the Azure network rules as Step 1 set them, and do not add more.
- **Keep the server patched.** Once a month:
  ```bash
  sudo apt update && sudo apt -y upgrade && sudo reboot
  ```
  It comes back on its own — that is what `systemctl enable` was for.
- **Run preflight after any settings change.** `.venv/bin/python scripts/preflight.py`.
  It is the quickest way to find out that something you changed last week
  broke sign-in for everybody.
- **Back up, once a month.** `rad_intel.db` holds every account and every
  review; `.radintel_key` is what makes the stored phone numbers readable.
  With Step 4b done the radiographs are already safe in Drive, so this is a
  small file. On the **server**:

  ```bash
  cd ~/Rad-Intel
  tar czf ~/radintel-backup-$(date +%Y%m%d).tar.gz \
    rad_intel.db .radintel_key .env
  ```

  Then, from **your own machine**, pull it down:

  ```bash
  scp radintel:~/radintel-backup-*.tar.gz .
  ```

  That file plus the Drive folder is the whole study. With both, month five is
  an afternoon's work on a new machine.
- **Watch it when you want to.** `journalctl -u radintel -f` for the server,
  `sudo journalctl -u caddy -f` for the certificate and traffic, and the
  Activity log in the console for who has actually been using it.
- **Keep an eye on the credit.** Portal → **Subscriptions** → your
  subscription → **Overview** shows what is left. Check it monthly, not on
  the last day.
