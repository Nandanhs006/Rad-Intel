# Moving off Azure, onto the 2013 Mac

Read this at **month four**, not month six. The Azure credit lasts roughly
five months, and the one thing you do not want is to be doing this on the day
the subscription is disabled.

Nothing here is urgent today. It exists so that the day you need it, it is an
afternoon and not a crisis.

---

## The one thing you must know first

**That Mac has to run Ubuntu, not macOS.**

This is not a preference. PyTorch stopped publishing macOS Intel builds after
version 2.2:

| torch | macOS Intel | macOS Apple Silicon | Linux Intel |
|---|---|---|---|
| 2.2.x | ✅ | ✅ | ✅ |
| 2.3 → 2.14 | **none** | ✅ | ✅ |

Rad-Intel uses **2.14.1**, and your trained checkpoint was produced with it.
On macOS, a mid-2013 Intel Mac simply cannot install it — `pip` will not find
a wheel, and there is nothing to be done about that.

The tempting workaround — pin torch 2.2.2 — is the wrong move here. It would
drag down `torchvision`, `timm` and the rest to match, and it would mean your
deployed model runs on a different numerical stack from the one your paper
reports. That is exactly the "second inconsistent inference path" the project
has avoided from the start.

**Ubuntu on that Mac gets you torch 2.14.1 — the identical build the Azure
server runs.** Same wheel family (`manylinux_2_28_x86_64`), same version, same
numbers. And because Azure is already Ubuntu, almost nothing else changes.

---

## What changes, and what does not

| | Azure | The 2013 Mac |
|---|---|---|
| Operating system | Ubuntu 24.04 | **Ubuntu 24.04** — same |
| PyTorch | 2.14.1+cpu | **identical** |
| `deploy/radintel.service` | same file | **same file** |
| `deploy/run-rad-intel.sh` | same file | **same file** |
| `.env` | same | same, one line differs |
| Public address | Caddy + Let's Encrypt | **Tailscale Funnel** |
| Who keeps it alive | Azure | you |

Only the last two rows are real work. Everything else is a copy.

**Why Tailscale instead of Caddy at home.** Caddy gets a certificate by
proving it controls a public address. A home connection in India usually has
no public address of its own, and no way to open a port to one. Tailscale
Funnel sidesteps both: it carries traffic in without any port being opened,
and brings its own HTTPS. Free, no card, no domain.

---

## Before the day comes: a dress rehearsal

Do this once, early, while Azure is still running. It takes an evening and it
is the difference between a smooth switch and a bad night.

1. Install Ubuntu on the Mac (below).
2. Follow steps 2–4 of the switch.
3. Point Tailscale at it and check the site loads **from your phone**.
4. Then stop it and leave it. Azure keeps serving.

When the day comes you are repeating something you have already done, with a
fresh copy of the data.

---

## Installing Ubuntu on a mid-2013 Mac

You need a USB stick of 8 GB or more. **Everything on the Mac is erased** —
if anything on it matters, copy it off first.

1. On your working machine, download **Ubuntu Server 24.04 LTS** (not
   Desktop — this is a server and the desktop just eats the 4 GB of memory).
2. Write it to the USB stick with [balenaEtcher](https://etcher.balena.io).
3. Put the stick in the Mac. Hold **Option (⌥)** while powering on.
4. Pick the orange **EFI Boot** icon.
5. Follow the installer. When it asks:
   - **Erase disk and install** — yes, the whole disk
   - **Your name / server name / username** — remember the username
   - **Install OpenSSH server** — **tick this**, or you cannot reach it
     remotely afterwards
6. Remove the stick and let it boot.

> **Wifi on these Macs is a known nuisance.** The Broadcom chip often needs a
> driver that is not on the install image. Use a wired ethernet connection —
> directly or through a USB-ethernet adapter — and the problem disappears. A
> machine serving a site should be on a cable anyway.

Find its address on your network:

```bash
ip addr show | grep "inet "
```

From then on you can work from your own machine:

```bash
ssh yourusername@192.168.1.xx
```

---

## The switch itself

### 1. Take a fresh backup from Azure

On the Azure server:

```bash
cd ~/Rad-Intel
sudo systemctl stop radintel
tar czf ~/radintel-final.tar.gz rad_intel.db .radintel_key .env
```

Stopping first matters: it means nothing is mid-write when the database is
copied.

From your own machine:

```bash
scp radintel:~/radintel-final.tar.gz .
```

(`radintel` is the SSH shortcut from DEPLOY.md Step 2.)

### 2. Set the Mac up

Exactly as **DEPLOY.md Step 3**, on the Mac this time:

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install python3-venv python3-pip git unzip
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

cd ~
git clone https://github.com/Nandanhs006/Rad-Intel.git
cd Rad-Intel
python3 -m venv .venv
.venv/bin/pip install -r requirements-deploy.txt
```

On a 2013 processor that install takes **thirty to forty-five minutes**. Let
it run.

### 3. Put the data back

From **your own machine**, send the backup across (the Mac's address is the
one you found after installing Ubuntu):

```bash
scp radintel-final.tar.gz yourusername@192.168.1.xx:~/
```

Then **on the Mac**:

```bash
cd ~/Rad-Intel
tar xzf ~/radintel-final.tar.gz
ls -la rad_intel.db .radintel_key .env
```

All three must be there. `.radintel_key` especially — without the exact same
key, every stored phone number becomes unreadable and everyone is signed out.

**The model files — all of them you were serving.** From your own machine:

```bash
cd ~/Documents/GitHub/Rad-Intel
scp weights/*.pt yourusername@192.168.1.xx:~/Rad-Intel/weights/
```

494 MB, but over your own network rather than the internet, so it is quick.
Send only `best_hybrid_model.pt` if you were only offering the one — whatever
is not there is not offered.

**The radiographs come back by themselves.** You do not need to copy
`feedback_store/`. Every reviewed radiograph is in your Google Drive, and the
first time somebody opens a case the server fetches it back and writes it
down. That is what Step 4b was for.

### 4. Change the one line in `.env`

```bash
nano .env
```

```
RAD_INTEL_TRUSTED_PROXY=
```

**Empty, not 1.** On Azure, Caddy wrote `X-Forwarded-For` and could be
believed. Tailscale Funnel is a different thing in front, and until you have
checked it you must not believe that header — if nothing sets it, anyone can
claim any address and the rate limits stop meaning anything.

Step 6 below is the one-minute check that settles it.

### 5. Start it

```bash
cd ~/Rad-Intel
sed -e "s|__USER__|$USER|g" -e "s|__HOME__|$HOME/Rad-Intel|g" \
  deploy/radintel.service | sudo tee /etc/systemd/system/radintel.service
sudo systemctl daemon-reload
sudo systemctl enable --now radintel
```

Two minutes, then:

```bash
curl -s localhost:8000/api/v1/health
```

### 6. Give it a public address

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale funnel --bg 8000
```

It prints a permanent address like
`radintel-mac.tail1234.ts.net`. **That is the new address** — send it to the
clinicians.

Now settle the proxy question. Open the site **from your phone with wifi
off**, sign in, then in the console: **Administrator → Activity**, and read
the address beside that sign-in.

- **Your phone's own public address** → set `RAD_INTEL_TRUSTED_PROXY=1` in
  `.env` and `sudo systemctl restart radintel`.
- **`127.0.0.1`** → leave it empty. Correct as it stands.

### 6b. Does it actually have everything?

Before you tell anybody the new address:

```bash
cd ~/Rad-Intel
ls -lh weights/*.pt          # the models you expect, right sizes
ls -la rad_intel.db .env .radintel_key
.venv/bin/python scripts/preflight.py
```

Then sign in as yourself and check the **Analyse** tab's model list shows the
same models it did on Azure. If one is missing, its file did not arrive.

### 7. Check, and only then let Azure go

```bash
cd ~/Rad-Intel && .venv/bin/python scripts/preflight.py
```

Then walk DEPLOY.md Step 7 — analyse a radiograph, sign in, send yourself a
test email, confirm `/docs` is a 404.

**Only when all of that passes** do you delete the Azure resource group. Keep
it until then, even if the credit is nearly gone; a disabled subscription can
be revived within 90 days, a deleted one cannot.

---

## Keeping it alive at home

- **Wired ethernet**, not wifi. See above.
- **Lid closed is fine** on Ubuntu, unlike macOS — but tell it so explicitly:
  ```bash
  sudo sed -i 's/^#*HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
  sudo systemctl restart systemd-logind
  ```
- **Set it to come back after a power cut.** On most Intel Macs:
  ```bash
  sudo apt -y install pmutils
  ```
  and in the Mac's own firmware there is no setting for this — so if power
  cuts are common where you are, a small UPS is worth more than any amount of
  configuration.
- **A 2013 processor is slow.** Expect five to fifteen seconds for an
  analysis where Azure took two or three. It works; it is not brisk. Tell the
  clinicians so they do not think it has hung.
- **Back up the same three files monthly**, exactly as on Azure. The machine
  is now a thirteen-year-old laptop; treat it like one.

---

## If the Mac will not do it

You will know within the evening of the dress rehearsal. Then the options are
the ones from before, in order of sense:

1. **Ask the college** for a machine that stays on. Still free, still the
   only option that does not expire.
2. **A small VPS**, about ₹350 a month for 4 GB — a quarter of what the
   original guide assumed. The whole of DEPLOY.md from Step 3 onwards applies
   unchanged.
3. **Renew Azure for Students** if you are still enrolled, in the 30-day
   window before your anniversary. Another $100, another five months.
