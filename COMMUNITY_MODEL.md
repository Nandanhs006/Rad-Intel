# Training the community model

The community model is a **second** model. It is not the one in your paper and
it never touches it.

| | The paper's model | The community model |
|---|---|---|
| File | `best_hybrid_model.pt` | `best_community_hybrid_v1_model.pt`, `v2`, `v3`… |
| Architecture | DenseNet121 + Swin-T + CBAM | **the same** |
| Trained on | the public dataset | the public dataset **+ every reviewed case** |
| Retrained | never | from scratch, every time |
| Who trains it | you, once, for the paper | you, whenever it is worth it |

Each run throws the previous community model away and builds a new one from
nothing. So a version is always made from *all* the evidence, not from the
leftovers of the last run.

Three steps. Steps 1 and 3 are clicks. Step 2 is this guide.

---

## Before you start

You need:

- an **administrator** login on the site (the Administrator door, not the
  clinician one),
- a **Google account**, for the free GPU,
- about **90 minutes**, most of it waiting.

You do **not** need the dataset, a paid service, or anything installed.

---

## Step 1 — Download what has been collected

On the site: **Clinician Feedback → Administrator console → The community
model → Build training bundle**.

It makes one zip holding every reviewed radiograph and a list of their labels.
It also freezes anything still pending, the same as pressing *Create training
export* — so the bundle keeps matching the records it was built from.

Press **Download**. You now have something like
`community-bundle-20261015T143000Z.zip`.

> This zip holds radiographs people uploaded to your study. Keep it somewhere
> private, and delete it from wherever you trained when you are done.

**If the button refuses:** it needs at least four reviewed cases, with both
NORMAL and PNEUMONIA among them. Training on fewer, or on one class only,
produces a model that is worse than useless — it is confidently wrong.

---

## Step 2 — Train it on a free GPU

### Why not on the server

A full retrain is a few hours of real computation. Your web host has a small
processor and is busy serving the site. Renting a GPU machine to sit idle
between runs would cost more than everything else put together.

So: train on a GPU you borrow for an hour, and bring the result back.

### Open a notebook

Go to **[colab.research.google.com](https://colab.research.google.com)** →
**New notebook**.

Then **Runtime → Change runtime type → T4 GPU → Save**. This is the free one.
If it says no GPU is available, try again in a few hours — it is shared.

### Cell 1 — check you actually got a GPU

```
!nvidia-smi
```

You want a table with **Tesla T4** in it. If you get
`command not found`, you are on a processor, not a GPU; go back and change the
runtime type. Training will still work, but it will take most of a day instead
of half an hour.

### Cell 2 — get the code

```
!git clone https://github.com/<your-github-username>/Rad-Intel.git
%cd Rad-Intel
!pip install -q -e . 2>&1 | tail -2
```

Put your own GitHub address in. If the repository is private, Colab will ask
you to sign in.

This installs the project's own code, so the training recipe and the image
preprocessing are **exactly** the ones the site uses. That is the point: the
only thing different about the community model is the data it saw.

### Cell 3 — get the public dataset

```
!pip install -q kagglehub
import kagglehub, os
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
DATA = os.path.join(path, "chest_xray")
print(DATA)
```

The first time, Kaggle asks for a token: **kaggle.com → your profile →
Settings → API → Create New Token**, then upload the `kaggle.json` file it
gives you when Colab asks.

It downloads 2.3 GB. Takes a few minutes.

### Cell 4 — upload your bundle

```
from google.colab import files
up = files.upload()
BUNDLE = list(up.keys())[0]
print(BUNDLE)
```

A **Choose files** button appears. Pick the zip from Step 1.

### Cell 5 — train

```
!python scripts/train_community_model.py \
    --bundle "$BUNDLE" \
    --data-dir "$DATA"
```

Now wait. On a T4 this is roughly **30–60 minutes**. You will see:

1. how many reviewed cases it found, and how they split normal/pneumonia,
2. the dataset split — around 4,400 train, 780 validation, **624 test**,
3. feature extraction, then two phases of training with a loss going down,
4. a results table, then `Saved …best_community_hybrid_v1_model.pt`.

**Do not close the tab.** Colab disconnects an idle notebook and you lose the
run. Leave it in front of you.

### Cell 6 — get the result out

**The easy way — save it to your Google Drive.** Google to Google, so it is
seconds rather than minutes, and you never move 136 MB through your own
connection at all.

```
from google.colab import drive
import glob, shutil, os
drive.mount("/content/drive")
out = sorted(glob.glob("weights/best_community_hybrid_v*_model.pt"))[-1]
dest = "/content/drive/MyDrive/" + os.path.basename(out)
shutil.copy(out, dest)
print("saved to", dest)
```

Then in Drive: right-click the file → **Share** → **Anyone with the link** →
**Copy link**. That link is what you paste in Step 3.

> **Anyone with the link** matters. On **Restricted** the server gets a
> sharing page instead of the file, and says so.

**Or download it to your laptop** if you would rather:

```
from google.colab import files
import glob
files.download(sorted(glob.glob("weights/best_community_hybrid_v*_model.pt"))[-1])
```

---

## Step 3 — Bring it back to the site

On the site: **Administrator console → The community model → Upload the
result**. Two ways, pick one:

- **Paste the share link** from Cell 6. The server fetches the file itself,
  at the speed two datacentres talk to each other. Nothing goes up through
  your connection. This is the one to use.
- **Choose the file** from your laptop, if you downloaded it. Fine on fibre,
  slow on a home connection, and some hosts refuse a body that large.

Either way, type your administrator password and press **Bring this version
in**.

It is checked before it is kept. It is refused if the weights do not fit the
architecture, if it carries no operating threshold, or if it reports no
scores — a model nobody can judge is worse than no model, because it returns
confident numbers either way.

**Uploading releases nothing.** The version appears in **Release a model to
the public** with its scores next to the other candidates. Read them. Then
release it only if it earns it — and that asks for your password again.

---

## Reading the scores

The community model is measured on the **same 624 held-out radiographs** as
every model in your paper, which were never trained on by either. So you can
compare the rows directly.

Your reported figures, for reference — the same two the site shows in the
model chooser:

| | accuracy | AUROC |
|---|---|---|
| hybrid (the default) | 0.920 | 0.974 |
| hybrid without CBAM | 0.934 | 0.977 |
| DenseNet121 | 0.915 | 0.962 |
| Swin-T | 0.901 | 0.975 |
| ResNet50 | 0.897 | 0.954 |

The community version reports accuracy, AUROC, precision, sensitivity,
specificity and F1, all on those same 624 images.

If the community version is **worse**, do not release it. That is a real
answer, not a failure — it means the contributed cases are not yet enough to
improve on 5,216 training images, which is exactly what you should expect
early on.

### What to honestly expect

With around **100 contributed radiographs** against the dataset's 5,216, the
numbers **will not move measurably**. A hundred images is about 2% of the
training data.

The value is not volume. It is that those hundred come from *different
hospitals and different machines* than the public dataset — which is the
weakness your paper names. But that only shows up in the numbers at something
like **500+ cases from several sites**.

So: collect for a while before expecting anything. Running this on 20 cases
tells you nothing except that it ran.

---

## Questions you will have

**Does the site go down while this trains?**
No. This trains on Colab, nowhere near your server. Even the small
fine-tuning run that *does* happen on the server does not take the site down:
it runs as a separate process while the web server keeps answering from the
model it already has in memory. The worst that happens is slightly slower
predictions.

**Can the clinicians do this?**
No, and deliberately. They upload cases, correct cases, and press *Create
training export* when a batch is worth keeping. Training and releasing are
yours.

**What happens to the old community version when I make a new one?**
Nothing — the file stays. Releasing a new one does not withdraw the old one;
do that yourself if you want only the newest on offer.

**A clinician deleted their case after I trained. Now what?**
The model you already made is unchanged — it is a record of what was actually
trained on. The deleted case stays out of every *future* bundle, so the next
community version will not contain it.

**Do I have to use Colab?**
No. Anything with a GPU and Python works — Kaggle Notebooks, a university
machine, a rented GPU for an hour. The command in Cell 5 is the whole recipe.

**Which links will the server fetch from?**
Google Drive, Dropbox, a GitHub release, and Hugging Face. Not any address
you like — a server that fetches whatever it is handed is a way to reach
things only it can reach, so it only fetches from places people actually put
a file to pass it on.

**There used to be a 'Start fine-tuning run' button. Where did it go?**
Gone, deliberately. It did a different thing — nudged the project's own
checkpoint with the latest batch — and having two training buttons meant
explaining which was which every time. One story now: bundle, train, bring it
back, release. The old script is still there as
`scripts/finetune_from_feedback.py` if you ever want it from a terminal.

**Can I run it with no GPU at all?**
Yes, and it will take the better part of a day. `--epochs 20
--finetune-epochs 10` are the defaults; lowering them makes it faster and the
model worse.
