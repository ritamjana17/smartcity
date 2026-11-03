# CityEye – AI-Powered Smart City Surveillance for Traffic and Littering Management

Camera-first web app for phones and laptops. Point the camera at a street and CityEye finds
**garbage on the street**, **people littering** and **traffic problems**. It combines:

1. **Live object boxes** on the device (TensorFlow.js COCO-SSD): vehicles, people, litter-like objects.
2. **A street cleanliness model trained on a Kaggle dataset** (MobileNetV2 + logistic regression),
   running in the browser: live *Clean 82%* / *Littered 91%* reading, a second opinion on every scan,
   and an offline fallback when the cloud is unreachable.
3. **Google Gemini** (through a secure Vercel function) for the full scene analysis: garbage, littering
   and traffic flags, cleanliness score, counts, labelled boxes, summary and suggested action.

## Dataset

**Clean/Dirty Road Classification** – Kaggle:
https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification
(237 road photos labelled *clean* or *dirty*). Check the licence on the Kaggle page and keep the credit.

The training notebook copies the de-duplicated images into `dataset/` with the exact split
(`dataset/split.csv`, 70 / 15 / 15, stratified, seed 42) so results can be reproduced.

## Project structure

```
index.html                 the app (camera, live detection, results, history)
model.html                 "Dataset and model" page: data, metrics, charts, in-browser test
cleanliness.js             runs the trained model in the browser
api/analyze.js             Vercel serverless function that calls Gemini (API key stays secret)
training/
  cityeye_training.ipynb   Kaggle notebook: trains + evaluates the model, exports everything below
  cityeye_training.py      same notebook as a plain Python script
dataset/                   ← added from the notebook output: images + split.csv + README
model/                     ← added from the notebook output: cleanliness_model.json
results/                   ← added from the notebook output: metrics.json + charts
vercel.json, manifest.json, icon.svg
```

The app works without `dataset/`, `model/` and `results/`; the cleanliness reading and the
Dataset page simply appear once those folders are added.

## Step by step

### 1. Train the model on Kaggle (about 10 minutes, free)
1. Sign in at kaggle.com → **Create → New Notebook**.
2. **File → Import Notebook** → upload `training/cityeye_training.ipynb`.
3. Right-hand panel → **Add Input** → search `cleandirty-road-classification` → add the dataset by *faizalkarim*.
4. **Add Input** → **Models** → search `mobilenet-v2` (by Google) → framework **TensorFlow 2**,
   variation **100-224-feature-vector** → add.
5. *Optional (compare with Gemini):* **Add-ons → Secrets** → add `GEMINI_API_KEY`, and switch
   **Internet** on in the notebook settings (Kaggle asks for phone verification once).
6. **Run All**. At the end the notebook prints the test accuracy.
7. In the **Output** section (`/kaggle/working`) download **`cityeye_outputs.zip`**.

### 2. Add the results to the project
Unzip `cityeye_outputs.zip` **inside this folder**, so `dataset/`, `model/` and `results/`
sit next to `index.html`.

### 3. Push to GitHub
The dataset has more than 100 files, which is more than the GitHub website accepts in one upload,
so use Git (or GitHub Desktop):
```bash
git add .
git commit -m "Add Kaggle dataset, trained model and results"
git push
```

### 4. Vercel
Vercel redeploys automatically after the push. The environment variable `GEMINI_API_KEY` must be set
(Settings → Environment Variables). Then:
- Open the site on your phone → **Start camera** → the top-left pill shows *Clean* or *Littered*.
- Open **Settings → Dataset & model → Open** (or `/model.html`) → **Run on test images**.
  The "average difference from the notebook" should be close to 0 (below about 0.05).

## Why scans are fast
- Frames are shrunk to 768 px before upload; fast lite Gemini model with automatic fallback
- Low thinking level, strict JSON schema, short answers; server warm-up; Mumbai region (`bom1`)
- The trained model and object boxes run on the device with no network wait

## Notes
- The camera only works over HTTPS (Vercel gives you this) or on localhost.
- On the Gemini free tier, use an Auto watch interval of 10 s or more.
- History is stored only on the device (localStorage).
- Recording people in public may be regulated where you live; check local rules before real deployment.
