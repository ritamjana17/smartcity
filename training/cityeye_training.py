# %% [markdown]
# # CityEye – Street Cleanliness Model (Kaggle notebook)
#
# This notebook trains the on-device **street cleanliness model** used by the CityEye app.
#
# * **Dataset:** [Clean/Dirty Road Classification](https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification) (Kaggle, 237 road images labelled *clean* or *dirty*).
# * **Method:** transfer learning. A pre-trained **MobileNetV2** turns every image into 1,280 numbers (features).
#   A **logistic regression** classifier is trained on those features to predict *clean* vs *dirty*.
# * **Output:** a small JSON model file that the CityEye web app runs live in the phone browser,
#   the dataset split, evaluation metrics and charts for the report.
#
# **Before you run:** in the right-hand panel click **Add Input** and add
# 1. the dataset `faizalkarim/cleandirty-road-classification`, and
# 2. the model `google/mobilenet-v2` → framework **TensorFlow 2** → variation **100-224-feature-vector**.
#
# Then click **Run All**. When it finishes, download `cityeye_outputs.zip` from the **Output** section.
#
# *Optional:* to also measure Gemini on the same test images, add your key under **Add-ons → Secrets**
# with the name `GEMINI_API_KEY`, and turn **Internet** on in the notebook settings.

# %%
import os, glob, json, shutil, random, time, re, hashlib, base64, io, zipfile, datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, classification_report)

SEED = 42
IMG = 224                      # MobileNetV2 input size
random.seed(SEED); np.random.seed(SEED)
INPUT_DIR = os.environ.get("CITYEYE_INPUT", "/kaggle/input")
WORK = os.environ.get("CITYEYE_WORK", "/kaggle/working")
OUT = os.path.join(WORK, "cityeye_outputs")
STUB = os.environ.get("CITYEYE_STUB") == "1"   # only used to test this notebook without TensorFlow
if not STUB:
    import tensorflow as tf
    print("TensorFlow", tf.__version__)

shutil.rmtree(OUT, ignore_errors=True)
for d in ["dataset/images/clean", "dataset/images/dirty", "model", "results"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
CLR = {"clean": "#0E6B45", "dirty": "#E8701A"}
print("Input folder:", INPUT_DIR)

# %% [markdown]
# ## 1. Find and label the images
# Labels come from the file name (`clean_12.jpg`, `dirty_7.jpg`), the folder name, or `metadata.csv`.
# Exact duplicate files are removed so the same picture cannot appear in both training and test sets.

# %%
def label_from(path, meta):
    name = os.path.basename(path).lower()
    if name.startswith("clean"): return "clean"
    if name.startswith("dirty") or name.startswith("litter"): return "dirty"
    parts = [p.lower() for p in path.split(os.sep)]
    if "clean" in parts: return "clean"
    if "dirty" in parts or "littered" in parts: return "dirty"
    v = meta.get(os.path.basename(path))
    if v is not None:
        s = str(v).strip().lower()
        if s in ("clean", "0"): return "clean"
        if s in ("dirty", "littered", "1"): return "dirty"
    return None

# metadata.csv (if present): map file name -> label
meta = {}
for m in glob.glob(os.path.join(INPUT_DIR, "**", "metadata.csv"), recursive=True):
    df = pd.read_csv(m)
    fcol = next((c for c in df.columns if df[c].astype(str).str.contains(r"\.(?:jpg|jpeg|png)$", case=False).any()), None)
    lcol = next((c for c in df.columns if c != fcol), None)
    if fcol and lcol:
        meta.update({os.path.basename(str(a)): b for a, b in zip(df[fcol], df[lcol])})

files = [p for p in glob.glob(os.path.join(INPUT_DIR, "**", "*"), recursive=True)
         if p.lower().endswith((".jpg", ".jpeg", ".png")) and "mobilenet" not in p.lower()]
rows, seen = [], set()
for p in sorted(files):
    lab = label_from(p, meta)
    if lab is None: continue
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in seen: continue
    seen.add(h)
    try:
        with Image.open(p) as im: w, hgt = im.size
    except Exception:
        continue
    rows.append({"src": p, "file": os.path.basename(p), "label": lab, "width": w, "height": hgt})

data = pd.DataFrame(rows)
assert len(data) > 20, "No labelled images found. Did you add the Kaggle dataset as an input?"
print(f"{len(data)} unique images")
print(data["label"].value_counts().to_string())

# %% [markdown]
# ## 2. Explore the dataset

# %%
counts = data["label"].value_counts().reindex(["clean", "dirty"])
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.bar(counts.index, counts.values, color=[CLR["clean"], CLR["dirty"]], width=0.55)
for i, v in enumerate(counts.values): ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
ax.set_ylabel("Images"); ax.set_title("Images per class")
fig.tight_layout(); fig.savefig(f"{OUT}/results/class_distribution.png"); plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for r, lab in enumerate(["clean", "dirty"]):
    for c, (_, row) in enumerate(data[data.label == lab].sample(5, random_state=SEED).iterrows()):
        ax = axes[r, c]; ax.imshow(ImageOps.exif_transpose(Image.open(row.src)).convert("RGB")); ax.axis("off")
        ax.set_title(lab, color=CLR[lab], fontweight="bold")
fig.suptitle("Sample images from the dataset"); fig.tight_layout()
fig.savefig(f"{OUT}/results/sample_images.png"); plt.close(fig)
print("Image width  (px): min %d, median %d, max %d" % (data.width.min(), data.width.median(), data.width.max()))
print("Image height (px): min %d, median %d, max %d" % (data.height.min(), data.height.median(), data.height.max()))

# %% [markdown]
# ## 3. Split into training, validation and test sets (70 / 15 / 15, stratified)

# %%
train_df, rest = train_test_split(data, test_size=0.30, stratify=data.label, random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
data["split"] = "train"
data.loc[val_df.index, "split"] = "val"
data.loc[test_df.index, "split"] = "test"
print(pd.crosstab(data.split, data.label, margins=True).to_string())

# Copy the images into the repository layout (resized to max 800 px so the GitHub repo stays small)
def save_copy(row):
    rel = f"images/{row.label}/{row.file.rsplit('.', 1)[0]}.jpg"
    im = ImageOps.exif_transpose(Image.open(row.src)).convert("RGB")
    im.thumbnail((800, 800))
    im.save(os.path.join(OUT, "dataset", rel), quality=88)
    return rel
data["path"] = data.apply(save_copy, axis=1)
data[["path", "label", "split", "width", "height"]].to_csv(f"{OUT}/dataset/split.csv", index=False)

# %% [markdown]
# ## 4. Feature extraction with MobileNetV2
# Each image is resized to 224 × 224, scaled to 0–1 and passed through MobileNetV2 (ImageNet weights).
# The output is a 1,280-number feature vector. The CityEye app uses **the same MobileNetV2**
# in the browser, so the features match.

# %%
def load_rgb(rel):
    im = np.asarray(Image.open(os.path.join(OUT, "dataset", rel)).convert("RGB"), dtype=np.float32) / 255.0
    if STUB:
        return np.asarray(Image.fromarray((im * 255).astype("uint8")).resize((IMG, IMG), Image.BILINEAR), dtype=np.float32) / 255.0
    # same resize as the browser (TensorFlow.js resizeBilinear with alignCorners = true)
    return tf.compat.v1.image.resize_bilinear(im[None], [IMG, IMG], align_corners=True)[0].numpy()

if STUB:
    # Test mode only: simple colour statistics instead of MobileNetV2
    def embed(batch):
        b = np.asarray(batch)
        return np.concatenate([b.mean((1, 2)), b.std((1, 2)), (b > 0.6).mean((1, 2))], axis=1)
    EXTRACTOR = "stub (test mode)"
else:
    found = [os.path.dirname(p) for p in glob.glob(os.path.join(INPUT_DIR, "**", "saved_model.pb"), recursive=True)
             if "feature" in p.lower() and "mobilenet" in p.lower()]
    HANDLE = found[0] if found else "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    print("Feature extractor:", HANDLE)
    if os.path.isdir(HANDLE):
        _m = tf.saved_model.load(HANDLE)          # model added from Kaggle Models (no internet needed)
    else:
        import tensorflow_hub as hub              # download from TF Hub (needs Internet on)
        _m = hub.load(HANDLE)
    def embed(batch):
        x = tf.constant(np.asarray(batch, dtype=np.float32))
        if callable(_m):
            return _m(x).numpy()
        sig = _m.signatures["serving_default"]          # older SavedModel format
        return list(sig(x).values())[0].numpy()
    assert embed(np.zeros((1, IMG, IMG, 3), np.float32)).shape == (1, 1280), "Unexpected feature size"
    EXTRACTOR = "MobileNetV2 1.0 224 (ImageNet) feature vector"

def features(df, flip=False):
    out = []
    for i in range(0, len(df), 32):
        batch = [load_rgb(p) for p in df.path.iloc[i:i + 32]]
        if flip: batch = [b[:, ::-1, :] for b in batch]
        out.append(embed(np.stack(batch)))
    return np.concatenate(out)

t0 = time.time()
tr, va, te = data[data.split == "train"], data[data.split == "val"], data[data.split == "test"]
X_tr, X_tr_flip, X_va, X_te = features(tr), features(tr, flip=True), features(va), features(te)
y_tr, y_va, y_te = (tr.label == "dirty").astype(int).values, (va.label == "dirty").astype(int).values, (te.label == "dirty").astype(int).values
print(f"Features: {X_tr.shape[1]} per image, extracted in {time.time() - t0:.1f} s")

# %% [markdown]
# ## 5. Train the classifier
# The regularisation strength `C` is chosen with 5-fold cross-validation on the training set.
# The final model is trained on the training images plus their mirror images (data augmentation).

# %%
pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
grid = GridSearchCV(pipe, {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy")
grid.fit(X_tr, y_tr)
best_C = grid.best_params_["clf__C"]
cv_mean = float(grid.best_score_); cv_std = float(grid.cv_results_["std_test_score"][grid.best_index_])
print(f"Best C = {best_C}, 5-fold CV accuracy = {cv_mean:.3f} ± {cv_std:.3f}")

fig, ax = plt.subplots(figsize=(5, 3.2))
Cs = grid.cv_results_["param_clf__C"].data.astype(float)
ax.errorbar(Cs, grid.cv_results_["mean_test_score"], yerr=grid.cv_results_["std_test_score"], marker="o", color="#2F6FD0", capsize=3)
ax.set_xscale("log"); ax.set_xlabel("C (regularisation)"); ax.set_ylabel("CV accuracy"); ax.set_title("Hyper-parameter search")
fig.tight_layout(); fig.savefig(f"{OUT}/results/hyperparameter_search.png"); plt.close(fig)

model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=best_C, max_iter=5000))])
model.fit(np.vstack([X_tr, X_tr_flip]), np.concatenate([y_tr, y_tr]))

# %% [markdown]
# ## 6. Evaluate on the validation and test sets

# %%
def scores(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    out = {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
           "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0)}
    try: out["roc_auc"] = roc_auc_score(y, p)
    except ValueError: out["roc_auc"] = float("nan")
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    return {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in out.items()}

p_va, p_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
p_tr = model.predict_proba(X_tr)[:, 1]
M = {"train": scores(y_tr, p_tr), "val": scores(y_va, p_va), "test": scores(y_te, p_te)}
for k in ["train", "val", "test"]:
    print(f"{k:5s}  acc {M[k]['accuracy']:.3f}  prec {M[k]['precision']:.3f}  rec {M[k]['recall']:.3f}  f1 {M[k]['f1']:.3f}  auc {M[k]['roc_auc']:.3f}")
print()
print(classification_report(y_te, (p_te >= 0.5).astype(int), target_names=["clean", "dirty"]))

def plot_cm(cm, title, path):
    cm = np.array(cm); fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["clean", "dirty"]); ax.set_yticks([0, 1], ["clean", "dirty"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
plot_cm(M["test"]["confusion_matrix"], "Confusion matrix (test set)", f"{OUT}/results/confusion_matrix.png")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
for name, y, p, c in [("Validation", y_va, p_va, "#2F6FD0"), ("Test", y_te, p_te, "#E8701A")]:
    fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.2f})", color=c, lw=2)
ax.plot([0, 1], [0, 1], "--", color="#999"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve"); ax.legend(loc="lower right"); fig.tight_layout()
fig.savefig(f"{OUT}/results/roc_curve.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.hist(p_te[y_te == 0], bins=10, range=(0, 1), alpha=.8, color=CLR["clean"], label="actually clean")
ax.hist(p_te[y_te == 1], bins=10, range=(0, 1), alpha=.8, color=CLR["dirty"], label="actually dirty")
ax.axvline(0.5, color="#333", ls="--"); ax.set_xlabel("Predicted probability of 'dirty'"); ax.set_ylabel("Test images")
ax.set_title("Prediction confidence (test set)"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/results/probability_histogram.png"); plt.close(fig)

te = te.assign(prob_dirty=np.round(p_te, 4), predicted=np.where(p_te >= 0.5, "dirty", "clean"))
te[["path", "label", "predicted", "prob_dirty"]].to_csv(f"{OUT}/results/test_predictions.csv", index=False)
wrong = te[te.label != te.predicted]
if len(wrong):
    n = min(len(wrong), 6); fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8)); axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, wrong.head(n).iterrows()):
        ax.imshow(Image.open(os.path.join(OUT, "dataset", r.path))); ax.axis("off")
        ax.set_title(f"actual {r.label}\npredicted {r.predicted} ({r.prob_dirty:.2f})", fontsize=9)
    fig.suptitle("Misclassified test images"); fig.tight_layout()
    fig.savefig(f"{OUT}/results/misclassified.png"); plt.close(fig)
print(f"{len(wrong)} of {len(te)} test images misclassified")

# %% [markdown]
# ## 7. Export the model for the web app
# The scaler is folded into the weights, so the browser only needs: `p = sigmoid(w · features + b)`.

# %%
scale, clf = model.named_steps["scale"], model.named_steps["clf"]
w = clf.coef_[0] / scale.scale_
b = float(clf.intercept_[0] - np.sum(clf.coef_[0] * scale.mean_ / scale.scale_))
# check that the folded model gives the same answers
assert np.allclose(1 / (1 + np.exp(-(X_te @ w + b))), p_te, atol=1e-6)

export = {
    "name": "CityEye street cleanliness model",
    "version": 1,
    "trained_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "classes": ["clean", "dirty"],
    "positive_class": "dirty",
    "threshold": 0.5,
    "feature_extractor": {"name": EXTRACTOR, "browser": "@tensorflow-models/mobilenet version 2, alpha 1.0, infer(image, true)",
                          "input": "RGB, resized to 224x224, pixel values 0-1"},
    "dataset": {"name": "Clean/Dirty Road Classification", "source": "https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification",
                "images": int(len(data))},
    "test_accuracy": M["test"]["accuracy"],
    "bias": round(b, 6),
    "weights": [round(float(x), 6) for x in w],
}
json.dump(export, open(f"{OUT}/model/cleanliness_model.json", "w"))
print("Model file:", os.path.getsize(f"{OUT}/model/cleanliness_model.json") // 1024, "KB,", len(w), "weights")

# %% [markdown]
# ## 8. (Optional) Compare with Gemini on the same test images
# Skipped automatically if no `GEMINI_API_KEY` secret is set or Internet is off.

# %%
GEMINI = None
key = os.environ.get("GEMINI_API_KEY")
if not key and not STUB:
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception:
        key = None
if key:
    import requests
    PROMPT = ("You analyse one street photo for a smart-city cleanliness system. "
              "garbage_detected is true if litter or waste is visible on the road, footpath, drain or open ground. "
              "cleanliness_score is 0-100 (100 = perfectly clean).")
    SCHEMA = {"type": "OBJECT", "properties": {"garbage_detected": {"type": "BOOLEAN"}, "cleanliness_score": {"type": "INTEGER"}},
              "required": ["garbage_detected", "cleanliness_score"]}
    models = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
    def ask(img_b64):
        global models
        for mdl in list(models):
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent",
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"}, timeout=60,
                              json={"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}, {"text": PROMPT}]}],
                                    "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json", "responseSchema": SCHEMA}})
            if r.status_code == 404: models.remove(mdl); continue
            if r.status_code == 429: time.sleep(30); return ask(img_b64)
            r.raise_for_status()
            txt = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
            return mdl, json.loads(txt)
        raise RuntimeError("No Gemini model available")
    rows, t_all = [], []
    for _, r in te.iterrows():
        im = Image.open(os.path.join(OUT, "dataset", r.path)).convert("RGB"); im.thumbnail((768, 768))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=72)
        t = time.time()
        try:
            mdl, ans = ask(base64.b64encode(buf.getvalue()).decode())
        except Exception as e:
            print("Gemini failed on", r.path, e); continue
        t_all.append(time.time() - t)
        rows.append({"path": r.path, "label": r.label, "gemini_garbage": bool(ans["garbage_detected"]),
                     "gemini_score": ans.get("cleanliness_score"), "model": mdl, "seconds": round(t_all[-1], 2)})
        time.sleep(float(os.environ.get("CITYEYE_GEMINI_DELAY", "4.5")))  # stay inside the free-tier rate limit
    if rows:
        g = pd.DataFrame(rows); g.to_csv(f"{OUT}/results/gemini_eval.csv", index=False)
        GEMINI = scores((g.label == "dirty").astype(int).values, g.gemini_garbage.astype(float).values)
        GEMINI["model"] = g.model.iloc[-1]; GEMINI["images"] = int(len(g))
        GEMINI["median_seconds"] = round(float(np.median(t_all)), 2)
        print("Gemini on test set:", {k: GEMINI[k] for k in ["accuracy", "precision", "recall", "f1"]}, "median", GEMINI["median_seconds"], "s")
else:
    print("Gemini comparison skipped (no GEMINI_API_KEY secret).")

# %% [markdown]
# ## 9. Save the results and package everything

# %%
if GEMINI:
    names = ["Trained model\n(on-device)", f"Gemini\n({GEMINI['model']})"]
    fig, ax = plt.subplots(figsize=(6, 3.4)); x = np.arange(4); wdt = 0.36
    for i, (src, c) in enumerate([(M["test"], "#0E6B45"), (GEMINI, "#2F6FD0")]):
        vals = [src[k] for k in ["accuracy", "precision", "recall", "f1"]]
        ax.bar(x + (i - .5) * wdt, vals, wdt, color=c, label=names[i].replace("\n", " "))
        for xi, v in zip(x, vals): ax.text(xi + (i - .5) * wdt, v + .01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1"]); ax.set_ylim(0, 1.3); ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.set_title("Test set: trained model vs Gemini"); fig.tight_layout()
    fig.savefig(f"{OUT}/results/model_vs_gemini.png"); plt.close(fig)

metrics = {
    "dataset": {"name": "Clean/Dirty Road Classification", "kaggle": "faizalkarim/cleandirty-road-classification",
                "images": int(len(data)), "per_class": data.label.value_counts().to_dict(),
                "split": {s: data[data.split == s].label.value_counts().to_dict() for s in ["train", "val", "test"]},
                "duplicates_removed": int(len(files) - len(data))},
    "method": {"feature_extractor": EXTRACTOR, "features": int(X_tr.shape[1]), "classifier": "Logistic regression",
               "best_C": best_C, "augmentation": "horizontal flip", "cv_folds": 5,
               "cv_accuracy_mean": round(cv_mean, 4), "cv_accuracy_std": round(cv_std, 4), "seed": SEED},
    "results": M,
    "gemini": GEMINI,
    "created": export["trained_at"],
}
json.dump(metrics, open(f"{OUT}/results/metrics.json", "w"), indent=2)

open(f"{OUT}/dataset/README.md", "w").write(f"""# Dataset: Clean/Dirty Road Classification

Source: https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification (Kaggle, uploaded by Faizal Karim).
Check the licence on the Kaggle page before redistributing, and keep this credit.

* {len(data)} unique images ({counts['clean']} clean, {counts['dirty']} dirty), resized to at most 800 px.
* `split.csv` lists every image with its label and split (train / val / test, 70 / 15 / 15, stratified, seed {SEED}).
* Used to train the CityEye on-device street cleanliness model (`model/cleanliness_model.json`).
""")

zip_path = os.path.join(WORK, "cityeye_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(OUT):
        for f in fs:
            full = os.path.join(root, f); z.write(full, os.path.relpath(full, OUT))
print("\nDone. Download", zip_path, f"({os.path.getsize(zip_path) // 1024} KB)")
print("Unzip it into your CityEye project folder: it adds dataset/, model/ and results/.")
print(json.dumps({"test": {k: M["test"][k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
                  "cv": f"{cv_mean:.3f} ± {cv_std:.3f}", "gemini": GEMINI and {k: GEMINI[k] for k in ["accuracy", "f1"]}}, indent=1))
