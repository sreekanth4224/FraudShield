# Training our own detectors (frozen backbone + small head)

We don't retrain big networks. A frozen pretrained model turns each clip into
1024 numbers; we train only a small classifier ("head") on those numbers.

| | Voice (do first) | Face (if time is left) |
|---|---|---|
| Frozen backbone | XLS-R 300M (`facebook/wav2vec2-xls-r-300m`), pretrained on 128 languages | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) image encoder |
| What it sees | 6 s speech segments, 16 kHz (same window as the live engine) | the same face crop the live engine uses (box + 20 % margin) |
| Head | logistic regression or MLP 1024→256→1, picked automatically | same |
| Data | real + fake Indian-language speech | real + deepfake webcam videos (e.g. DeepSpeak v2) |
| Shows on dashboard as | its own check in the Voice module | its own check in the Face module |

The head's weight in the risk score scales with its **held-out test AUC**,
so a weak head can't cause false alarms. Without a trained head nothing changes.

---

## 0. One-time setup (in the FraudShield folder, venv active)

```cmd
venv\Scripts\activate
pip install -U "torch>=2.6" pyarrow
hf auth login
```
(`huggingface-cli login` on older versions; paste a token from huggingface.co → Settings → Access Tokens.)
Accept the terms on each dataset page you use (IndicSynth, DeepSpeak) first.

## 1. Voice data → `data\voice\`

Every speaker must have their own sub-folder (or a `group` in the manifest), so
test speakers are never heard in training. Aim for ≥ 500 real + ≥ 500 fake clips
from ≥ 30 speakers each.

**Fake:** IndicSynth (synthetic only, 12 Indian languages)
```cmd
python -m tools.fetch_hf_audio --repo vdivyasharma/IndicSynth --list
python -m tools.fetch_hf_audio --repo vdivyasharma/IndicSynth --label fake --filter hindi --n 600 --tag hindi
python -m tools.fetch_hf_audio --repo vdivyasharma/IndicSynth --label fake --filter malayalam --n 600 --tag malayalam
```

**Real:** genuine speech in the same languages, e.g. FLEURS
```cmd
python -m tools.fetch_hf_audio --repo google/fleurs --revision refs/convert/parquet --list --filter hi_in
python -m tools.fetch_hf_audio --repo google/fleurs --revision refs/convert/parquet --label real --filter hi_in --n 600 --tag hindi
python -m tools.fetch_hf_audio --repo google/fleurs --revision refs/convert/parquet --label real --filter ml_in --n 600 --tag malayalam
```
Use `--list` first: the `--filter` text must match the file paths it prints.
If a repo has another layout, download by hand into
`data\voice\real\<speaker>\*.wav` and `data\voice\fake\<speaker>\*.wav`.

**Best extra test set:** record 10+ friends reading a paragraph on their phones
(real), and clone some of them with a free voice-cloning tool (fake). Put them in
`data\voice_own\real\<name>\` and `data\voice_own\fake\<name>\`.

## 2. Extract voice features (GPU: roughly 10-30 min for ~2,000 clips)

```cmd
python -m tools.extract_voice_features --manifest data\voice\manifest.csv --augment --out features\voice.npz
```
or with folders: `--real data\voice\real --fake data\voice\fake`.
Quick trial first: add `--limit 20`. Interrupted? Run the same command again; it resumes.

## 3. Train + test the voice head (CPU, a few minutes)

```cmd
python -m tools.train_head features\voice.npz --label "Indian-language voice detector (ours)"
```
It prints the validation AUC of every layer × model, then the **test** result
on unseen speakers (AUC, EER, % fakes caught, % real flagged, per language), and
saves `models\custom\voice_head.npz` + `voice_report.json`.

## 4. Face (optional, same pattern)

Put videos in `data\face\real\<person>\*.mp4` and `data\face\fake\<person>\*.mp4`
(the same person's folder name under both), or write `data\face\manifest.csv`.
```cmd
python -m tools.extract_face_features --real data\face\real --fake data\face\fake --out features\face.npz
python -m tools.train_head features\face.npz --label "Webcam deepfake detector (ours)"
```
DeepSpeak v2 is 134 GB: download only a few hundred videos (balanced real/fake,
many different people).

## 5. Use it

```cmd
python -m tools.download_models
python app.py
```
`download_models` should list `CustomVoiceDetector ready`. On the dashboard the
AI pill shows "+ 1 ours" and the Voice module gets a new check with your label.
Turn the heads off with `set FRAUDSHIELD_CUSTOM=0`.

## Honest-results checklist (judges will ask)

* Report the **test** numbers (unseen speakers), never the training ones.
* Real and fake came from different datasets, so a head can learn "which
  recording setup" instead of "real vs fake". `--augment` (phone-call copies of
  both classes) reduces this; the own-recordings test set in step 1 is the real
  proof. Evaluate on it with a head trained without it.
* Say what it is: "XLS-R features + a classifier we trained on Indian-language
  real vs synthetic speech, AUC X on held-out speakers."
* Licences: IndicSynth is CC BY-NC 4.0 and DeepSpeak is academic-use only,
  which is fine for a hackathon but not for a commercial product.

## Troubleshooting

| Problem | Fix |
|---|---|
| `the test split has only one class` | more speakers of both classes, or `--test-pct 30` |
| NaN warning, "switching to float32" | harmless (half precision overflowed); or `set FRAUDSHIELD_FP32=1` |
| `torch.load` / weights-only error loading XLS-R | `pip install -U "torch>=2.6"` (use the cu124 build for GPU) |
| out of GPU memory | close other apps; the two backbones need ~1-2 GB together |
| face extraction uses "OpenCV Haar" | `pip install mediapipe==0.10.14` |
