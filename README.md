<p align="center">
  <img src="https://github.com/Runa8147/Hackathena_Readme_Template/blob/d0add823684f0ac28b76a99636c729f80b0ca8ff/hackathena_banner.png" alt="Hackathena '26 2.0" width="100%">
</p>

<h1 align="center">FraudShield</h1>

<p align="center">
  <strong>Real-time deepfake detection for video KYC calls: face, voice and lip-sync forensics with live liveness challenges.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Hackathena-'26%202.0-black?style=for-the-badge" alt="Hackathena">
  <img src="https://img.shields.io/badge/Theme-AI%20Fraud%20Detection-red?style=for-the-badge" alt="Theme">
  <img src="https://img.shields.io/badge/Status-Prototype-white?style=for-the-badge&labelColor=black" alt="Status">
</p>

---

## 👥 Team

**Team Name:** `n00bmasters`

| Member     | Role      | Institution |
| ---------- | --------- | ----------- |
| **Sreekanth S** | Team Lead | Amrita Vishwa Vidyapeetham, Coimbatore   |
| **Nethra Iyer** | [Role]    | Amrita Vishwa Vidyapeetham, Coimbatore  |
| **Lakshmi Harshita** | [Role]    | Amrita Vishwa Vidyapeetham, Coimbatore   |
| **Midhun Chelat** | [Role]    | Amrita Vishwa Vidyapeetham, Coimbatore   |

---

## 🎯 Problem Statement

The rapid advancement of generative AI has made it increasingly difficult to distinguish authentic content from artificially generated or manipulated content.

Deepfakes, cloned voices, synthetic images, fabricated documents, and other AI-assisted techniques can enable **impersonation, misinformation, identity theft, financial fraud, and social engineering attacks**.

Banks and fintechs onboard customers through **video KYC calls**, where an officer verifies a person's identity over a live video call. Real-time face swaps, cloned voices, and replayed or photographed faces let a fraudster pass as someone else and open accounts in their name. The officer has only their eyes to tell a real face from a generated one.

---

## 💡 Solution

### FraudShield

**FraudShield** is a **web dashboard backed by a local Python engine** that detects **deepfake and presentation attacks during live video KYC calls**.

The officer shares their screen (or just the call's browser tab) with FraudShield and runs the call as usual, in any conferencing app. FraudShield finds the customer's face in the call window and listens to the call audio. Trained deepfake detectors, liveness checks, a voice-clone detector and a lip-sync check feed a **deepfake-risk score that updates every second**, with a verdict, the evidence behind it and suggested challenges.

### Key Features

* 🔴 **Works with any video-call app**: screen or tab capture, so no integration with Zoom, Teams, Meet or the bank's KYC portal is needed.
* ⚪ **Trained deepfake detectors**: an MS-EffGCViT-B0 ensemble (FaceForensics++ and Celeb-DF v2) on the face, a wav2vec2 voice-clone detector on the audio, and a ViT detector for fully AI-generated video.
* ⚫ **Liveness and replay checks**: blinks, expression dynamics, 3-D parallax, remote pulse (rPPG), screen-recapture moiré, and lip-sync between mouth and voice.
* 🔴 **Live liveness challenges**: "read this phrase aloud" (transcribed with Whisper and checked against lip movement), "wave a hand across the face" and "show a side profile", which break most real-time face swaps.
* ⚪ **Explainable, exportable verdicts**: a live gauge, risk timeline and per-check evidence, a pop-out HUD that floats over the call, and a downloadable JSON evidence report.

---

## 🔄 How It Works

```text
 Officer's browser                          FraudShield engine (Python, localhost)
 ─────────────────                          ──────────────────────────────────────
 getDisplayMedia (entire screen             FaceStream   find customer face on screen (tiled BlazeFace)
   + system audio)                            │          → Face Mesh 478 landmarks on native pixels
   │                                          │          → trained deepfake detector (FF++ + Celeb-DF)
   │                                          │          → liveness checks over a rolling 20 s window
 capture-worker.js ── JPEG frames 15 fps ──▶  │
   (Web Worker: keeps running when the     VoiceStream  rolling 12 s of call audio → voice-clone
    dashboard is behind the call window)      │          detector + call-tuned voice checks
 pcm-worklet.js ───── PCM audio ──────────▶  avsync      lip opening ↔ voice envelope correlation
                                              │
 dashboard  ◀──── state JSON every 1 s ───── fusion      confidence-weighted score, smoothing,
   gauge · overlay · timeline · evidence                 verdict with hysteresis, events, challenges
```

Evidence is split into two groups, scored separately (`modules/scoring.py`):

- **Synthesis**: is the face or voice itself generated?
- **Liveness**: is a live person in front of the camera, rather than a photo, a screen or a replay?

A module's score is the **worse** of the two groups. A deepfake blinks, turns its head and moves its lips like a real person, so it passes every liveness check. Averaging those "clear" results with the synthesis evidence would let deepfakes through as genuine.

**Face** (`modules/face_module.py`)

| Check | Group | Catches |
|---|---|---|
| **AI deepfake detector**: MS-EffGCViT-B0 × 2 (FaceForensics++ and Celeb-DF v2 checkpoints), 4 face crops/s | synthesis | face swaps, reenactment, neural textures |
| **AI-generated scene detector** (CommunityForensics ViT) | synthesis | fully generated video (Sora / Veo / Kling style) |
| Landmark jitter (residual after rigid alignment) | synthesis (minor) | frame-by-frame synthesised faces |
| Face-swap seam (cheek vs. neck sensor noise) | synthesis (minor) | swapped faces blended onto a real head |
| Face tracking / second face | liveness | face-swap dropouts, coached / assisted sessions |
| Blink behaviour (rate, depth, left/right symmetry) | liveness | photos, replays |
| Expression dynamics | liveness | photo, frozen or looped frame |
| 3-D parallax (landmarks vs. a single homography) | liveness | flat photo / screen held to the camera |
| Remote pulse / rPPG (POS on forehead + cheeks) | liveness | photos, replays |
| Screen recapture (moiré peaks + glare) | liveness | a phone or monitor held up to the customer's camera |

The two face checkpoints are combined by a small logistic stacker (`modules/calibration.py`). The detector's reliability falls with the face's *effective* resolution, so a blurry, upscaled low-bandwidth stream counts for less.

**Voice** (`modules/voice_module.py`): a trained **voice-clone detector** (wav2vec2-base fine-tuned on the In-the-Wild corpus) plus intonation, pitch micro-jitter, periodicity (HNR), rhythm & pauses and loudspeaker replay. Checks that conferencing apps break (digital silence, bandwidth, breathing) are reported as *not applicable* on call audio.

**Lip-sync** (`modules/avsync.py`): peak cross-correlation (±400 ms) between lip opening and the voice envelope, plus a hard alarm when a voice is heard but the lips don't move. This catches voice-overs, cloned voices played over a real or recorded face, and replays.

**Challenges** (`modules/challenge.py`): a random phrase to read aloud (Whisper transcription + lip-sync + reaction time), a hand passed across the face, and a side profile. A passed challenge pulls the score down; a failed one is pinned on the risk timeline.

**Fusion** (`modules/fusion.py`): each check reports a risk, a weight and a reliability for the current window. Fusion is a confidence-weighted average in which one confident red module can't be averaged away. Risk rises fast and decays slowly.

Verdicts: **Calibrating** (not enough evidence yet), **Likely genuine** (< 35), **Suspicious** (35–65, run a liveness challenge), **Likely deepfake** (≥ 65, stop and escalate).

---

## 🛠️ Technology Stack

### Software

| Layer          | Technologies                             |
| -------------- | ---------------------------------------- |
| **Frontend**   | HTML, CSS, vanilla JavaScript; Screen Capture API, Web Worker, AudioWorklet, Document Picture-in-Picture |
| **Backend**    | Python 3.11, FastAPI, Uvicorn (WebSocket streaming) |
| **AI / ML**    | PyTorch, timm, Hugging Face Transformers; MS-EffGCViT-B0 (FF++ / Celeb-DF v2), wav2vec2 voice-clone detector, CommunityForensics ViT, Whisper (ASR for challenges), MediaPipe BlazeFace + Face Mesh |
| **Database**   | None: frames and audio are analysed in memory only |
| **Processing** | OpenCV, NumPy, SciPy, librosa, soundfile |
| **Deployment** | Runs locally on the officer's machine (`localhost`), CPU or NVIDIA GPU |

### Tools

* Git & GitHub
* Hugging Face Hub (model weights and datasets)
* Chrome / Edge (screen + system-audio capture)

---

## 📊 Results

`tools/evaluate.py` runs labelled clips through exactly the live pipeline. The sample is 69 public clips (66 with a trackable face): 35 from Google's DeepFakeDetection (DFD) set and 34 from Celeb-DF v2, about half real and half fake. Each clip is run at full resolution, as a 720p call (1280 px, JPEG q70) and as a poor 360p call (640 px, JPEG q40, upscaled).

Face module with the shipped calibration (fitted on these same clips, so *in-sample*):

| Video quality | Face score AUC | Fakes flagged *Likely deepfake* (≥ 65) | Genuine flagged ≥ 65 | Genuine flagged *Suspicious* (≥ 35) |
|---|---|---|---|---|
| Full resolution | 0.95 | 25 / 33 | 0 / 32 | 7 / 32 |
| 720p call | 0.92 | 22 / 34 | 1 / 32 | 12 / 32 |
| 360p call | 0.82 | 11 / 34 | 0 / 32 | 8 / 32 |

Out-of-sample check (calibrate on one dataset, test on the other, full resolution):

| Tested on | Detector AUC | Fakes flagged ≥ 65 | Genuine flagged ≥ 65 |
|---|---|---|---|
| Celeb-DF (calibrated on DFD) | 0.98 | 14 / 18 | 1 / 16 |
| DFD (calibrated on Celeb-DF) | 0.89 | 3 / 15 | 0 / 16 |

| Metric                 | Result                                      |
| ---------------------- | ------------------------------------------- |
| **Face detector AUC**  | 0.95 full resolution · 0.92 at 720p · 0.82 at 360p |
| **Voice clones**       | 0 false alarms on 25 genuine clips; 4 / 18 commercial clones flagged ≥ 65 |
| **Response Time**      | Score updates every 1 s; frames analysed at 15 fps |
| **Resource use**       | ~30 % of one CPU core (engine) + ~97 % (Chrome dashboard + capture) |
| **Supported Input**    | Live screen / tab capture with system audio; video and audio files via `tools/` |

What this shows:

- **The trained detector is what catches deepfakes.** The forensic checks were near chance (AUC ≈ 0.5) at telling a deepfake from a real face; they matter for photo, screen and replay attacks.
- **It degrades with bandwidth.** At 360p there is too little real detail in the face, so reliability drops and the dashboard asks the officer to have the customer move closer.
- **Calibration is domain-specific.** Calibrate on recordings from your own KYC channel (see below).
- **These numbers are a sanity check, not a certification.** 66 clips give wide error bars, and the Celeb-DF checkpoint may have seen some of these clips in training.
- **Voice-clone detection is the weakest part.** A "clear" from the voice detector is treated as weak evidence; lip-sync and the read-aloud challenge are the main voice defences.

---

## 🚀 Getting Started

### Prerequisites

* Windows with Python 3.11
* Chrome or Edge
* Optional: an NVIDIA GPU with CUDA
* A few GB of disk for model weights (downloaded from Hugging Face into `./models`)

### Setup tutorial (one time)

Run these in **Command Prompt** or **PowerShell**.

**1. Get the code**

```bash
git clone https://github.com/sreekanth4224/FraudShield.git
cd FraudShield
```

**2. Create a virtual environment**

```bash
py -3.11 -m venv venv
venv\Scripts\python -m pip install --upgrade pip
```

**3. Install PyTorch** (pick one)

```bash
:: NVIDIA GPU (recommended, much faster)
venv\Scripts\pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

:: CPU only
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Always install torch and torchvision **together from the same index**. A mismatched torchvision breaks the detectors (`operator torchvision::nms does not exist`).

**4. Install the other dependencies**

```bash
venv\Scripts\pip install -r requirements.txt
```

**5. Download the model weights** (into `./models`, a few minutes the first time)

```bash
venv\Scripts\python -m tools.download_models
```

**6. Check the setup** (optional: models, GPU and webcam)

```bash
venv\Scripts\python -m tools.diagnose
```

### Run the app

```bash
cd FraudShield
venv\Scripts\python app.py
```

The dashboard opens automatically at **http://localhost:8000**. Stop the server with **Ctrl + C**.

Useful options:

```bash
venv\Scripts\python app.py --port 8080          # use another port
venv\Scripts\python app.py --no-browser         # don't open the dashboard automatically
venv\Scripts\python app.py --window full        # dashboard window: right (default) | left | full
```

Screen capture only works on a secure page, so open the dashboard as `localhost` on the officer's own machine (not via a LAN IP). To reach it from another machine, put the server behind HTTPS.

### Optional settings

No environment variables are required. To change a setting, set it before `python app.py`:

```bash
:: Command Prompt
set FRAUDSHIELD_SCENE=0

# PowerShell
$env:FRAUDSHIELD_SCENE = "0"
```

```env
FRAUDSHIELD_FACE_MODELS=b0-ff++,b0-celeb      # add b5-ff++ for a more compression-robust face model (GPU recommended)
FRAUDSHIELD_SCENE=1                           # 0 turns off the AI-generated-scene detector
FRAUDSHIELD_ASR=openai/whisper-small          # speech model for the read-aloud challenge
FRAUDSHIELD_CUSTOM=0                          # 1 enables our own trained heads (see TRAINING.md)
FRAUDSHIELD_FP32=0                            # 1 disables half precision on GPU
FRAUDSHIELD_WINDOW=right                      # dashboard window placement: right | left | full
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `py -3.11` not found | Install Python 3.11 from python.org and tick *Add to PATH* |
| `operator torchvision::nms does not exist` | Reinstall torch and torchvision together from the same index (step 3) |
| **AI models** pill is not green | Hover over it to see why; usually step 5 was skipped. The app still runs on the forensic checks alone |
| Share screen does nothing / is blocked | Use Chrome or Edge and open `http://localhost:8000`, not an IP address |
| No voice score | Tick *Also share tab audio* / *Also share system audio* when sharing |
| Port 8000 already in use | `venv\Scripts\python app.py --port 8080` |

### Using it

1. Click **Share screen** and pick either **the call's browser tab** with *Also share tab audio* ticked (most private), or **Entire screen** with *Also share system audio* ticked (for desktop apps like Zoom or Teams).
2. Start or continue the video call. FraudShield locks onto the largest face on the screen and ignores the officer's small self-view. If several people are on screen, click **Select region** and drag a box around the customer's tile.
3. Keep the dashboard on a second monitor or beside the call; it keeps analysing when the call window is in front. **Pop-out HUD** floats the score over the call.
4. Follow the **Suggested challenges** when evidence is missing or suspicious.
5. **Report** downloads a JSON evidence report: verdict, every check, the risk timeline and the event log. **New customer** resets the session.

---

## 🧪 Example

*Illustrative walk-through of the dashboard's output for a face-swap attempt.*

**Input**

```text
Live video KYC call shared from Google Meet: customer's face tile + call audio
```

**System Analysis**

```text
Face      AI deepfake detector       ALERT   (synthesis)
          Blink / expression / pulse CLEAR   (liveness)
Voice     Voice-clone detector       WATCH
Lip-sync  Lips vs. voice             CLEAR
Challenge "Hand across the face"     FAILED  (face swap broke up under occlusion)
```

**Result**

```text
LIKELY DEEPFAKE
Risk score: 78 / 100
Risk Level: HIGH (≥ 65: stop the call and escalate)
```

---

## 🧰 Calibrating and Training

**Calibrate on your own KYC recordings.** Put genuine and deepfake recordings from your channel in folders and run:

```bash
venv\Scripts\python -m tools.evaluate --real kyc\real --fake kyc\fake --degrade none --calibrate
```

It prints each check's AUC and the catch / false-alarm counts at the dashboard thresholds, then writes `models/calibration.json`, which the engine loads at start-up. Add `--cross` for a held-out check; use `--real-audio` / `--fake-audio` for voice.

**Check a single recording** exactly as the live engine would, and optionally log it as labelled evidence:

```bash
venv\Scripts\python -m tools.check_video "C:\path\video.mp4" --label fake
venv\Scripts\python -m tools.train_evidence
```

**Train our own detectors.** FraudShield can run small classifiers trained on top of frozen pretrained models (XLS-R 300M for Indian-language voice clones, CLIP ViT-L/14 for webcam face deepfakes). Each shows up as its own check, weighted by its held-out AUC. See [TRAINING.md](TRAINING.md).

**Diagnose a setup** (models, GPU, webcam): `venv\Scripts\python -m tools.diagnose`.

---

## 🔐 Security & Privacy

* Frames and audio go only to the engine on `localhost`, are analysed in memory (last ~30 s) and are never written to disk.
* The engine listens on `127.0.0.1` only, and its WebSocket accepts connections only from pages served on `localhost`, so other websites open in the officer's browser can't connect to it.
* Sharing just the call's browser tab keeps everything else on the officer's screen out of the capture.
* Once the face is found, only the region around it is streamed to the engine.
* No accounts, API keys or cloud services: all models run locally.

---

## ⚙️ Performance

FraudShield has to run next to the video call without slowing it down. Measured with a live session (% of one CPU core):

| | Before optimisation | Now |
|---|---|---|
| Chrome (dashboard + capture) | ~252 % | ~97 % |
| Python engine | ~99 % | ~30 % |
| Frames analysed per second | 11 | 15 |

- **Face-region streaming**: the whole screen is sent only while searching for the customer; otherwise just the face region at native resolution.
- **Work done once**: per-frame features are computed on arrival; slower checks refresh every 2 s.
- **GPU without CPU spin**: the face detector replays as a single CUDA graph, and GPU results are awaited without busy-waiting.
- **No thread storms**: numeric libraries are capped at 2 threads with passive OpenMP waits.
- **A dashboard that rests**: canvases redraw only when their data changes, and nothing animates in a loop during a session.

---

## 🔮 Future Scope

* [ ] Calibrate and fine-tune on real KYC-channel recordings
* [ ] Stronger voice-clone detection for modern commercial TTS and Indian languages
* [ ] Re-train the face detector on newer real-time face-swap generators
* [ ] Integrate directly with bank video-KYC platforms instead of screen capture
* [ ] HTTPS deployment for remote officers and centralised audit of evidence reports
* [ ] More liveness challenges (random head-motion paths, lighting changes)

---

## ⚠️ Honest Limits

- **Deepfake generators move fast.** The face detector is trained on FaceForensics++ / Celeb-DF era face swaps and reenactment, so brand-new generators may evade it until it is re-trained. The liveness challenges are the fallback that breaks most real-time swaps.
- **Voice-clone detection is the weakest part** (see Results).
- **Poor video lowers detection power.** With a 360p stream or a small tile, the dashboard shows lower confidence instead of guessing.

---

## 🏆 Hackathena '26 2.0

This project was developed as part of **Hackathena '26 2.0**, organized by the **Department of Computer Science & Engineering and CESA, Jyothi Engineering College**.

### Theme

> **Detection and Prevention of AI-Based Frauds**

The project focuses on addressing emerging forms of fraud enabled or amplified by generative artificial intelligence, including **deepfakes, cloned voices, synthetic media, fabricated documents, and AI-assisted impersonation**.

---

## 📄 Repository Structure

```text
.
├── app.py                     # FastAPI server: static dashboard + /ws streaming endpoint
├── web/                       # Dashboard (frontend)
│   ├── index.html, styles.css, app.js
│   ├── capture-worker.js      # WebSocket + frame encoder (Web Worker)
│   └── pcm-worklet.js         # System-audio capture (AudioWorklet)
├── modules/                   # Detection engine
│   ├── live.py                # Per-call session: orchestration, events, challenges
│   ├── face_module.py         # Screen face finder + face forensics
│   ├── voice_module.py        # Voice forensics
│   ├── avsync.py              # Lip-sync check
│   ├── challenge.py           # Read-aloud, hand and side-profile challenges
│   ├── fusion.py              # Real-time fusion and verdict
│   ├── scoring.py             # Evidence aggregation (liveness vs. synthesis)
│   ├── calibration.py         # Detector output → risk
│   ├── evidence.py            # Learned fusion from labelled recordings
│   └── detectors/             # Face, voice, scene and custom-head detectors
│       └── deepguard/         # Vendored MS-EffGCViT model code (MIT, see NOTICE.md)
├── tools/                     # Evaluation, calibration, training and diagnostics CLIs
├── models/                    # Downloaded weights + calibration (git-ignored)
├── data/                      # Training / evidence data (git-ignored)
├── requirements.txt           # Python dependencies
├── TRAINING.md                # How to train our own detectors
└── README.md
```

---

## 📬 Contact

For questions, collaboration, or further information:

**Team:** n00bmasters
**Team Lead:** Sreekanth S
**GitHub:** [github.com/sreekanth4224/FraudShield](https://github.com/sreekanth4224/FraudShield)

---

<p align="center">

<strong>Hackathena '26 2.0</strong>

<br>

Detection & Prevention of AI-Based Frauds

<br><br>

<img src="https://img.shields.io/badge/Built%20at-Jyothi%20Engineering%20College-black?style=flat-square">
<img src="https://img.shields.io/badge/Hackathena-2026-red?style=flat-square">

</p>
