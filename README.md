<p align="center">
  <img src="https://github.com/Runa8147/Hackathena_Readme_Template/blob/d0add823684f0ac28b76a99636c729f80b0ca8ff/hackathena_banner.png" alt="Hackathena '26 2.0" width="100%">
</p>

<h1 align="center">FraudShield</h1>

<p align="center">
  <strong>Real-time deepfake detection for live video calls (Video KYC, remote job interviews and online exams): face, voice and lip-sync forensics with live liveness challenges.</strong>
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
| **Nethra Iyer** | Backend (Model)    | Amrita Vishwa Vidyapeetham, Coimbatore  |
| **Lakshmi Harshita** | Backend (Server)    | Amrita Vishwa Vidyapeetham, Coimbatore   |
| **Midhun Chelat** | Frontend (UI)    | Amrita Vishwa Vidyapeetham, Coimbatore   |

---

## 🎯 Problem Statement

Generative AI makes it hard to tell real people from synthetic ones. Deepfakes and cloned voices enable **impersonation, identity theft, fraud and social engineering**.

Identity is increasingly verified over live video: **Video KYC** for bank and fintech onboarding, **remote job interviews**, and **online exams**. Real-time face swaps, cloned voices and replayed or photographed faces let a fraudster pass as someone else, open accounts, get hired under a false identity, or sit an exam for another candidate. The reviewer (KYC officer, interviewer or proctor) has only their eyes to tell a real face from a generated one.

---

## 💡 Solution

### FraudShield

**FraudShield** is a **web dashboard backed by a local Python engine** that detects **deepfake and presentation attacks during live video calls**.

The reviewer shares their screen (or just the call's browser tab) with FraudShield and runs the call as usual, in any conferencing or proctoring app. FraudShield finds the participant's face, listens to the call audio, and produces a **deepfake-risk score every second** with a verdict, evidence and suggested challenges.

### Key Features

* 🔴 **Works with any video-call app**: screen or tab capture, no integration with Zoom, Teams, Meet, KYC portals or proctoring tools needed.
* ⚪ **Trained deepfake detectors**: MS-EffGCViT-B0 ensemble (FaceForensics++, Celeb-DF v2) for faces, wav2vec2 for voice clones, a ViT for fully AI-generated video.
* ⚫ **Liveness and replay checks**: blinks, expressions, 3-D parallax, remote pulse (rPPG), screen-recapture moiré, lip-sync.
* 🔴 **Live liveness challenges**: read a phrase aloud (Whisper + lip check), wave a hand across the face, show a side profile.
* ⚪ **Explainable, exportable verdicts**: live gauge, risk timeline, per-check evidence, pop-out HUD and JSON evidence report.

---

## 🔄 How It Works

```text
 Reviewer's browser                         FraudShield engine (Python, localhost)
 ──────────────────                         ──────────────────────────────────────
 getDisplayMedia (entire screen             FaceStream   find participant face on screen (tiled BlazeFace)
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

Evidence is scored in two groups (`modules/scoring.py`): **Synthesis** (is the face/voice generated?) and **Liveness** (is a live person present, not a photo, screen or replay?). A module's score is the **worse** of the two, since a deepfake passes liveness checks and averaging would let it through.

**Face** (`modules/face_module.py`)

| Check | Group | Catches |
|---|---|---|
| **AI deepfake detector**: MS-EffGCViT-B0 × 2 (FF++, Celeb-DF v2), 4 crops/s | synthesis | face swaps, reenactment, neural textures |
| **AI-generated scene detector** (CommunityForensics ViT) | synthesis | fully generated video (Sora / Veo / Kling) |
| Landmark jitter | synthesis (minor) | frame-by-frame synthesised faces |
| Face-swap seam (cheek vs. neck noise) | synthesis (minor) | swapped faces blended onto a real head |
| Face tracking / second face | liveness | swap dropouts, coached or assisted sessions |
| Blink behaviour | liveness | photos, replays |
| Expression dynamics | liveness | photo, frozen or looped frame |
| 3-D parallax | liveness | flat photo / screen held to the camera |
| Remote pulse / rPPG | liveness | photos, replays |
| Screen recapture (moiré + glare) | liveness | phone or monitor held up to the camera |

The two checkpoints are combined by a logistic stacker (`modules/calibration.py`); low effective resolution lowers their weight.

**Voice** (`modules/voice_module.py`): wav2vec2 voice-clone detector (In-the-Wild corpus) plus intonation, pitch jitter, HNR, rhythm and loudspeaker-replay checks. Checks that call codecs break are marked *not applicable*.

**Lip-sync** (`modules/avsync.py`): lip–voice cross-correlation (±400 ms), plus an alarm when voice is heard but lips don't move.

**Challenges** (`modules/challenge.py`): read-aloud phrase, hand across the face, side profile. Passing lowers the score; failing is pinned on the timeline.

**Fusion** (`modules/fusion.py`): confidence-weighted average where one confident red module can't be averaged away. Risk rises fast and decays slowly.

Verdicts: **Calibrating**, **Likely genuine** (< 35), **Suspicious** (35–65, run a challenge), **Likely deepfake** (≥ 65, stop and escalate).

---

## 🛠️ Technology Stack

### Software

| Layer          | Technologies                             |
| -------------- | ---------------------------------------- |
| **Frontend**   | HTML, CSS, vanilla JavaScript; Screen Capture API, Web Worker, AudioWorklet, Document Picture-in-Picture |
| **Backend**    | Python 3.11, FastAPI, Uvicorn (WebSocket streaming) |
| **AI / ML**    | PyTorch, timm, Hugging Face Transformers; MS-EffGCViT-B0, wav2vec2, CommunityForensics ViT, Whisper, MediaPipe BlazeFace + Face Mesh |
| **Database**   | None: frames and audio are analysed in memory only |
| **Processing** | OpenCV, NumPy, SciPy, librosa, soundfile |
| **Deployment** | Runs locally on the reviewer's machine (`localhost`), CPU or NVIDIA GPU |

### Tools

* Git & GitHub
* Hugging Face Hub (model weights and datasets)
* Chrome / Edge (screen + system-audio capture)

---

## 📊 Results

`tools/evaluate.py` runs labelled clips through the live pipeline: 69 public clips (35 DFD, 34 Celeb-DF v2, ~half fake), at full resolution, as a 720p call (JPEG q70) and as a 360p call (JPEG q40, upscaled).

Face module, shipped calibration (*in-sample*):

| Video quality | Face score AUC | Fakes flagged ≥ 65 | Genuine flagged ≥ 65 | Genuine flagged ≥ 35 |
|---|---|---|---|---|
| Full resolution | 0.95 | 25 / 33 | 0 / 32 | 7 / 32 |
| 720p call | 0.92 | 22 / 34 | 1 / 32 | 12 / 32 |
| 360p call | 0.82 | 11 / 34 | 0 / 32 | 8 / 32 |

Out-of-sample (calibrate on one dataset, test on the other):

| Tested on | Detector AUC | Fakes flagged ≥ 65 | Genuine flagged ≥ 65 |
|---|---|---|---|
| Celeb-DF (calibrated on DFD) | 0.98 | 14 / 18 | 1 / 16 |
| DFD (calibrated on Celeb-DF) | 0.89 | 3 / 15 | 0 / 16 |

| Metric                 | Result                                      |
| ---------------------- | ------------------------------------------- |
| **Voice clones**       | 0 false alarms on 25 genuine clips; 4 / 18 commercial clones flagged ≥ 65 |
| **Response Time**      | Score every 1 s; frames at 15 fps |
| **Resource use**       | ~30 % of one CPU core (engine) + ~97 % (Chrome) |
| **Supported Input**    | Live screen / tab capture with audio; files via `tools/` |

Takeaways:

- **The trained detector catches deepfakes**; forensic checks (AUC ≈ 0.5) matter for photo, screen and replay attacks.
- **Accuracy drops with bandwidth**; at 360p the dashboard asks the participant to move closer.
- **Calibration is domain-specific**; calibrate on recordings from your own channel.
- **A sanity check, not a certification**: 66 clips give wide error bars.
- **Voice-clone detection is the weakest part**; lip-sync and read-aloud are the main voice defences.

---

## 🚀 Getting Started

### Prerequisites

* Windows with Python 3.11
* Chrome or Edge
* Optional: NVIDIA GPU with CUDA
* A few GB of disk for model weights

### Setup tutorial (one time)

Run in **Command Prompt** or **PowerShell**:

```bash
git clone https://github.com/sreekanth4224/FraudShield.git
cd FraudShield
py -3.11 -m venv venv
venv\Scripts\python -m pip install --upgrade pip

:: PyTorch: NVIDIA GPU (recommended)
venv\Scripts\pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
:: ...or CPU only
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m tools.download_models     # weights into ./models
venv\Scripts\python -m tools.diagnose            # optional: check models, GPU, webcam
```

Install torch and torchvision **together from the same index**, or the detectors break.

### Run the app

```bash
venv\Scripts\python app.py                    # opens http://localhost:8000
venv\Scripts\python app.py --port 8080        # another port
venv\Scripts\python app.py --no-browser       # don't auto-open the dashboard
venv\Scripts\python app.py --window full      # right (default) | left | full
```

Screen capture needs a secure page: open it as `localhost`, not a LAN IP (use HTTPS for remote access).

### Using it

1. Click **Share screen**: pick the call's browser tab (with tab audio, most private) or the entire screen (with system audio, for desktop apps).
2. Start the call. FraudShield locks onto the largest face; use **Select region** if several people are on screen.
3. Keep the dashboard beside the call or use **Pop-out HUD** to float the score over it.
4. Follow the **Suggested challenges** when evidence is missing or suspicious.
5. **Report** downloads a JSON evidence report; **New customer** resets the session.

---

## 🧪 Example

*Illustrative output for a face-swap attempt.*

**Input**

```text
Live video call (Video KYC / interview / online exam) shared from Google Meet: participant's face tile + call audio
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

**Calibrate on your own recordings** (from your KYC, interview or exam channel):

```bash
venv\Scripts\python -m tools.evaluate --real calib\real --fake calib\fake --degrade none --calibrate
```

Prints per-check AUC and catch/false-alarm counts, and writes `models/calibration.json`. Add `--cross` for a held-out check; `--real-audio` / `--fake-audio` for voice.

**Check a single recording** as the live engine would:

```bash
venv\Scripts\python -m tools.check_video "C:\path\video.mp4" --label fake
venv\Scripts\python -m tools.train_evidence
```

**Train our own detectors** on frozen backbones (XLS-R 300M for Indian-language voice clones, CLIP ViT-L/14 for webcam deepfakes). See [TRAINING.md](TRAINING.md).

---

## 🔐 Security & Privacy

* Frames and audio stay on `localhost`, in memory (last ~30 s), never written to disk.
* The engine listens on `127.0.0.1` and accepts WebSockets only from `localhost` pages.
* Sharing just the call tab keeps the rest of the screen private; only the face region is streamed once found.
* No accounts, API keys or cloud services.

---

## ⚙️ Performance

Measured in a live session (% of one CPU core):

| | Before optimisation | Now |
|---|---|---|
| Chrome (dashboard + capture) | ~252 % | ~97 % |
| Python engine | ~99 % | ~30 % |
| Frames analysed per second | 11 | 15 |

Gains come from face-region-only streaming, computing features once, CUDA-graph face detection without busy-waiting, capped thread pools, and redrawing the dashboard only on data changes.

---

## 🔮 Future Scope

* [ ] Calibrate and fine-tune on real Video KYC, interview and exam recordings
* [ ] Stronger voice-clone detection for modern TTS and Indian languages
* [ ] Re-train the face detector on newer real-time face-swap generators
* [ ] Direct integration with Video KYC, hiring and proctoring platforms
* [ ] HTTPS deployment and centralised audit of evidence reports
* [ ] More liveness challenges (random head paths, lighting changes)

---

## ⚠️ Honest Limits

- **Generators move fast**: brand-new face swaps may evade the detector until re-trained; liveness challenges are the fallback.
- **Voice-clone detection is the weakest part** (see Results).
- **Poor video lowers detection power**: the dashboard shows lower confidence instead of guessing.

---

## 🏆 Hackathena '26 2.0

This project was developed as part of **Hackathena '26 2.0**, organized by the **Department of Computer Science & Engineering and CESA, Jyothi Engineering College**.

### Theme

> **Detection and Prevention of AI-Based Frauds**

The project addresses fraud enabled by generative AI, including **deepfakes, cloned voices, synthetic media and AI-assisted impersonation**.

---

## 📄 Repository Structure

```text
.
├── app.py                     # FastAPI server: dashboard + /ws streaming endpoint
├── web/                       # Dashboard (index.html, styles.css, app.js, capture-worker.js, pcm-worklet.js)
├── modules/                   # Detection engine
│   ├── live.py                # Per-call session orchestration
│   ├── face_module.py         # Face finder + face forensics
│   ├── voice_module.py        # Voice forensics
│   ├── avsync.py              # Lip-sync check
│   ├── challenge.py           # Liveness challenges
│   ├── fusion.py              # Real-time fusion and verdict
│   ├── scoring.py             # Liveness vs. synthesis aggregation
│   ├── calibration.py         # Detector output → risk
│   ├── evidence.py            # Learned fusion from labelled recordings
│   └── detectors/             # Face, voice, scene and custom detectors (deepguard: MIT, see NOTICE.md)
├── tools/                     # Evaluation, calibration, training and diagnostics CLIs
├── models/                    # Weights + calibration (git-ignored)
├── data/                      # Training / evidence data (git-ignored)
├── requirements.txt
├── TRAINING.md
└── README.md
```

---

## 📬 Contact

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
