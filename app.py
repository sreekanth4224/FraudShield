import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import argparse
import asyncio
import json
import struct
import sys
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from modules.detectors.face_classifier import FaceDeepfakeClassifier
from modules.detectors.voice_classifier import VoiceDeepfakeClassifier
from modules.live import LiveSession

ROOT = Path(__file__).parent
WEB = ROOT / "web"
ANALYSIS_PERIOD_S = 0.25

HEADER = struct.Struct("<BxxxId")
KIND_FRAME, KIND_AUDIO = 1, 2
REGION = struct.Struct("<HHHH")

cv2.setNumThreads(2)

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

def origin_allowed(origin) -> bool:
    if origin is None:
        return True
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and (parts.hostname or "") in ALLOWED_HOSTS

@asynccontextmanager
async def lifespan(_app):
    from modules.challenge import Transcriber
    from modules.detectors.custom_heads import CustomFaceDetector, CustomVoiceDetector
    from modules.detectors.scene_classifier import SceneDeepfakeClassifier
    for cls in (FaceDeepfakeClassifier, VoiceDeepfakeClassifier, SceneDeepfakeClassifier, Transcriber,
                CustomFaceDetector, CustomVoiceDetector):
        if cls.status == "not loaded":
            cls.status = "queued"
    threading.Thread(target=_load_models_in_order, name="fs-model-load", daemon=True).start()
    yield

def _load_models_in_order():
    try:
        import transformers
        from transformers import (AutoConfig, AutoModel, AutoModelForImageClassification,
                                  ViTForImageClassification, Wav2Vec2FeatureExtractor,
                                  Wav2Vec2ForSequenceClassification, pipeline)
    except Exception as e:
        print(f"  transformers import problem: {e}")
    from modules.challenge import Transcriber
    from modules.detectors.custom_heads import CustomFaceDetector, CustomVoiceDetector
    from modules.detectors.scene_classifier import SceneDeepfakeClassifier
    for cls in (FaceDeepfakeClassifier, VoiceDeepfakeClassifier, SceneDeepfakeClassifier,
                Transcriber, CustomFaceDetector, CustomVoiceDetector):
        try:
            cls.get()
        except Exception as e:
            print(f"  {cls.__name__} failed to load: {e}")
    print("  models: face " + FaceDeepfakeClassifier.status + " · voice " + VoiceDeepfakeClassifier.status
          + " · AI-video " + SceneDeepfakeClassifier.status + " · speech-to-text " + Transcriber.status)

app = FastAPI(title="FraudShield Live", lifespan=lifespan)

@app.middleware("http")
async def no_stale_dashboard(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response

@app.get("/api/health")
def health():
    return {"ok": True, "face_model": FaceDeepfakeClassifier.status, "voice_model": VoiceDeepfakeClassifier.status}

@app.websocket("/ws")
async def session_socket(ws: WebSocket):
    if not origin_allowed(ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    await ws.accept()
    loop = asyncio.get_running_loop()
    frame_pool = ThreadPoolExecutor(1, thread_name_prefix="fs-frame")
    analysis_pool = ThreadPoolExecutor(1, thread_name_prefix="fs-analysis")
    session = await loop.run_in_executor(frame_pool, LiveSession)
    outbox: asyncio.Queue = asyncio.Queue(maxsize=64)
    frame_busy = False

    async def sender():
        while True:
            msg = await outbox.get()
            await ws.send_text(json.dumps(msg, separators=(",", ":")))

    async def analyzer():
        while True:
            await asyncio.sleep(ANALYSIS_PERIOD_S)
            state = await loop.run_in_executor(analysis_pool, session.analyze)
            if state:
                await outbox.put(state)

    async def handle_frame(ts, jpeg, origin, full_size):
        nonlocal frame_busy
        try:
            ack = await loop.run_in_executor(frame_pool, session.process_frame, ts, jpeg, origin, full_size)
            await outbox.put(ack)
        except Exception as e:
            await outbox.put({"type": "ack", "error": str(e)})
        finally:
            frame_busy = False

    tasks = [asyncio.create_task(sender()), asyncio.create_task(analyzer())]
    await outbox.put({"type": "hello", "backend": session.face.backend, "challenge": session.challenge})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                if len(data) < HEADER.size:
                    continue
                kind, extra, ts = HEADER.unpack_from(data)
                payload = data[HEADER.size:]
                if kind == KIND_FRAME:
                    if frame_busy:
                        await outbox.put({"type": "ack", "dropped": True})
                        continue
                    origin, full_size = (0, 0), None
                    if extra & 1 and len(payload) > REGION.size:
                        x, y, w, h = REGION.unpack_from(payload)
                        origin, full_size, payload = (x, y), (w, h), payload[REGION.size:]
                    frame_busy = True
                    asyncio.create_task(handle_frame(ts, payload, origin, full_size))
                elif kind == KIND_AUDIO:
                    session.add_audio(ts, extra, payload)
                continue
            text = msg.get("text")
            if not text:
                continue
            m = json.loads(text)
            if m.get("type") == "config":
                await loop.run_in_executor(frame_pool, session.configure, m)
            elif m.get("type") == "challenge":
                if m.get("action") == "start":
                    st = await loop.run_in_executor(frame_pool, session.start_challenge,
                                                    str(m.get("kind") or "phrase"), (m.get("phrase") or "")[:120])
                    await outbox.put({"type": "challenge", "challenge": st})
                else:
                    session.cancel_challenge()
                    await outbox.put({"type": "challenge", "challenge": None})
            elif m.get("type") == "reset":
                await loop.run_in_executor(frame_pool, session.reset)
                await outbox.put({"type": "hello", "backend": session.face.backend, "challenge": session.challenge})
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
        frame_pool.submit(session.close)
        frame_pool.shutdown(wait=False)
        analysis_pool.shutdown(wait=False, cancel_futures=True)

app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

def _browser_exe():
    if os.name != "nt":
        return None
    import shutil
    bases = [os.environ.get(k, "") for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    rel = [r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe",
           r"BraveSoftware\Brave-Browser\Application\brave.exe"]
    for r in rel:
        for b in bases:
            p = os.path.join(b, r)
            if b and os.path.exists(p):
                return p
    return shutil.which("chrome") or shutil.which("msedge")

def _work_area():
    try:
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0)
        return r.left, r.top, r.right - r.left, r.bottom - r.top
    except Exception:
        return None

def open_dashboard(url, side="right"):
    import subprocess
    exe = _browser_exe() if side != "full" else None
    area = _work_area() if exe else None
    if exe and area:
        x, y, w, h = area
        half = w // 2
        left = x + (half if side == "right" else 0)
        try:
            subprocess.Popen([exe, f"--app={url}", f"--window-size={half},{h}", f"--window-position={left},{y}"])
            return
        except Exception as e:
            print(f"  could not open the half-screen window ({e}) — opening a normal tab")
    webbrowser.open(url)

if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="FraudShield Live dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--window", choices=["right", "left", "full"], default=os.environ.get("FRAUDSHIELD_WINDOW", "right"),
                    help="open the dashboard as a half-screen app window on the right (default) / left, or a normal tab")
    args = ap.parse_args()
    url = f"http://localhost:{args.port}"
    print(f"\n  FraudShield Live  →  {url}\n  (use Chrome or Edge; screen capture needs localhost or HTTPS)\n")
    if not args.no_browser:
        threading.Timer(1.2, open_dashboard, args=(url, args.window)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
