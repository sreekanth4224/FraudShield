"""
Step 1 (face): turn real / fake videos (or face images) into CLIP feature vectors.

    python -m tools.extract_face_features --real data\\face\\real --fake data\\face\\fake
    python -m tools.extract_face_features --manifest data\\face\\manifest.csv

Each video -> 12 frames spread over its first 30 s -> largest face -> the SAME
face crop the live engine sends to its detectors (box + 20 % margin) ->
frozen CLIP ViT-L/14 -> one 1024-number vector per crop for each of 5 layers.
Saved to features/face.npz; re-running resumes.

--degrade mix (default) passes half the frames through a simulated video call
(720p, JPEG q70) so the head learns fake-vs-real, not sharp-vs-compressed.
Put each person's videos in their own sub-folder (real/person07/...) or give a
manifest with a group column, so test people are never seen in training.

First run downloads CLIP ViT-L/14 (~1.7 GB) into models/hf-cache.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from modules.detectors.custom_heads import FACE_BACKBONE, FACE_LAYERS, FaceBackbone
from modules.detectors.face_classifier import face_crop
from tools.custom_common import IMAGE_EXT, VIDEO_EXT, collect, load_partial, save_features
from tools.evaluate import DEGRADE

MIN_FACE = 48

class FaceFinder:

    def __init__(self):
        self.mp = None
        try:
            import mediapipe as mp
            self.mp = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self.name = "MediaPipe BlazeFace"
        except Exception:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.name = "OpenCV Haar (install mediapipe==0.10.14 for better crops)"

    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        boxes = []
        if self.mp is not None:
            res = self.mp.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                b = d.location_data.relative_bounding_box
                boxes.append((b.xmin * W, b.ymin * H, b.width * W, b.height * H))
        else:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            boxes = [tuple(map(float, b)) for b in self.haar.detectMultiScale(g, 1.1, 5, minSize=(MIN_FACE, MIN_FACE))]
        boxes = [b for b in boxes if min(b[2], b[3]) >= MIN_FACE]
        return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None

def frames_of(path: str, n: int, max_s: float = 30.0):
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(path)
        return [img] if img is not None else []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last = min(total, int(max_s * fps)) if total > 0 else int(max_s * fps)
    first = min(int(0.5 * fps), max(0, last - 1))
    want = set(np.linspace(first, max(first, last - 1), n).astype(int).tolist())
    out, i = [], 0
    while i <= max(want):
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out.append(frame)
        i += 1
    cap.release()
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="*", help="folders of genuine videos / face images")
    ap.add_argument("--fake", nargs="*", help="folders of deepfake videos / face images")
    ap.add_argument("--manifest", help="CSV with path,label,group[,split][,tag]")
    ap.add_argument("--out", default="features/face.npz")
    ap.add_argument("--frames", type=int, default=12, help="frames sampled per video")
    ap.add_argument("--degrade", choices=["none", "call", "call-low", "mix"], default="mix")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files per label (quick test)")
    ap.add_argument("--backbone", default=FACE_BACKBONE)
    args = ap.parse_args()

    items = collect(args.real, args.fake, args.manifest, VIDEO_EXT | IMAGE_EXT)
    if args.limit:
        items = [it for lab in (0, 1) for it in [i for i in items if i["label"] == lab][: args.limit]]
    out = Path(args.out)
    rows = load_partial(out)
    done = set(rows["path"])
    todo = [it for it in items if it["path"] not in done]
    print(f"{len(todo)} files to process -> {out}")

    finder = FaceFinder()
    bb = FaceBackbone(args.backbone, max_layer=max(FACE_LAYERS))
    print(f"face finder: {finder.name}; backbone {args.backbone} on {bb.device}")
    rng = np.random.default_rng(0)
    t0, no_face = time.time(), 0
    for i, it in enumerate(todo, 1):
        crops, augs = [], []
        try:
            for frame in frames_of(it["path"], args.frames):
                mode = args.degrade
                if mode == "mix":
                    mode = "none" if rng.random() < 0.5 else ("call" if rng.random() < 0.7 else "call-low")
                deg = DEGRADE.get(mode)
                f = deg(frame) if deg is not None else frame
                box = finder(f)
                crop = face_crop(f, box) if box is not None else None
                if crop is not None:
                    crops.append(crop)
                    augs.append(0 if mode == "none" else 1)
        except Exception as e:
            print(f"  skip {it['path']}: {e}")
        if not crops:
            no_face += 1
            continue
        X = bb.features(crops, FACE_LAYERS)
        for x, aug in zip(X, augs):
            rows["X"].append(x)
            rows["y"].append(it["label"])
            rows["aug"].append(aug)
            for k in ("group", "split", "tag", "path"):
                rows[k].append(it[k])
        if i % 25 == 0 or i == len(todo):
            save_features(out, rows, FACE_LAYERS, args.backbone, "face")
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)} files  ({rate:.2f}/s, ~{(len(todo) - i) / rate / 60:.0f} min left)")
    save_features(out, rows, FACE_LAYERS, args.backbone, "face")
    print(f"done: {len(rows['y'])} face crops from {len(set(rows['path']))} files "
          f"({no_face} files without a usable face) -> {out}")

if __name__ == "__main__":
    main()
