from __future__ import annotations

import csv
import zlib
from pathlib import Path
from typing import List, Optional

import numpy as np

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".webm", ".3gp", ".amr"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _label(v: str) -> int:
    v = str(v).strip().lower()
    if v in ("1", "fake", "spoof", "deepfake", "synthetic"):
        return 1
    if v in ("0", "real", "bonafide", "bona-fide", "genuine"):
        return 0
    raise ValueError(f"unknown label {v!r} (use real/fake)")

def collect(real: Optional[List[str]], fake: Optional[List[str]], manifest: Optional[str], exts) -> List[dict]:
    items: List[dict] = []
    if manifest:
        root = Path(manifest).resolve().parent
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = Path(row["path"])
                p = p if p.is_absolute() else root / p
                items.append({"path": str(p), "label": _label(row["label"]),
                              "group": str(row.get("group") or p.stem),
                              "split": (row.get("split") or "").strip().lower(),
                              "tag": (row.get("tag") or "").strip()})
    for dirs, lab in ((real, 0), (fake, 1)):
        for d in dirs or []:
            d = Path(d)
            if not d.exists():
                raise SystemExit(f"folder not found: {d}")
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in exts and p.is_file():
                    rel = p.relative_to(d).parts
                    group = rel[0] if len(rel) > 1 else p.stem
                    items.append({"path": str(p), "label": lab, "group": group,
                                  "split": "", "tag": ""})
    missing = [it["path"] for it in items if not Path(it["path"]).exists()]
    if missing:
        print(f"warning: {len(missing)} files in the manifest don't exist, e.g. {missing[0]}")
        items = [it for it in items if Path(it["path"]).exists()]
    if not items:
        raise SystemExit("no input files found — check the folders / manifest")
    n_fake = sum(it["label"] for it in items)
    print(f"{len(items)} files: {len(items) - n_fake} real, {n_fake} fake, "
          f"{len({it['group'] for it in items})} groups")
    return items

def load_any_audio(path, max_seconds: float = 60.0):
    import shutil
    import subprocess
    import tempfile
    from modules.voice_module import load_audio
    path = Path(path)
    if path.suffix.lower() in (".wav", ".flac"):
        return load_audio(str(path), max_seconds=max_seconds)
    ff = shutil.which("ffmpeg")
    if ff is None:
        try:
            return load_audio(str(path), max_seconds=max_seconds)
        except Exception:
            raise RuntimeError(f"can't decode {path.suffix} without ffmpeg (winget install Gyan.FFmpeg, "
                               f"then open a new Command Prompt), or convert it to .wav")
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "16000",
                        "-t", str(max_seconds), str(wav)], check=True)
        return load_audio(str(wav), max_seconds=max_seconds)

def hashed(group: str, salt: str = "") -> int:
    return zlib.crc32(f"{salt}{group}".encode("utf-8")) % 100

def save_features(out: Path, rows: dict, layers, backbone: str, kind: str, channel: str = "none"):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez(tmp,
             X=np.asarray(rows["X"], np.float16) if rows["X"] else np.zeros((0, len(layers), 1), np.float16),
             y=np.asarray(rows["y"], np.int8), group=np.asarray(rows["group"], str),
             split=np.asarray(rows["split"], str), tag=np.asarray(rows["tag"], str),
             path=np.asarray(rows["path"], str), aug=np.asarray(rows["aug"], np.int8),
             layers=np.asarray(layers, np.int16), backbone=np.array(backbone), kind=np.array(kind),
             channel=np.array(channel))
    tmp.replace(out)

def load_partial(out: Path, channel: str = "none"):
    rows = {k: [] for k in ("X", "y", "group", "split", "tag", "path", "aug")}
    if not out.exists():
        return rows
    with np.load(out, allow_pickle=False) as z:
        old = str(z["channel"]) if "channel" in z.files else "none"
        if old != channel:
            raise SystemExit(f"{out} was extracted with --channel {old}; use a new --out for --channel {channel}")
        for k in rows:
            rows[k] = list(z[k])
    print(f"resuming: {len(set(rows['path']))} files already in {out}")
    return rows
