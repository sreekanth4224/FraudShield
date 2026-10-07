"""
Check the spoken-phrase challenge's speech-to-text on a recording, before using it live.

    python -m tools.test_phrase "C:\\path\\clip.mp4" --phrase "garden window 7 3 9 1"

Record yourself (Windows Camera app) reading a phrase, then run this. It prints what Whisper
heard, the word match, and the verdict the live challenge would give (without the lip-sync part).
The first run downloads Whisper (small ~1 GB on a GPU / base ~300 MB on CPU).
"""

from __future__ import annotations

import argparse
import sys
import time

from modules.challenge import Transcriber, judge, match_score, normalize
from modules.voice_module import _resample
from tools.custom_common import load_any_audio

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help='video / audio file, or "latest" = newest Windows Camera clip')
    ap.add_argument("--phrase", required=True, help='e.g. "garden window 7 3 9 1"')
    args = ap.parse_args()
    if args.file.lower() == "latest":
        import glob
        import os
        roll = os.path.expanduser(r"~\OneDrive\Pictures\Camera Roll")
        clips = sorted(glob.glob(os.path.join(roll, "*.mp4")) +
                       glob.glob(os.path.expanduser(r"~\Pictures\Camera Roll\*.mp4")), key=os.path.getmtime)
        if not clips:
            raise SystemExit(f"no .mp4 found in {roll}")
        args.file = clips[-1]
        print(f"using newest Camera clip: {os.path.basename(args.file)}")
    asr = Transcriber.get()
    if asr is None:
        raise SystemExit(f"speech-to-text not available: {Transcriber.status}")
    print(f"model: {asr.name}")
    y, sr = load_any_audio(args.file, 20.0)
    t0 = time.time()
    text = asr(_resample(y, sr))
    words, missed = match_score(normalize(args.phrase), normalize(text))
    risk, verdict, msg = judge(words, None, len(y) / sr, None)
    print(f"heard:   “{text}”   ({time.time() - t0:.1f} s)")
    print(f"phrase:  {args.phrase}")
    print(f"match:   {100 * words:.0f}% of words" + (f"   missed: {', '.join(missed)}" if missed else ""))
    print(f"verdict: {verdict.upper()} — {msg}")

if __name__ == "__main__":
    main()
