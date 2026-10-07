"""
Download a SAMPLE of speech clips from a Hugging Face dataset into data/voice/<real|fake>/<speaker>/
and append them to data/voice/manifest.csv (used by tools.extract_voice_features --manifest).

    # fake Indian-language speech (IndicSynth is synthetic only)
    python -m tools.fetch_hf_audio --repo vdivyasharma/IndicSynth --label fake --filter hindi --n 600 --tag hindi

    # real Indian-language speech, e.g. FLEURS (auto-converted parquet branch)
    python -m tools.fetch_hf_audio --repo google/fleurs --revision refs/convert/parquet --label real --filter hi_in --n 600 --tag hindi

    # just look at what a repo contains
    python -m tools.fetch_hf_audio --repo vdivyasharma/IndicSynth --list

Handles the two common layouts:
  * plain audio files (.wav/.flac/.mp3) -> a random sample is downloaded one by one;
    a metadata .csv/.tsv next to them is used for speaker ids if it has a speaker column
  * parquet shards with an audio column -> shards are read until --n clips are written
    (needs: pip install pyarrow)

Gated datasets: accept the terms on the dataset page, then run `hf auth login`
(older huggingface_hub: `huggingface-cli login`) once.
If a repo uses another layout the script prints what it found; then download it
by hand and point extract_voice_features at the folders instead.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
from pathlib import Path

import numpy as np

AUDIO = (".wav", ".flac", ".mp3", ".ogg", ".opus")
SPK_COLS = ("speaker_id", "speaker", "spk_id", "spk", "client_id", "reader_id", "speakerid")

def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:60] or "unknown"

def speaker_map(api_dl, meta_files):
    out = {}
    for mf in meta_files:
        try:
            path = api_dl(mf)
            delim = "\t" if mf.endswith(".tsv") else ","
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f, delimiter=delim))
        except Exception as e:
            print(f"  (could not read {mf}: {e})")
            continue
        if not rows:
            continue
        cols = {c.lower().strip(): c for c in rows[0].keys() if c}
        spk = next((cols[c] for c in SPK_COLS if c in cols), None)
        fcol = next((cols[c] for c in ("file_name", "filename", "file", "path", "audio", "audio_path", "id", "utt_id")
                     if c in cols), None)
        if spk and fcol:
            for r in rows:
                out[Path(str(r[fcol])).stem] = r[spk]
            print(f"  speaker ids from {mf} (column '{spk}'): {len(out)} entries")
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None, help="e.g. refs/convert/parquet")
    ap.add_argument("--label", choices=["real", "fake"])
    ap.add_argument("--filter", default="", help="only files whose path contains this (case-insensitive)")
    ap.add_argument("--n", type=int, default=500, help="clips to fetch")
    ap.add_argument("--tag", default="", help="language: prefixes the speaker group (e.g. hindi)")
    ap.add_argument("--source", default="", help="dataset name for the per-source report (e.g. fleurs, kathbath)")
    ap.add_argument("--out", default="data/voice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=4,
                    help="parquet datasets without a train/test split: how many shards to sample from")
    ap.add_argument("--speaker-col", default=None, help="column holding the speaker id (auto-detected)")
    ap.add_argument("--audio-col", default=None,
                    help="audio column to take (default: 'audio'); e.g. IndicSynth's real reference recordings: "
                         "--audio-col \"Target Reference Audio\" --label real")
    ap.add_argument("--list", action="store_true", help="only print the repo layout")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(args.repo, repo_type="dataset", revision=args.revision)
    flt = args.filter.lower()
    sel = [f for f in files if flt in f.lower()]
    exts = {}
    for f in sel:
        exts[Path(f).suffix.lower()] = exts.get(Path(f).suffix.lower(), 0) + 1
    print(f"{args.repo}: {len(files)} files, {len(sel)} match '{args.filter}': "
          + ", ".join(f"{k or '(none)'}×{v}" for k, v in sorted(exts.items(), key=lambda kv: -kv[1])))
    if args.list or not args.label:
        for f in sel[:40]:
            print("  ", f)
        return

    dl = lambda f: hf_hub_download(args.repo, f, repo_type="dataset", revision=args.revision)
    out = Path(args.out)
    man = out / "manifest.csv"
    new_manifest = not man.exists()
    rng = random.Random(args.seed)
    written = []

    audio = [f for f in sel if f.lower().endswith(AUDIO)]
    parquet = [f for f in sel if f.lower().endswith(".parquet")]
    if audio:
        metas = [f for f in sel if f.lower().endswith((".csv", ".tsv")) and "meta" in f.lower()][:5]
        spk = speaker_map(dl, metas)
        rng.shuffle(audio)
        for f in audio[: args.n]:
            src = Path(dl(f))
            group = spk.get(src.stem) or Path(f).parent.name
            dst = out / args.label / safe(group) / f"{safe(Path(f).parent.name)}_{src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            written.append((dst, group, ""))
            if len(written) % 50 == 0:
                print(f"  {len(written)}/{args.n}")
    elif parquet:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit("this repo stores audio in parquet files: run  pip install pyarrow  and try again")
        import soundfile as sf
        split_of = lambda f: ("test" if re.search(r"(test|valid|dev)", f.lower())
                              else ("train" if "train" in f.lower() else ""))
        by = {sp: [f for f in parquet if split_of(f) == sp] for sp in ("train", "test", "")}
        official = bool(by["train"] and by["test"])
        quota = {}
        if official:
            for sp, share in (("train", 0.75), ("test", 0.25)):
                for f in by[sp]:
                    quota[f] = int(round(args.n * share / len(by[sp])))
        else:
            chosen = rng.sample(parquet, min(len(parquet), max(1, args.shards)))
            for f in chosen:
                quota[f] = -(-args.n // len(chosen))
            print(f"  sampling {len(chosen)} of {len(parquet)} shards (--shards to change)")
        shown = False
        warned_type = False
        seen = set()
        for shard in [f for f in parquet if quota.get(f, 0) > 0]:
            want = quota.get(shard, 0)
            if want <= 0:
                continue
            split = split_of(shard) if official else ""
            pf = pq.ParquetFile(dl(shard))
            cols = pf.schema_arrow.names
            acol = args.audio_col or next((c for c in cols if c.lower() in ("audio", "speech", "wav")), None)
            if acol is not None and acol not in cols:
                raise SystemExit(f"--audio-col {acol!r} not in columns {cols}")
            if acol is None:
                print(f"  no audio column in {shard} (columns: {cols})")
                continue
            scol = args.speaker_col or next((c for c in cols if c.lower() in SPK_COLS), None)
            if scol is not None and scol not in cols:
                raise SystemExit(f"--speaker-col {scol!r} not in columns {cols}")
            if not shown:
                print(f"  columns: {cols}  -> audio '{acol}' ({pf.schema_arrow.field(acol).type}), "
                      f"speaker '{scol or 'none found'}'")
                shown = True
            total = pf.metadata.num_rows
            if args.audio_col:
                pick = set(range(total))
            else:
                pick = set(rng.sample(range(total), min(total, int(want * 1.2) + 5)))
            got, i = 0, 0
            for batch in pf.iter_batches(batch_size=64, columns=[acol] + ([scol] if scol else [])):
                for r in batch.to_pylist():
                    idx, i = i, i + 1
                    if idx not in pick or got >= want:
                        continue
                    a = r[acol]
                    data = a.get("bytes") if isinstance(a, dict) else (a if isinstance(a, (bytes, bytearray)) else None)
                    if not data:
                        if not warned_type:
                            print(f"  '{acol}' holds {type(a).__name__} values, not audio bytes, e.g. {str(a)[:80]!r}")
                            warned_type = True
                        continue
                    h = hash(bytes(data[:4096])) ^ len(data)
                    if h in seen:
                        continue
                    seen.add(h)
                    try:
                        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                    except Exception:
                        continue
                    group = (str(r[scol]) if scol and r.get(scol) is not None
                             else f"{Path(shard).parent.name}-{split or 'x'}-{got // 25}")
                    dst = out / args.label / safe(group) / f"{safe(Path(shard).parent.name)}_{idx}.wav"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(dst, y.mean(1), sr)
                    written.append((dst, group, "" if scol else split))
                    got += 1
                    if len(written) >= args.n:
                        break
                if got >= want or len(written) >= args.n:
                    break
            print(f"  {shard} ({split or 'no split'}): {got} clips, total {len(written)}")
    else:
        raise SystemExit("no audio or parquet files matched — run with --list to see the layout, "
                         "or download by hand into data/voice/<real|fake>/<speaker>/")

    if not written:
        raise SystemExit("no clips written — see the messages above")
    out.mkdir(parents=True, exist_ok=True)
    with open(man, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_manifest:
            w.writerow(["path", "label", "group", "split", "tag"])
        for dst, group, split in written:
            w.writerow([dst.relative_to(out).as_posix(), args.label, f"{args.tag}:{group}" if args.tag else group,
                        split, args.source or args.tag])
    groups = {g for _, g, _ in written}
    print(f"wrote {len(written)} {args.label} clips from {len(groups)} speakers -> {out / args.label}, "
          f"listed in {man}")
    if len(groups) < 5:
        print("warning: very few speaker ids — the train/test split will be coarse "
              "(check the metadata, or add a 'split' column by hand)")

if __name__ == "__main__":
    main()
