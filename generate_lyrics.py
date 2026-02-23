#!/usr/bin/env python3
"""
Auto-generate Premiere-friendly captions (SRT/VTT) from audio using faster-whisper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import timedelta
from typing import List, Dict, Any, Tuple
from pathlib import Path

import srt
from faster_whisper import WhisperModel


PUNCT_END_RE = re.compile(r"[.!?…]+$")
PUNCT_BREAK_RE = re.compile(r"[,;:—–-]+$")


def ts_to_srt_time(seconds: float) -> timedelta:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    return timedelta(milliseconds=int(round(seconds * 1000)))


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    return text


def split_into_caption_blocks(
    words: List[Dict[str, Any]],
    max_chars: int,
    max_words: int,
    max_pause: float,
    prefer_punct: bool = True,
) -> List[Tuple[float, float, str, List[Dict[str, Any]]]]:

    blocks = []
    cur: List[Dict[str, Any]] = []
    cur_chars = 0

    def flush():
        nonlocal cur, cur_chars
        if not cur:
            return
        start = cur[0]["start"]
        end = cur[-1]["end"]
        text = normalize_spaces(" ".join(w["word"] for w in cur))
        blocks.append((start, end, text, cur.copy()))
        cur = []
        cur_chars = 0

    for w in words:
        token = w["word"]
        token_len = len(token) + (1 if cur else 0)

        pause = 0.0
        if cur:
            prev_end = cur[-1]["end"]
            pause = max(0.0, (w["start"] or 0.0) - (prev_end or 0.0))

        prev_word = cur[-1]["word"] if cur else ""
        prev_is_end = bool(PUNCT_END_RE.search(prev_word))
        prev_is_break = bool(PUNCT_BREAK_RE.search(prev_word))

        if cur and pause >= max_pause:
            flush()

        if cur and (
            (cur_chars + token_len) > max_chars
            or (len(cur) >= max_words)
        ):
            flush()

        cur.append(w)
        cur_chars += token_len

        if prefer_punct and cur:
            if prev_is_end:
                flush()
            elif prev_is_break and (cur_chars >= int(max_chars * 0.7)):
                flush()

    flush()
    return blocks


def blocks_to_srt(blocks):
    subs = []
    for idx, (start, end, text, _) in enumerate(blocks, start=1):
        subs.append(
            srt.Subtitle(
                index=idx,
                start=ts_to_srt_time(start),
                end=ts_to_srt_time(end),
                content=text,
            )
        )
    return srt.compose(subs)


def srt_to_vtt(srt_text: str) -> str:
    vtt = "WEBVTT\n\n"
    lines = srt_text.splitlines()
    out_lines = []
    for line in lines:
        if re.fullmatch(r"\d+", line.strip()):
            continue
        out_lines.append(line.replace(",", "."))
    vtt += "\n".join(out_lines).strip() + "\n"
    return vtt


def process_file(input_path: Path, model: WhisperModel, args):

    segments, info = model.transcribe(
        str(input_path),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        word_timestamps=True,
    )

    all_words = []
    for seg in segments:
        if getattr(seg, "words", None):
            for ww in seg.words:
                all_words.append(
                    {"word": ww.word.strip(), "start": ww.start, "end": ww.end}
                )
        else:
            all_words.append(
                {"word": seg.text.strip(), "start": seg.start, "end": seg.end}
            )

    for w in all_words:
        w["word"] = w["word"].replace("\u200b", "").strip()

    blocks = split_into_caption_blocks(
        all_words,
        max_chars=args.max_chars,
        max_words=args.max_words,
        max_pause=args.max_pause,
        prefer_punct=not args.no_prefer_punct,
    )

    srt_text = blocks_to_srt(blocks)

    out_path = input_path.with_suffix(".srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    if args.vtt:
        with open(input_path.with_suffix(".vtt"), "w", encoding="utf-8") as f:
            f.write(srt_to_vtt(srt_text))

    if args.json:
        payload = {
            "input": str(input_path),
            "model": args.model,
            "language": info.language,
            "duration": info.duration,
            "words": all_words,
            "blocks": [
                {"start": s, "end": e, "text": t}
                for (s, e, t, _) in blocks
            ],
        }
        with open(input_path.with_suffix(".words.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="Audio/video file path")
    ap.add_argument("--folder", help="Process all audio/video files in folder")
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute_type", default="auto")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--max_chars", type=int, default=42)
    ap.add_argument("--max_words", type=int, default=10)
    ap.add_argument("--max_pause", type=float, default=0.65)
    ap.add_argument("--no_prefer_punct", action="store_true")
    ap.add_argument("--vtt", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.input and not args.folder:
        raise SystemExit("Provide either a file path or --folder")

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            raise SystemExit(f"Folder not found: {folder}")

        audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".mp4", ".mov"}
        files = [f for f in folder.iterdir() if f.suffix.lower() in audio_exts]

        for f in files:
            print(f"Processing: {f.name}")
            process_file(f, model, args)

    else:
        process_file(Path(args.input), model, args)


if __name__ == "__main__":
    main()