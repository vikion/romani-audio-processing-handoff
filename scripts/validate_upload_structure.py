#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rXX/segments.txt plus rXX/segments/*.wav layout.")
    parser.add_argument("upload_dir", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    recording_dirs = sorted(p for p in args.upload_dir.glob("r*") if p.is_dir())
    if not recording_dirs:
        errors.append(f"No r* recording folders found in {args.upload_dir}")

    for rec_dir in recording_dirs:
        segments_dir = rec_dir / "segments"
        segments_txt = rec_dir / "segments.txt"
        if not segments_dir.is_dir():
            errors.append(f"{rec_dir.name}: missing segments/ directory")
            continue
        if not segments_txt.exists():
            errors.append(f"{rec_dir.name}: missing segments.txt")
            continue

        wavs = {p.name for p in segments_dir.glob("*.wav")}
        lines = [line for line in segments_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        listed: list[str] = []
        for index, line in enumerate(lines, start=1):
            if "\t" not in line:
                errors.append(f"{rec_dir.name}: line {index} has no tab delimiter")
                continue
            filename, text = line.split("\t", 1)
            filename = filename.strip()
            if not text.strip():
                errors.append(f"{rec_dir.name}: line {index} has empty transcript")
            listed.append(filename)
            if filename not in wavs:
                errors.append(f"{rec_dir.name}: line {index} lists missing WAV {filename}")

        extra = sorted(wavs - set(listed))
        if extra:
            errors.append(f"{rec_dir.name}: {len(extra)} WAV files are not listed in segments.txt")
        if len(lines) != len(wavs):
            errors.append(f"{rec_dir.name}: segments.txt lines={len(lines)} wavs={len(wavs)}")

        print(f"{rec_dir.name}\tsegments.txt={len(lines)}\twavs={len(wavs)}")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
