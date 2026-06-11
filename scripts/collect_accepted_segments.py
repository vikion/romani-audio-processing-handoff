#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def recording_folder_name(rec_code: str, collapse_parts: bool) -> str:
    if collapse_parts:
        match = re.fullmatch(r"(r\d+)[a-z]", rec_code)
        if match:
            return match.group(1)
    return rec_code


def rec_code_from_segment_dir(path: Path) -> str:
    match = re.search(r"segments_transcript_aligned_(r\d+[a-z]?)$", path.name)
    if not match:
        raise ValueError(f"Cannot infer recording code from {path}")
    return match.group(1)


def copy_segment(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect accepted aligned clips into rXX/segments plus segments.txt."
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Batch folder with nahravka_* directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination upload-style folder")
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep r20a/r20b as separate folders instead of combining into r20",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied WAVs and segments.txt")
    args = parser.parse_args()

    segment_dirs = sorted(args.base_dir.glob("nahravka_*/segments_transcript_aligned_r*"))
    if not segment_dirs:
        raise SystemExit(f"No segments_transcript_aligned_r* folders found under {args.base_dir}")

    grouped_lines: dict[str, list[str]] = {}
    copied = 0
    for segment_dir in segment_dirs:
        rec_code = rec_code_from_segment_dir(segment_dir)
        folder_name = recording_folder_name(rec_code, collapse_parts=not args.keep_parts)
        metadata_path = segment_dir / "segments_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing {metadata_path}")
        payload = load_json(metadata_path)
        rows = payload.get("rows", [])
        dest_dir = args.output_dir / folder_name
        dest_segments = dest_dir / "segments"
        grouped_lines.setdefault(folder_name, [])

        for row in rows:
            filename = str(row["file"])
            text = str(row.get("text", "")).strip()
            src_wav = segment_dir / filename
            if not src_wav.exists():
                raise FileNotFoundError(f"Missing WAV listed in metadata: {src_wav}")
            copy_segment(src_wav, dest_segments / filename, overwrite=args.overwrite)
            grouped_lines[folder_name].append(f"{filename}\t{text}")
            copied += 1

    for folder_name, lines in grouped_lines.items():
        dest_txt = args.output_dir / folder_name / "segments.txt"
        if dest_txt.exists() and not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dest_txt}")
        dest_txt.parent.mkdir(parents=True, exist_ok=True)
        dest_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Copied {copied} accepted clips into {args.output_dir}")
    for folder_name in sorted(grouped_lines):
        print(f"{folder_name}\t{len(grouped_lines[folder_name])}")


if __name__ == "__main__":
    main()
