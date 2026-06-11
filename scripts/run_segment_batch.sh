#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 BASE_DIR [REC_CODE ...]" >&2
  echo "Example: $0 /path/to/OneDrive_1_5-15-2026 r17 r19 r20a" >&2
  exit 2
fi

BASE_DIR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if command -v brew >/dev/null 2>&1 && brew --prefix ffmpeg@7 >/dev/null 2>&1; then
  export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib:${DYLD_LIBRARY_PATH:-}"
fi

CMD=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/segment_transcript_aligned_single_speaker.py"
  --base-dir "$BASE_DIR"
  --out-prefix segments_transcript_aligned
)

for REC_CODE in "$@"; do
  CMD+=(--recording "$REC_CODE")
done

PYTHONUNBUFFERED=1 "${CMD[@]}"
