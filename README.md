# Romani Audio Processing Handoff

This folder contains the scripts and notes needed to repeat the audio
processing workflow.

The goal of the workflow is to turn folders like this:

```text
OneDrive_1_5-15-2026/
  nahravka_17_respondent_29/
    nahravka_17_respondent_29.WAV
    text_nahravka_17_respondent_29.docx
```

into upload-ready folders like this:

```text
romani_final_upload/
  r17/
    segments.txt
    segments/
      20260328_r17_s0001.wav
      20260328_r17_s0002.wav
```

`segments.txt` uses this format:

```text
audio_filename.wav<TAB>transcript text
```

## Scripts

- `scripts/segment_transcript_aligned_single_speaker.py`
  - Main processing script.
  - Reads `.WAV` plus `.docx`.
  - Runs Whisper ASR.
  - Runs pyannote speaker diarization.
  - Aligns transcript text to audio.
  - Cuts accepted single-speaker WAV clips with `ffmpeg`.
  - Writes `segments_metadata.json`, `accepted_segments.ndjson`,
    `rejections.json`, `segments.tsv`, and accepted WAV clips.

- `scripts/run_segment_batch.sh`
  - Small wrapper around the main script.
  - Adds the `ffmpeg@7` library path workaround needed on this Mac.

- `scripts/collect_accepted_segments.py`
  - Copies only accepted WAV clips into upload layout.
  - Creates `rXX/segments/`.
  - Creates `rXX/segments.txt`.
  - By default, combines part folders like `r20a` and `r20b` into `r20`.

- `scripts/validate_upload_structure.py`
  - Checks the final upload layout.
  - Verifies every folder has `segments/` and `segments.txt`.
  - Verifies every line in `segments.txt` has a matching WAV file.

## Requirements

Use a Python environment with the audio-processing dependencies installed.
The environment used here was the repository `.venv`.

Important Python packages:

- `torch`
- `faster-whisper`
- `pyannote.audio`
- `whisperx`
- `python-docx`
- `rapidfuzz`
- `numpy`

System tools:

- `ffmpeg`
- `ffprobe`
- on this Mac, `ffmpeg@7` from Homebrew was needed for pyannote/torchcodec

The script also expects the required Hugging Face models to be available.
In our repo version, the main script sets offline Hugging Face mode, so cached
models are required unless the script is changed.

## Step 1: Run Segmentation

From this handoff folder:

```bash
cd /path/to/romani_audio_processing_handoff
```

Use the Python environment that has the dependencies. For example:

```bash
export PYTHON=/path/to/venv/bin/python
```

Run the whole batch:

```bash
scripts/run_segment_batch.sh /path/to/OneDrive_1_5-15-2026
```

Run only selected recordings:

```bash
scripts/run_segment_batch.sh /path/to/OneDrive_1_5-15-2026 r17 r19 r20a
```

The script creates output folders inside each recording folder:

```text
nahravka_17_respondent_29/
  segments_transcript_aligned_r17/
    accepted_segments.ndjson
    rejections.json
    segments_metadata.json
    segments.tsv
    20260328_r17_s0001.wav
```

## Step 2: Collect Accepted Clips

After segmentation finishes, collect only accepted clips into an upload-ready
folder:

```bash
scripts/collect_accepted_segments.py \
  --base-dir /path/to/OneDrive_1_5-15-2026 \
  --output-dir /path/to/romani_final_upload \
  --overwrite
```

This creates or updates folders like:

```text
romani_final_upload/
  r17/
    segments.txt
    segments/
      20260328_r17_s0001.wav
```

By default, part recordings such as `r20a` and `r20b` are combined into `r20`.
To keep them separate, add:

```bash
--keep-parts
```

## Step 3: Validate The Upload Folder

Run:

```bash
scripts/validate_upload_structure.py /path/to/romani_final_upload
```

Expected result:

```text
OK
```

If there are errors, the validator prints which recording folder has the
problem.

## Notes From Our Run

The May 15 batch used:

```bash
DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib:$DYLD_LIBRARY_PATH" \
PYTHONUNBUFFERED=1 \
.venv/bin/python segment_transcript_aligned_single_speaker.py \
  --base-dir /path/to/OneDrive_1_5-15-2026 \
  --out-prefix segments_transcript_aligned
```

`r17` first failed after ASR because pyannote/torchcodec could not load the
right FFmpeg libraries. We fixed that by using the `ffmpeg@7` library path and
reran `r17` with its existing ASR cache:

```bash
--recording r17 \
--reuse-asr-dir /path/to/OneDrive_1_5-15-2026/nahravka_17_respondent_29/segments_transcript_aligned_r17
```

Then we processed:

```bash
--recording r19 --recording r20a --recording r20b --recording r21 --recording r22
```

Final accepted clip counts from that batch:

```text
r17   35
r19   52
r20   114
r21   109
r22   109
```

`r20b` produced only 4 accepted clips because the transcript document did not
have a separate part marker that the script could detect, so both `r20a` and
`r20b` were aligned against the same `main` transcript section.
