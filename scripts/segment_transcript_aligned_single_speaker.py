#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import torch
from docx import Document
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from rapidfuzz import fuzz

try:
    import whisperx
    from whisperx.audio import load_audio as whisperx_load_audio
except ImportError:
    whisperx = None
    whisperx_load_audio = None


AUDIO_SAMPLE_RATE = 16000
WORD_RE = re.compile(r"[A-Za-zÀ-ž]+(?:['’][A-Za-zÀ-ž]+)?")
PART_MARKER_RE = re.compile(r"^\s*Nahr[aá]vka\s+č\.?\s*(\d+)\.?:?\s*$", re.IGNORECASE)
PAREN_GROUP_RE = re.compile(r"(?<!\w)\([^)]{1,80}\)(?!\w)")
ELLIPSIS_ONLY_RE = re.compile(r"^[.\s…]+$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


COMMON_STOPWORDS = {
    "a",
    "aj",
    "ale",
    "ano",
    "asi",
    "ako",
    "alebo",
    "ani",
    "ano",
    "abo",
    "bo",
    "by",
    "co",
    "čo",
    "da",
    "de",
    "do",
    "hej",
    "hi",
    "hin",
    "hoj",
    "i",
    "ich",
    "im",
    "ja",
    "je",
    "jej",
    "jemu",
    "k",
    "kaj",
    "ked",
    "ke",
    "keď",
    "len",
    "ma",
    "me",
    "mi",
    "mne",
    "na",
    "ne",
    "nie",
    "no",
    "o",
    "oda",
    "on",
    "ona",
    "oni",
    "po",
    "pre",
    "pro",
    "sa",
    "sar",
    "si",
    "so",
    "som",
    "ta",
    "tak",
    "te",
    "to",
    "tu",
    "u",
    "um",
    "uhm",
    "už",
    "uz",
    "vam",
    "vám",
    "vareso",
    "vase",
    "vás",
    "vas",
    "však",
    "ze",
    "že",
}


_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
warnings.filterwarnings("ignore")


@dataclass
class TranscriptSentence:
    sentence_id: str
    turn_id: str
    order: int
    text: str
    norm: str
    tokens: List[str]
    anchors: List[str]
    low_trust: bool


@dataclass
class TranscriptTurn:
    turn_id: str
    line_no: int
    text: str
    norm: str
    tokens: List[str]
    anchors: List[str]
    low_trust: bool
    sentences: List[TranscriptSentence] = field(default_factory=list)


@dataclass
class TranscriptSection:
    name: str
    turns: List[TranscriptTurn]


@dataclass
class TranscriptDoc:
    docx_path: Path
    verbatim_until_s: Optional[float]
    global_low_trust: bool
    preface_notes: List[str]
    sections: Dict[str, TranscriptSection]


@dataclass
class AsrWord:
    start: float
    end: float
    text: str
    norm: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class AsrSentence:
    idx: int
    start: float
    end: float
    text: str
    norm: str
    tokens: List[str]
    anchors: List[str]
    word_start: int
    word_end: int


@dataclass
class DiarTurn:
    start: float
    end: float
    speaker: str


@dataclass
class MatchWindow:
    start_idx: int
    end_idx: int
    score: float
    common_anchors: int
    text: str


@dataclass
class TurnCandidate:
    match: MatchWindow
    value: float


@dataclass
class CandidateSegment:
    sentence: TranscriptSentence
    start: float
    end: float
    alignment_score: float
    matched_asr_text: str
    used_proportional_fallback: bool = False


@dataclass
class RecordingJob:
    rec_code: str
    wav_path: Path
    docx_path: Path
    section_name: str
    out_dir: Path
    verbatim_until_s: Optional[float]
    global_low_trust: bool
    section_note: Optional[str] = None


@dataclass
class ForceAlignResources:
    model: Any
    metadata: Dict[str, Any]
    device: str
    language_code: str


@dataclass
class CandidateValidation:
    ok: bool
    reason: Optional[str]
    start: float
    end: float
    force_core_start: float
    force_core_end: float
    clip_asr_text: str
    clip_asr_score: float
    text_rate: float
    force_word_count: int
    force_mean_score: float
    force_median_score: float
    force_max_score: float
    force_ge20_ratio: float
    force_anchor_count: int
    force_anchor_hit_count: int
    force_anchor_hit_ratio: float
    force_anchor_mean: float
    force_anchor_max: float
    force_keep_ratio: float


@dataclass
class SpeakerMetrics:
    speaker: str
    purity: float
    overlap_ratio: float
    covered_purity: float
    coverage_ratio: float


@dataclass
class AsrCachePaths:
    sentences: Path
    words: Path


def wav_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def recording_part_number(stem: str) -> Optional[int]:
    if "cast_1" in stem or "cast 1" in stem:
        return 1
    if "cast_2" in stem or "cast 2" in stem:
        return 2
    if "cast_3" in stem or "cast 3" in stem:
        return 3
    return None


def recording_part_suffix(stem: str) -> str:
    part = recording_part_number(stem)
    if part is None:
        return ""
    return chr(ord("a") + part - 1)


def recording_number(folder: Path, wav: Path) -> Optional[int]:
    for text in (wav.stem, folder.name):
        match = re.search(r"nahravka[_ ](\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def docx_recording_number(docx_path: Path) -> Optional[int]:
    match = re.search(r"nahravka[_ ](\d+)", docx_path.stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def choose_docx_for_wav(docxs: Sequence[Path], rec_num: Optional[int]) -> Path:
    if len(docxs) == 1 or rec_num is None:
        return docxs[0]
    matching = [path for path in docxs if docx_recording_number(path) == rec_num]
    if matching:
        return matching[0]
    return docxs[0]


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def clean_transcript_text(text: str) -> str:
    text = normalize_spaces(text)
    text = text.replace("_", " ")
    text = PAREN_GROUP_RE.sub(" ", text)
    text = normalize_spaces(text)
    text = re.sub(r"^\s*[.…]{2,}\s*", "", text)
    text = re.sub(r"\s*[.…]{2,}\s*$", "", text)
    return normalize_spaces(text.strip(" -"))


def normalize_for_match(text: str) -> str:
    text = clean_transcript_text(text)
    text = strip_accents(text.lower().replace("’", "'"))
    tokens = WORD_RE.findall(text)
    return " ".join(tokens)


def tokenize_norm(text: str) -> List[str]:
    return normalize_for_match(text).split()


def anchor_tokens(tokens: Sequence[str]) -> List[str]:
    return [tok for tok in tokens if len(tok) >= 4 and tok not in COMMON_STOPWORDS]


def split_into_sentences(text: str) -> List[str]:
    parts = [normalize_spaces(p) for p in SENTENCE_SPLIT_RE.split(text) if normalize_spaces(p)]
    return parts or [normalize_spaces(text)]


def find_hf_snapshot(repo_id: str, required_files: Optional[Sequence[str]] = None) -> Path:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not cache_root.exists():
        raise FileNotFoundError(f"Missing cached model snapshot for {repo_id}: {cache_root}")
    snapshots = sorted([p for p in cache_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshots found for {repo_id}: {cache_root}")
    if required_files:
        for snapshot in snapshots:
            if all((snapshot / filename).exists() for filename in required_files):
                return snapshot
        raise FileNotFoundError(f"No cached snapshots for {repo_id} contained required files: {required_files}")
    return snapshots[0]


def ffmpeg_cut(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{max(0.01, end - start):.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def ffmpeg_excerpt(src: Path, dst: Path, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-t",
        f"{max(0.01, end):.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def write_json(path: Path, payload: Dict | List) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_ndjson(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_progress(
    out_dir: Path,
    *,
    stage: str,
    recording: Path,
    docx: Path,
    section_name: str,
    accepted: int,
    rejected: int,
    total_turns: int,
    processed_turns: int,
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        "stage": stage,
        "recording": str(recording),
        "docx": str(docx),
        "transcript_section": section_name,
        "accepted_segments": accepted,
        "rejected_segments": rejected,
        "processed_turns": processed_turns,
        "total_turns": total_turns,
    }
    if extra:
        payload["extra"] = extra
    write_json(out_dir / "progress.json", payload)


def write_checkpoint(
    out_dir: Path,
    *,
    recording: Path,
    docx: Path,
    section_name: str,
    accepted_rows: List[Dict],
    reject_rows: List[Dict],
    params: Dict,
) -> None:
    write_json(
        out_dir / "segments_metadata.json",
        {
            "recording": str(recording),
            "docx": str(docx),
            "transcript_section": section_name,
            "segments": len(accepted_rows),
            "rejected": len(reject_rows),
            "params": params,
            "rows": accepted_rows,
        },
    )
    write_json(out_dir / "rejections.json", reject_rows)


def parse_docx(docx_path: Path) -> TranscriptDoc:
    doc = Document(str(docx_path))
    raw_lines = [normalize_spaces(p.text) for p in doc.paragraphs]
    lines = [line for line in raw_lines if line]
    if not lines:
        raise ValueError(f"No text found in {docx_path}")

    preface: List[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        if line == "***":
            body_start = i + 1
            break
        preface.append(line)
    body = lines[body_start:]

    preface_text = "\n".join(preface)
    verbatim_until_s = None
    match = re.search(r"doslovn[ýy]\s+prepis\s+do\s+(\d+)\.\s*min", preface_text, re.IGNORECASE)
    if match:
        verbatim_until_s = float(int(match.group(1)) * 60)

    global_low_trust = bool(
        re.search(r"ned[oô]sledn[ýy]\s+prepis|v[ýy]pisky|nepresnosti|je potrebn[ée]", preface_text, re.IGNORECASE)
    )

    sections_raw: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    current_section = "main"
    for line_no, line in enumerate(body, start=1):
        marker = PART_MARKER_RE.match(line)
        if marker:
            current_section = f"part{int(marker.group(1))}"
            continue
        sections_raw[current_section].append((line_no, line))

    sections: Dict[str, TranscriptSection] = {}
    for section_name, section_lines in sections_raw.items():
        turns: List[TranscriptTurn] = []
        turn_index = 1
        sent_index = 1
        for line_no, raw_line in section_lines:
            cleaned = clean_transcript_text(raw_line)
            if not cleaned or cleaned == "***" or ELLIPSIS_ONLY_RE.match(cleaned):
                continue
            if not WORD_RE.search(cleaned):
                continue
            low_trust = global_low_trust or ("..." in raw_line) or ("…" in raw_line) or ("_" in raw_line)
            turn_tokens = tokenize_norm(cleaned)
            if not turn_tokens:
                continue
            turn = TranscriptTurn(
                turn_id=f"{section_name}_t{turn_index:04d}",
                line_no=line_no,
                text=cleaned,
                norm=" ".join(turn_tokens),
                tokens=turn_tokens,
                anchors=anchor_tokens(turn_tokens),
                low_trust=low_trust,
            )
            for piece in split_into_sentences(cleaned):
                sent_tokens = tokenize_norm(piece)
                if not sent_tokens:
                    continue
                turn.sentences.append(
                    TranscriptSentence(
                        sentence_id=f"{section_name}_s{sent_index:04d}",
                        turn_id=turn.turn_id,
                        order=sent_index,
                        text=piece,
                        norm=" ".join(sent_tokens),
                        tokens=sent_tokens,
                        anchors=anchor_tokens(sent_tokens),
                        low_trust=low_trust,
                    )
                )
                sent_index += 1
            if turn.sentences:
                turns.append(turn)
                turn_index += 1
        sections[section_name] = TranscriptSection(name=section_name, turns=turns)

    return TranscriptDoc(
        docx_path=docx_path,
        verbatim_until_s=verbatim_until_s,
        global_low_trust=global_low_trust,
        preface_notes=preface,
        sections=sections,
    )


def build_jobs(base_dir: Path, out_prefix: str) -> List[RecordingJob]:
    jobs: List[RecordingJob] = []
    for rec_dir in sorted(p for p in base_dir.glob("nahravka_*") if p.is_dir()):
        wavs = wav_files(rec_dir)
        docxs = sorted(rec_dir.glob("*.docx"))
        if not wavs or not docxs:
            continue
        transcript_cache: Dict[Path, TranscriptDoc] = {}
        single_docx = docxs[0] if len(docxs) == 1 else None

        for wav in wavs:
            stem = wav.stem.lower()
            rec_num = recording_number(rec_dir, wav)
            if rec_num is None:
                continue
            docx_path = choose_docx_for_wav(docxs, rec_num)
            transcript = transcript_cache.get(docx_path)
            if transcript is None:
                transcript = parse_docx(docx_path)
                transcript_cache[docx_path] = transcript

            section_names = sorted(transcript.sections)
            section_name = "main" if "main" in transcript.sections else section_names[0]
            numbered_sections = {
                int(re.search(r"\d+", name).group(0)): name
                for name in transcript.sections
                if name.startswith("part") and re.search(r"\d+", name)
            }
            if single_docx is not None and len(wavs) > 1:
                part = recording_part_number(stem)
                if part is not None:
                    section_name = numbered_sections.get(part) or section_name
                elif numbered_sections:
                    section_name = "main" if "main" in transcript.sections else section_names[0]

            suffix = recording_part_suffix(stem)
            rec_code = f"r{rec_num:02d}{suffix}"
            jobs.append(
                RecordingJob(
                    rec_code=rec_code,
                    wav_path=wav,
                    docx_path=docx_path,
                    section_name=section_name,
                    out_dir=rec_dir / f"{out_prefix}_{rec_code}",
                    verbatim_until_s=transcript.verbatim_until_s,
                    global_low_trust=transcript.global_low_trust,
                )
            )
    return jobs


def load_whisper_model(model_path: Optional[Path], compute_type: str) -> WhisperModel:
    resolved = model_path or find_hf_snapshot("Systran/faster-whisper-small")
    return WhisperModel(str(resolved), device="cpu", compute_type=compute_type)


def localize_pyannote_config(config_path: Path) -> Path:
    config_text = config_path.read_text(encoding="utf-8")
    replacements = {
        "    embedding: pyannote/wespeaker-voxceleb-resnet34-LM": (
            "    embedding:\n"
            f"      checkpoint: {find_hf_snapshot('pyannote/wespeaker-voxceleb-resnet34-LM') / 'pytorch_model.bin'}"
        ),
        "    segmentation: pyannote/segmentation-3.0": (
            "    segmentation:\n"
            f"      checkpoint: {find_hf_snapshot('pyannote/segmentation-3.0') / 'pytorch_model.bin'}"
        ),
    }
    for old_value, new_value in replacements.items():
        config_text = config_text.replace(old_value, new_value)
    fd, tmp_name = tempfile.mkstemp(prefix="pyannote_local_", suffix=".yaml")
    os.close(fd)
    localized = Path(tmp_name)
    localized.write_text(config_text, encoding="utf-8")
    return localized


def load_diarization_pipeline(model_config: Optional[Path], device: str) -> Pipeline:
    if model_config is None:
        model_config = find_hf_snapshot("pyannote/speaker-diarization-3.1") / "config.yaml"
    localized_config = localize_pyannote_config(model_config)
    try:
        pipeline = Pipeline.from_pretrained(localized_config)
    finally:
        if localized_config.exists():
            localized_config.unlink()
    pipeline.to(torch.device(device))
    return pipeline


def default_force_align_repo(language_code: Optional[str]) -> str:
    lang = (language_code or "sk").split("-")[0].lower()
    if lang == "sk":
        return "comodoro/wav2vec2-xls-r-300m-sk-cv8"
    raise ValueError(f"No offline force-align model configured for language '{language_code}'")


def load_force_aligner(model_path: Optional[Path], language_code: Optional[str], device: str) -> ForceAlignResources:
    if whisperx is None:
        raise ImportError("whisperx is required for word-level transcript validation")
    align_language = (language_code or "sk").split("-")[0].lower()
    resolved = model_path or find_hf_snapshot(
        default_force_align_repo(align_language),
        required_files=["preprocessor_config.json", "config.json"],
    )
    model, metadata = whisperx.load_align_model(language_code=align_language, device=device, model_name=str(resolved))
    return ForceAlignResources(model=model, metadata=metadata, device=device, language_code=align_language)


def load_recording_audio(wav_path: Path) -> np.ndarray:
    if whisperx_load_audio is None:
        raise ImportError("whisperx audio loader is required for word-level transcript validation")
    return whisperx_load_audio(str(wav_path))


def clip_audio_span(start: float, end: float, total_duration: float, pad: float) -> Tuple[float, float]:
    left = max(0.0, start - pad)
    right = min(total_duration, end + pad)
    if right <= left:
        right = min(total_duration, max(left + 0.05, end))
    return left, right


def slice_audio(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    left = max(0, min(len(audio), int(math.floor(start * AUDIO_SAMPLE_RATE))))
    right = max(left + 1, min(len(audio), int(math.ceil(end * AUDIO_SAMPLE_RATE))))
    return audio[left:right]


def transcribe_clip_text(model: WhisperModel, audio_clip: np.ndarray, language: Optional[str]) -> str:
    segments, _ = model.transcribe(
        audio_clip,
        language=language,
        beam_size=3,
        best_of=3,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    return normalize_spaces(" ".join((seg.text or "").strip() for seg in segments if (seg.text or "").strip()))


def read_diarization_turns(path: Path) -> List[DiarTurn]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    turns: List[DiarTurn] = []
    for item in payload:
        start = float(item["start"])
        end = float(item["end"])
        speaker = str(item["speaker"])
        if end > start:
            turns.append(DiarTurn(start=start, end=end, speaker=speaker))
    turns.sort(key=lambda turn: (turn.start, turn.end))
    return turns


def synthesize_words_from_sentences(sentence_payload: Sequence[Dict[str, Any]]) -> Tuple[List[AsrSentence], List[AsrWord]]:
    words: List[AsrWord] = []
    sentences: List[AsrSentence] = []
    word_cursor = 0
    for idx, item in enumerate(sentence_payload):
        raw = normalize_spaces(str(item.get("text", "")).strip())
        tokens = tokenize_norm(raw)
        if not raw or not tokens:
            continue
        start = float(item["start"])
        end = float(item["end"])
        duration = max(0.01, end - start)
        piece = duration / max(1, len(tokens))
        word_start = word_cursor
        for tok_idx, token in enumerate(tokens):
            token_start = start + piece * tok_idx
            token_end = end if tok_idx == len(tokens) - 1 else min(end, start + piece * (tok_idx + 1))
            words.append(
                AsrWord(
                    start=token_start,
                    end=max(token_start + 0.01, token_end),
                    text=token,
                    norm=token,
                )
            )
            word_cursor += 1
        sentences.append(
            AsrSentence(
                idx=int(item.get("idx", idx)),
                start=start,
                end=end,
                text=raw,
                norm=" ".join(tokens),
                tokens=tokens,
                anchors=anchor_tokens(tokens),
                word_start=word_start,
                word_end=word_cursor,
            )
        )
    return sentences, words


def read_asr_cache(sentences_path: Path, words_path: Optional[Path]) -> Tuple[List[AsrSentence], List[AsrWord]]:
    sentence_payload = json.loads(sentences_path.read_text(encoding="utf-8"))
    if words_path is None or not words_path.exists():
        return synthesize_words_from_sentences(sentence_payload)

    word_payload = json.loads(words_path.read_text(encoding="utf-8"))

    words: List[AsrWord] = []
    for item in word_payload:
        raw = normalize_spaces(str(item.get("text", "")).strip())
        norm = normalize_for_match(raw)
        if not raw or not norm:
            continue
        words.append(
            AsrWord(
                start=float(item["start"]),
                end=float(item["end"]),
                text=raw,
                norm=norm,
            )
        )

    sentences: List[AsrSentence] = []
    running_word_start = 0
    for idx, item in enumerate(sentence_payload):
        raw = normalize_spaces(str(item.get("text", "")).strip())
        tokens = tokenize_norm(raw)
        if not raw or not tokens:
            continue
        if "word_start" in item and "word_end" in item:
            word_start = int(item["word_start"])
            word_end = int(item["word_end"])
        else:
            word_start = running_word_start
            word_end = running_word_start + len(tokens)
            running_word_start = word_end
        if word_end <= word_start:
            continue
        sentences.append(
            AsrSentence(
                idx=int(item.get("idx", idx)),
                start=float(item["start"]),
                end=float(item["end"]),
                text=raw,
                norm=" ".join(tokens),
                tokens=tokens,
                anchors=anchor_tokens(tokens),
                word_start=word_start,
                word_end=word_end,
            )
        )

    return sentences, words


def parse_force_aligned_words(word_segments: Sequence[Dict], offset: float = 0.0) -> List[Dict[str, Any]]:
    aligned_words: List[Dict[str, Any]] = []
    for item in word_segments:
        text = normalize_spaces(str(item.get("word", "")).strip())
        if not text:
            continue
        tokens = tokenize_norm(text)
        if not tokens:
            continue
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        aligned_words.append(
            {
                "text": text,
                "tokens": tokens,
                "score": float(item.get("score", 0.0) or 0.0),
                "start": float(start) + offset,
                "end": float(end) + offset,
            }
        )
    return aligned_words


def force_align_sentence(
    sentence: TranscriptSentence,
    audio: np.ndarray,
    rough_start: float,
    rough_end: float,
    aligner: ForceAlignResources,
    window_pad: float,
    align_slack: float,
    refine_pad: float,
) -> Tuple[List[Dict[str, Any]], float, float]:
    total_duration = len(audio) / AUDIO_SAMPLE_RATE
    window_start, window_end = clip_audio_span(rough_start, rough_end, total_duration, window_pad)
    audio_clip = slice_audio(audio, window_start, window_end)
    local_duration = max(0.01, window_end - window_start)
    guide_start = max(0.0, (rough_start - window_start) - align_slack)
    guide_end = min(local_duration, (rough_end - window_start) + align_slack)
    try:
        result = whisperx.align(
            [{"start": guide_start, "end": guide_end, "text": sentence.text}],
            aligner.model,
            aligner.metadata,
            audio_clip,
            aligner.device,
            return_char_alignments=False,
        )
    except Exception:
        return [], rough_start, rough_end

    aligned_words = parse_force_aligned_words(result.get("word_segments", []), offset=window_start)
    if not aligned_words:
        return [], rough_start, rough_end

    refined_start = max(window_start, aligned_words[0]["start"] - refine_pad)
    refined_end = min(window_end, aligned_words[-1]["end"] + refine_pad)
    if refined_end <= refined_start:
        return aligned_words, rough_start, rough_end
    return aligned_words, refined_start, refined_end


def summarize_force_alignment(sentence: TranscriptSentence, aligned_words: Sequence[Dict[str, Any]]) -> Dict[str, float | int]:
    scores = [word["score"] for word in aligned_words]
    anchor_set = set(sentence.anchors)
    anchor_scores = [
        word["score"]
        for word in aligned_words
        if any(token in anchor_set for token in word["tokens"])
    ]
    anchor_hit_count = sum(score >= 0.2 for score in anchor_scores)
    return {
        "force_word_count": len(aligned_words),
        "force_mean_score": statistics.fmean(scores) if scores else 0.0,
        "force_median_score": statistics.median(scores) if scores else 0.0,
        "force_max_score": max(scores) if scores else 0.0,
        "force_ge20_ratio": (sum(score >= 0.2 for score in scores) / len(scores)) if scores else 0.0,
        "force_anchor_count": len(anchor_scores),
        "force_anchor_hit_count": anchor_hit_count,
        "force_anchor_hit_ratio": (anchor_hit_count / len(anchor_scores)) if anchor_scores else 0.0,
        "force_anchor_mean": statistics.fmean(anchor_scores) if anchor_scores else 0.0,
        "force_anchor_max": max(anchor_scores) if anchor_scores else 0.0,
    }


def validation_quality(force_metrics: Dict[str, float | int], clip_asr_score: float) -> float:
    return max(
        clip_asr_score,
        float(force_metrics.get("force_anchor_mean", 0.0)),
        float(force_metrics.get("force_anchor_max", 0.0)) * 0.9,
        float(force_metrics.get("force_mean_score", 0.0)) * 0.8,
    )


def force_alignment_core_span(
    sentence: TranscriptSentence,
    aligned_words: Sequence[Dict[str, Any]],
    fallback_start: float,
    fallback_end: float,
    pad: float = 0.03,
    min_word_score: float = 0.2,
) -> Tuple[float, float]:
    if not aligned_words:
        return fallback_start, fallback_end

    anchor_set = set(sentence.anchors)
    anchor_words = [
        word
        for word in aligned_words
        if word["score"] >= min_word_score and any(token in anchor_set for token in word["tokens"])
    ]
    if anchor_words:
        chosen = anchor_words
    else:
        chosen = [word for word in aligned_words if word["score"] >= min_word_score] or list(aligned_words)

    core_start = max(fallback_start, float(chosen[0]["start"]) - pad)
    core_end = min(fallback_end, float(chosen[-1]["end"]) + pad)
    if core_end <= core_start:
        return fallback_start, fallback_end
    return core_start, core_end


def validate_candidate_segment(
    cand: CandidateSegment,
    audio: np.ndarray,
    whisper_model: WhisperModel,
    force_aligner: ForceAlignResources,
    language: Optional[str],
    weights: Dict[str, float],
    max_text_rate: float,
    force_window_pad: float,
    force_align_slack: float,
    force_refine_pad: float,
    clip_asr_pad: float,
) -> CandidateValidation:
    aligned_words, refined_start, refined_end = force_align_sentence(
        sentence=cand.sentence,
        audio=audio,
        rough_start=cand.start,
        rough_end=cand.end,
        aligner=force_aligner,
        window_pad=force_window_pad,
        align_slack=force_align_slack,
        refine_pad=force_refine_pad,
    )
    force_metrics = summarize_force_alignment(cand.sentence, aligned_words)
    rough_duration = max(0.01, cand.end - cand.start)
    refined_duration = max(0.01, refined_end - refined_start)
    force_core_start, force_core_end = force_alignment_core_span(
        cand.sentence,
        aligned_words,
        refined_start,
        refined_end,
    )
    keep_ratio = min(2.0, refined_duration / rough_duration)
    text_rate = len(cand.sentence.tokens) / refined_duration

    if text_rate > max_text_rate:
        return CandidateValidation(
            ok=False,
            reason="text_too_dense",
            start=refined_start,
            end=refined_end,
            force_core_start=force_core_start,
            force_core_end=force_core_end,
            clip_asr_text="",
            clip_asr_score=0.0,
            text_rate=text_rate,
            force_keep_ratio=keep_ratio,
            **force_metrics,
        )

    clip_start, clip_end = clip_audio_span(refined_start, refined_end, len(audio) / AUDIO_SAMPLE_RATE, clip_asr_pad)
    clip_text = transcribe_clip_text(whisper_model, slice_audio(audio, clip_start, clip_end), language=language)
    clip_norm = normalize_for_match(clip_text)
    clip_tokens = clip_norm.split()
    clip_score, _ = score_match(
        cand.sentence.norm,
        cand.sentence.anchors,
        clip_norm,
        anchor_tokens(clip_tokens),
        weights,
    )

    anchor_count = len(cand.sentence.anchors)
    token_count = len(cand.sentence.tokens)
    force_anchor_max = float(force_metrics["force_anchor_max"])
    force_anchor_mean = float(force_metrics["force_anchor_mean"])
    force_anchor_hit_ratio = float(force_metrics["force_anchor_hit_ratio"])
    force_mean = float(force_metrics["force_mean_score"])

    clip_threshold = 0.62 if token_count <= 2 or anchor_count == 0 else 0.56
    ok = False
    if clip_score >= clip_threshold:
        ok = True
    elif anchor_count >= 2:
        ok = (
            force_anchor_mean >= 0.5
            and force_anchor_hit_ratio >= 0.5
        ) or (
            force_anchor_mean >= 0.4
            and force_mean >= 0.35
            and force_anchor_hit_ratio >= 0.5
        )
    elif anchor_count == 1:
        ok = force_anchor_max >= 0.6 or (force_anchor_max >= 0.52 and force_mean >= 0.35)

    if ok and cand.used_proportional_fallback and validation_quality(force_metrics, clip_score) < 0.52:
        ok = False

    if ok and keep_ratio < 0.42 and clip_score < clip_threshold:
        ok = False

    return CandidateValidation(
        ok=ok,
        reason=None if ok else "transcript_mismatch",
        start=refined_start,
        end=refined_end,
        force_core_start=force_core_start,
        force_core_end=force_core_end,
        clip_asr_text=clip_text,
        clip_asr_score=clip_score,
        text_rate=text_rate,
        force_keep_ratio=keep_ratio,
        **force_metrics,
    )


def transcribe_to_sentences(
    model: WhisperModel,
    wav_path: Path,
    language: Optional[str],
    sentence_gap: float,
) -> Tuple[List[AsrSentence], List[AsrWord]]:
    segments, _ = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )

    words: List[AsrWord] = []
    for seg in segments:
        for w in list(seg.words or []):
            if w.start is None or w.end is None:
                continue
            raw = normalize_spaces((w.word or "").strip())
            norm = normalize_for_match(raw)
            if not raw or not norm:
                continue
            words.append(AsrWord(start=float(w.start), end=float(w.end), text=raw, norm=norm))

    if not words:
        return [], []

    sentences: List[AsrSentence] = []
    start_idx = 0
    current: List[AsrWord] = [words[0]]

    def flush() -> None:
        nonlocal current, start_idx
        if not current:
            return
        sent_text = normalize_spaces(" ".join(w.text for w in current))
        sent_tokens = tokenize_norm(sent_text)
        if sent_tokens:
            sentences.append(
                AsrSentence(
                    idx=len(sentences),
                    start=current[0].start,
                    end=current[-1].end,
                    text=sent_text,
                    norm=" ".join(sent_tokens),
                    tokens=sent_tokens,
                    anchors=anchor_tokens(sent_tokens),
                    word_start=start_idx,
                    word_end=start_idx + len(current),
                )
            )
        start_idx += len(current)
        current = []

    for word in words[1:]:
        prev = current[-1]
        gap = max(0.0, word.start - prev.end)
        should_break = prev.text.endswith((".", "!", "?", "…")) or gap >= sentence_gap
        if should_break:
            flush()
            current = [word]
        else:
            current.append(word)
    flush()
    return sentences, words


def diarize_full_recording(pipeline: Pipeline, wav_path: Path, num_speakers: Optional[int]) -> List[DiarTurn]:
    if num_speakers is None:
        annotation = pipeline(str(wav_path))
    else:
        annotation = pipeline(str(wav_path), num_speakers=max(1, num_speakers))
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    turns: List[DiarTurn] = []
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        start, end = float(seg.start), float(seg.end)
        if end > start:
            turns.append(DiarTurn(start=start, end=end, speaker=str(speaker)))
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def token_weights(turns: Sequence[TranscriptTurn], asr_sentences: Sequence[AsrSentence]) -> Dict[str, float]:
    freq = Counter()
    for turn in turns:
        freq.update(set(turn.anchors))
    for sent in asr_sentences:
        freq.update(set(sent.anchors))
    weights: Dict[str, float] = {}
    for token, count in freq.items():
        weights[token] = 1.0 / math.sqrt(count)
    return weights


def weighted_overlap(left: Sequence[str], right: Sequence[str], weights: Dict[str, float]) -> Tuple[float, int]:
    left_set = set(left)
    right_set = set(right)
    common = left_set & right_set
    if not left_set:
        return 0.0, 0
    common_weight = sum(weights.get(tok, 1.0) for tok in common)
    total_weight = sum(weights.get(tok, 1.0) for tok in left_set)
    return common_weight / max(total_weight, 1e-9), len(common)


def score_match(query_norm: str, query_anchors: Sequence[str], window_norm: str, window_anchors: Sequence[str], weights: Dict[str, float]) -> Tuple[float, int]:
    if not query_norm or not window_norm:
        return 0.0, 0
    ratio = fuzz.ratio(query_norm, window_norm) / 100.0
    partial = fuzz.partial_ratio(query_norm, window_norm) / 100.0
    token_set = fuzz.token_set_ratio(query_norm, window_norm) / 100.0
    overlap, common_anchor_count = weighted_overlap(query_anchors, window_anchors, weights)
    if query_anchors:
        score = 0.35 * ratio + 0.20 * partial + 0.20 * token_set + 0.25 * overlap
    else:
        score = 0.45 * ratio + 0.30 * partial + 0.25 * token_set
    if common_anchor_count >= 2:
        score += 0.03
    return min(score, 1.0), common_anchor_count


def alignment_threshold(token_count: int, min_score: float, short_turn_bonus: float) -> float:
    if token_count >= 3:
        return min_score
    return min_score + short_turn_bonus


def candidate_value(score: float, threshold: float, common_anchors: int) -> float:
    return (score - threshold) + 0.02 * min(common_anchors, 2)


def search_best_window(
    query_norm: str,
    query_anchors: Sequence[str],
    asr_sentences: Sequence[AsrSentence],
    weights: Dict[str, float],
    cursor: int,
    lookahead: int,
    max_merge: int,
) -> Optional[MatchWindow]:
    if not asr_sentences:
        return None
    start_min = max(0, cursor - 1)
    start_max = min(len(asr_sentences), cursor + lookahead)
    best: Optional[MatchWindow] = None
    for start in range(start_min, start_max):
        window_norm_parts: List[str] = []
        window_anchors: List[str] = []
        window_text_parts: List[str] = []
        for end in range(start + 1, min(len(asr_sentences), start + max_merge) + 1):
            unit = asr_sentences[end - 1]
            window_norm_parts.append(unit.norm)
            window_anchors.extend(unit.anchors)
            window_text_parts.append(unit.text)
            score, common_count = score_match(
                query_norm,
                query_anchors,
                " ".join(window_norm_parts),
                window_anchors,
                weights,
            )
            candidate = MatchWindow(
                start_idx=start,
                end_idx=end,
                score=score,
                common_anchors=common_count,
                text=normalize_spaces(" ".join(window_text_parts)),
            )
            if best is None or (candidate.score, candidate.common_anchors, -candidate.start_idx) > (
                best.score,
                best.common_anchors,
                -best.start_idx,
            ):
                best = candidate
    return best


def align_turns_greedy(
    turns: Sequence[TranscriptTurn],
    asr_sentences: Sequence[AsrSentence],
    weights: Dict[str, float],
    lookahead: int,
    max_merge: int,
    min_score: float,
    short_turn_bonus: float,
) -> Tuple[List[Tuple[TranscriptTurn, MatchWindow]], List[Dict]]:
    matches: List[Tuple[TranscriptTurn, MatchWindow]] = []
    rejects: List[Dict] = []
    cursor = 0
    for turn in turns:
        best = search_best_window(
            query_norm=turn.norm,
            query_anchors=turn.anchors,
            asr_sentences=asr_sentences,
            weights=weights,
            cursor=cursor,
            lookahead=lookahead,
            max_merge=max_merge,
        )
        threshold = alignment_threshold(len(turn.tokens), min_score, short_turn_bonus)
        if best is None or best.score < threshold:
            rejects.append(
                {
                    "type": "turn_unmatched",
                    "turn_id": turn.turn_id,
                    "line_no": turn.line_no,
                    "text": turn.text,
                    "best_score": round(best.score, 3) if best else None,
                    "best_asr_text": best.text if best else None,
                }
            )
            continue
        matches.append((turn, best))
        cursor = max(cursor, best.end_idx)
    return matches, rejects


def top_turn_candidates(
    turn: TranscriptTurn,
    asr_sentences: Sequence[AsrSentence],
    weights: Dict[str, float],
    max_merge: int,
    min_score: float,
    short_turn_bonus: float,
    candidate_limit: int = 6,
) -> List[TurnCandidate]:
    threshold = alignment_threshold(len(turn.tokens), min_score, short_turn_bonus)
    candidates: List[TurnCandidate] = []

    for start in range(len(asr_sentences)):
        window_norm_parts: List[str] = []
        window_anchors: List[str] = []
        window_text_parts: List[str] = []
        for end in range(start + 1, min(len(asr_sentences), start + max_merge) + 1):
            unit = asr_sentences[end - 1]
            window_norm_parts.append(unit.norm)
            window_anchors.extend(unit.anchors)
            window_text_parts.append(unit.text)
            score, common_count = score_match(
                turn.norm,
                turn.anchors,
                " ".join(window_norm_parts),
                window_anchors,
                weights,
            )
            if score < threshold:
                continue
            candidates.append(
                TurnCandidate(
                    match=MatchWindow(
                        start_idx=start,
                        end_idx=end,
                        score=score,
                        common_anchors=common_count,
                        text=normalize_spaces(" ".join(window_text_parts)),
                    ),
                    value=candidate_value(score, threshold, common_count),
                )
            )

    candidates.sort(
        key=lambda cand: (
            cand.value,
            cand.match.score,
            cand.match.common_anchors,
            -cand.match.start_idx,
        ),
        reverse=True,
    )
    return candidates[:candidate_limit]


def align_turns_global(
    turns: Sequence[TranscriptTurn],
    asr_sentences: Sequence[AsrSentence],
    weights: Dict[str, float],
    max_merge: int,
    min_score: float,
    short_turn_bonus: float,
) -> Tuple[List[Tuple[TranscriptTurn, MatchWindow]], List[Dict]]:
    if not turns or not asr_sentences:
        return [], [
            {
                "type": "turn_unmatched",
                "turn_id": turn.turn_id,
                "line_no": turn.line_no,
                "text": turn.text,
                "best_score": None,
                "best_asr_text": None,
            }
            for turn in turns
        ]

    turn_candidates: List[List[TurnCandidate]] = []
    for turn in turns:
        turn_candidates.append(
            top_turn_candidates(
                turn=turn,
                asr_sentences=asr_sentences,
                weights=weights,
                max_merge=max_merge,
                min_score=min_score,
                short_turn_bonus=short_turn_bonus,
            )
        )

    num_positions = len(asr_sentences)
    prev = [0.0] * (num_positions + 1)
    parents: List[List[Tuple[str, int, int]]] = []

    for candidates in turn_candidates:
        new = prev[:]
        parent: List[Tuple[str, int, int]] = [("skip", pos, -1) for pos in range(num_positions + 1)]

        best_score = prev[0]
        best_pos = 0
        prefix_best: List[Tuple[float, int]] = [(prev[0], 0)] + [(0.0, 0)] * num_positions
        for pos in range(1, num_positions + 1):
            if prev[pos] > best_score:
                best_score = prev[pos]
                best_pos = pos
            prefix_best[pos] = (best_score, best_pos)

        for cand_idx, cand in enumerate(candidates):
            base_score, base_pos = prefix_best[cand.match.start_idx]
            total = base_score + cand.value
            if total > new[cand.match.end_idx]:
                new[cand.match.end_idx] = total
                parent[cand.match.end_idx] = ("match", base_pos, cand_idx)

        for pos in range(1, num_positions + 1):
            if new[pos - 1] > new[pos]:
                new[pos] = new[pos - 1]
                parent[pos] = ("carry", pos - 1, -1)

        prev = new
        parents.append(parent)

    pos = max(range(num_positions + 1), key=lambda idx: prev[idx])
    chosen_by_turn: Dict[int, MatchWindow] = {}
    for turn_idx in range(len(turns) - 1, -1, -1):
        kind, parent_pos, cand_idx = parents[turn_idx][pos]
        while kind == "carry":
            pos = parent_pos
            kind, parent_pos, cand_idx = parents[turn_idx][pos]
        if kind == "match":
            chosen_by_turn[turn_idx] = turn_candidates[turn_idx][cand_idx].match
            pos = parent_pos
        else:
            pos = parent_pos

    resolved_by_turn = dict(chosen_by_turn)
    matched_turn_indexes = sorted(chosen_by_turn)
    for turn_idx, turn in enumerate(turns):
        if turn_idx in resolved_by_turn:
            continue

        prev_candidates = [idx for idx in resolved_by_turn if idx < turn_idx]
        next_candidates = [idx for idx in matched_turn_indexes if idx > turn_idx]
        gap_start = resolved_by_turn[max(prev_candidates)].end_idx if prev_candidates else 0
        gap_end = resolved_by_turn[min(next_candidates)].start_idx if next_candidates else num_positions
        if gap_end <= gap_start:
            continue

        local_units = list(asr_sentences[gap_start:gap_end])
        if not local_units:
            continue

        threshold = alignment_threshold(len(turn.tokens), min_score, short_turn_bonus)
        recovered: Optional[MatchWindow] = None
        best = search_best_window(
            query_norm=turn.norm,
            query_anchors=turn.anchors,
            asr_sentences=local_units,
            weights=weights,
            cursor=0,
            lookahead=len(local_units),
            max_merge=min(max_merge, len(local_units)),
        )
        if best is not None and best.score >= threshold:
            recovered = MatchWindow(
                start_idx=gap_start + best.start_idx,
                end_idx=gap_start + best.end_idx,
                score=best.score,
                common_anchors=best.common_anchors,
                text=best.text,
            )
        else:
            sentence_window = recover_turn_window_from_sentences(
                turn=turn,
                matched_units=local_units,
                weights=weights,
                min_score=min_score,
                short_turn_bonus=short_turn_bonus,
            )
            if sentence_window is not None:
                recovered = MatchWindow(
                    start_idx=gap_start + sentence_window.start_idx,
                    end_idx=gap_start + sentence_window.end_idx,
                    score=sentence_window.score,
                    common_anchors=sentence_window.common_anchors,
                    text=sentence_window.text,
                )

        if recovered is not None:
            resolved_by_turn[turn_idx] = recovered

    matches: List[Tuple[TranscriptTurn, MatchWindow]] = []
    rejects: List[Dict] = []
    for turn_idx, turn in enumerate(turns):
        chosen = resolved_by_turn.get(turn_idx)
        if chosen is not None:
            matches.append((turn, chosen))
            continue
        best = turn_candidates[turn_idx][0].match if turn_candidates[turn_idx] else None
        rejects.append(
            {
                "type": "turn_unmatched",
                "turn_id": turn.turn_id,
                "line_no": turn.line_no,
                "text": turn.text,
                "best_score": round(best.score, 3) if best else None,
                "best_asr_text": best.text if best else None,
            }
        )
    return matches, rejects


def align_turns(
    turns: Sequence[TranscriptTurn],
    asr_sentences: Sequence[AsrSentence],
    weights: Dict[str, float],
    lookahead: int,
    max_merge: int,
    min_score: float,
    short_turn_bonus: float,
    mode: str,
) -> Tuple[List[Tuple[TranscriptTurn, MatchWindow]], List[Dict]]:
    if mode == "greedy":
        return align_turns_greedy(
            turns=turns,
            asr_sentences=asr_sentences,
            weights=weights,
            lookahead=lookahead,
            max_merge=max_merge,
            min_score=min_score,
            short_turn_bonus=short_turn_bonus,
        )
    return align_turns_global(
        turns=turns,
        asr_sentences=asr_sentences,
        weights=weights,
        max_merge=max_merge,
        min_score=min_score,
        short_turn_bonus=short_turn_bonus,
    )


def proportional_sentence_chunks(sentences: Sequence[TranscriptSentence], words: Sequence[AsrWord], base_score: float, matched_asr_text: str) -> List[CandidateSegment]:
    if not sentences or not words:
        return []
    if len(sentences) == 1:
        return [
            CandidateSegment(
                sentence=sentences[0],
                start=words[0].start,
                end=words[-1].end,
                alignment_score=base_score,
                matched_asr_text=matched_asr_text,
                used_proportional_fallback=True,
            )
        ]

    token_counts = [max(1, len(s.tokens)) for s in sentences]
    total_tokens = sum(token_counts)
    boundaries = [0]
    cumulative = 0
    for count in token_counts[:-1]:
        cumulative += count
        pos = round((cumulative / total_tokens) * len(words))
        pos = max(boundaries[-1] + 1, min(len(words) - (len(sentences) - len(boundaries)), pos))
        boundaries.append(pos)
    boundaries.append(len(words))

    segments: List[CandidateSegment] = []
    for idx, sentence in enumerate(sentences):
        start_word = boundaries[idx]
        end_word = boundaries[idx + 1]
        chunk = words[start_word:end_word]
        if not chunk:
            continue
        segments.append(
            CandidateSegment(
                sentence=sentence,
                start=chunk[0].start,
                end=chunk[-1].end,
                alignment_score=base_score,
                matched_asr_text=normalize_spaces(" ".join(w.text for w in chunk)),
                used_proportional_fallback=True,
            )
        )
    return segments


def sentence_alignment_candidates(
    sentences: Sequence[TranscriptSentence],
    matched_units: Sequence[AsrSentence],
    weights: Dict[str, float],
    min_score: float,
    short_turn_bonus: float,
) -> Dict[int, List[TurnCandidate]]:
    if not sentences or not matched_units:
        return {}
    local_max_merge = min(
        len(matched_units),
        max(3, math.ceil(len(matched_units) / max(1, len(sentences))) + 2),
    )
    by_index: Dict[int, List[TurnCandidate]] = {}
    for idx, sentence in enumerate(sentences):
        threshold = alignment_threshold(len(sentence.tokens), min_score, short_turn_bonus)
        candidates: List[TurnCandidate] = []
        for start in range(len(matched_units)):
            window_norm_parts: List[str] = []
            window_anchors: List[str] = []
            window_text_parts: List[str] = []
            for end in range(start + 1, min(len(matched_units), start + local_max_merge) + 1):
                unit = matched_units[end - 1]
                window_norm_parts.append(unit.norm)
                window_anchors.extend(unit.anchors)
                window_text_parts.append(unit.text)
                score, common_count = score_match(
                    sentence.norm,
                    sentence.anchors,
                    " ".join(window_norm_parts),
                    window_anchors,
                    weights,
                )
                if score < threshold:
                    continue
                candidates.append(
                    TurnCandidate(
                        match=MatchWindow(
                            start_idx=start,
                            end_idx=end,
                            score=score,
                            common_anchors=common_count,
                            text=normalize_spaces(" ".join(window_text_parts)),
                        ),
                        value=candidate_value(score, threshold, common_count),
                    )
                )
        candidates.sort(
            key=lambda cand: (
                cand.value,
                cand.match.score,
                cand.match.common_anchors,
                -cand.match.start_idx,
            ),
            reverse=True,
        )
        by_index[idx] = candidates[:8]
    return by_index


def choose_sentence_windows(
    sentences: Sequence[TranscriptSentence],
    matched_units: Sequence[AsrSentence],
    weights: Dict[str, float],
    min_score: float,
    short_turn_bonus: float,
) -> Dict[int, MatchWindow]:
    if not sentences or not matched_units:
        return {}

    candidate_map = sentence_alignment_candidates(
        sentences=sentences,
        matched_units=matched_units,
        weights=weights,
        min_score=min_score,
        short_turn_bonus=short_turn_bonus,
    )
    num_positions = len(matched_units)
    prev = [0.0] * (num_positions + 1)
    parents: List[List[Tuple[str, int, int]]] = []

    for sent_idx in range(len(sentences)):
        candidates = candidate_map.get(sent_idx, [])
        new = prev[:]
        parent: List[Tuple[str, int, int]] = [("skip", pos, -1) for pos in range(num_positions + 1)]

        best_score = prev[0]
        best_pos = 0
        prefix_best: List[Tuple[float, int]] = [(prev[0], 0)] + [(0.0, 0)] * num_positions
        for pos in range(1, num_positions + 1):
            if prev[pos] > best_score:
                best_score = prev[pos]
                best_pos = pos
            prefix_best[pos] = (best_score, best_pos)

        for cand_idx, cand in enumerate(candidates):
            base_score, base_pos = prefix_best[cand.match.start_idx]
            total = base_score + cand.value
            if total > new[cand.match.end_idx]:
                new[cand.match.end_idx] = total
                parent[cand.match.end_idx] = ("match", base_pos, cand_idx)

        for pos in range(1, num_positions + 1):
            if new[pos - 1] > new[pos]:
                new[pos] = new[pos - 1]
                parent[pos] = ("carry", pos - 1, -1)

        prev = new
        parents.append(parent)

    pos = max(range(num_positions + 1), key=lambda idx: prev[idx])
    chosen_by_sentence: Dict[int, MatchWindow] = {}
    for sent_idx in range(len(sentences) - 1, -1, -1):
        kind, parent_pos, cand_idx = parents[sent_idx][pos]
        while kind == "carry":
            pos = parent_pos
            kind, parent_pos, cand_idx = parents[sent_idx][pos]
        if kind == "match":
            chosen_by_sentence[sent_idx] = candidate_map[sent_idx][cand_idx].match
            pos = parent_pos
        else:
            pos = parent_pos

    return chosen_by_sentence


def recover_turn_window_from_sentences(
    turn: TranscriptTurn,
    matched_units: Sequence[AsrSentence],
    weights: Dict[str, float],
    min_score: float,
    short_turn_bonus: float,
) -> Optional[MatchWindow]:
    if len(turn.sentences) <= 1 or not matched_units:
        return None

    local_min_score = max(0.48, min_score - 0.04)
    chosen_by_sentence = choose_sentence_windows(
        sentences=turn.sentences,
        matched_units=matched_units,
        weights=weights,
        min_score=local_min_score,
        short_turn_bonus=short_turn_bonus,
    )
    if not chosen_by_sentence:
        return None

    matched_indexes = sorted(chosen_by_sentence)
    matched_sentences = [turn.sentences[idx] for idx in matched_indexes]
    token_total = sum(len(sentence.tokens) for sentence in turn.sentences)
    token_covered = sum(len(sentence.tokens) for sentence in matched_sentences)
    anchor_sentence_total = sum(1 for sentence in turn.sentences if sentence.anchors)
    anchor_sentence_hits = sum(1 for sentence in matched_sentences if sentence.anchors)
    coverage_ratio = token_covered / max(1, token_total)
    matched_ratio = len(matched_sentences) / max(1, len(turn.sentences))
    mean_score = statistics.fmean(chosen_by_sentence[idx].score for idx in matched_indexes)

    if len(matched_sentences) < min(2, len(turn.sentences)) and coverage_ratio < 0.55:
        return None
    if anchor_sentence_total and anchor_sentence_hits == 0:
        return None
    if coverage_ratio < 0.45 or matched_ratio < 0.4:
        return None
    if mean_score < max(0.48, min_score - 0.03):
        return None

    windows = [chosen_by_sentence[idx] for idx in matched_indexes]
    start_idx = min(window.start_idx for window in windows)
    end_idx = max(window.end_idx for window in windows)
    window_text = normalize_spaces(" ".join(unit.text for unit in matched_units[start_idx:end_idx]))
    return MatchWindow(
        start_idx=start_idx,
        end_idx=end_idx,
        score=mean_score,
        common_anchors=max(window.common_anchors for window in windows),
        text=window_text,
    )


def align_sentences_within_turn(
    sentences: Sequence[TranscriptSentence],
    matched_units: Sequence[AsrSentence],
    words: Sequence[AsrWord],
    weights: Dict[str, float],
    min_score: float,
    short_turn_bonus: float,
) -> List[CandidateSegment]:
    if not sentences or not matched_units:
        return []

    chosen_by_sentence = choose_sentence_windows(
        sentences=sentences,
        matched_units=matched_units,
        weights=weights,
        min_score=min_score,
        short_turn_bonus=short_turn_bonus,
    )

    segments: List[CandidateSegment] = []
    for sent_idx, sentence in enumerate(sentences):
        match = chosen_by_sentence.get(sent_idx)
        if match is None:
            continue
        piece_units = matched_units[match.start_idx : match.end_idx]
        if not piece_units:
            continue
        piece_words = words[piece_units[0].word_start : piece_units[-1].word_end]
        if not piece_words:
            continue
        segments.append(
            CandidateSegment(
                sentence=sentence,
                start=piece_words[0].start,
                end=piece_words[-1].end,
                alignment_score=match.score,
                matched_asr_text=match.text,
                used_proportional_fallback=False,
            )
        )
    return segments


def split_turn_to_sentences(
    turn: TranscriptTurn,
    match: MatchWindow,
    asr_sentences: Sequence[AsrSentence],
    words: Sequence[AsrWord],
    weights: Dict[str, float],
    min_score: float,
    short_turn_bonus: float,
) -> List[CandidateSegment]:
    matched_units = list(asr_sentences[match.start_idx : match.end_idx])
    if not matched_units:
        return []
    start_word = matched_units[0].word_start
    end_word = matched_units[-1].word_end
    matched_words = words[start_word:end_word]
    if len(turn.sentences) <= 1:
        return proportional_sentence_chunks(turn.sentences or [], matched_words, match.score, match.text)

    local_candidates: List[CandidateSegment] = []
    cursor = 0
    local_min_score = max(0.50, min_score - 0.05)
    for sentence in turn.sentences:
        best = search_best_window(
            query_norm=sentence.norm,
            query_anchors=sentence.anchors,
            asr_sentences=matched_units,
            weights=weights,
            cursor=cursor,
            lookahead=min(4, len(matched_units)),
            max_merge=min(3, len(matched_units)),
        )
        sentence_threshold = alignment_threshold(len(sentence.tokens), local_min_score, short_turn_bonus)
        if best is None or best.score < sentence_threshold:
            return proportional_sentence_chunks(turn.sentences, matched_words, match.score, match.text)
        piece_units = matched_units[best.start_idx : best.end_idx]
        if not piece_units:
            return proportional_sentence_chunks(turn.sentences, matched_words, match.score, match.text)
        piece_words = words[piece_units[0].word_start : piece_units[-1].word_end]
        if not piece_words:
            return proportional_sentence_chunks(turn.sentences, matched_words, match.score, match.text)
        local_candidates.append(
            CandidateSegment(
                sentence=sentence,
                start=piece_words[0].start,
                end=piece_words[-1].end,
                alignment_score=min(match.score, best.score),
                matched_asr_text=best.text,
                used_proportional_fallback=False,
            )
        )
        cursor = max(cursor, best.end_idx)
    return local_candidates


def speaker_metrics(start: float, end: float, turns: Sequence[DiarTurn]) -> SpeakerMetrics:
    span = max(0.0, end - start)
    if span <= 0:
        return SpeakerMetrics(speaker="UNK", purity=0.0, overlap_ratio=1.0, covered_purity=0.0, coverage_ratio=0.0)
    clipped: List[Tuple[float, float, str]] = []
    for turn in turns:
        if turn.end <= start:
            continue
        if turn.start >= end:
            break
        s = max(start, turn.start)
        e = min(end, turn.end)
        if e > s:
            clipped.append((s, e, turn.speaker))
    if not clipped:
        return SpeakerMetrics(speaker="UNK", purity=0.0, overlap_ratio=1.0, covered_purity=0.0, coverage_ratio=0.0)

    boundaries = sorted({start, end, *[c[0] for c in clipped], *[c[1] for c in clipped]})
    clean_duration: Dict[str, float] = defaultdict(float)
    overlap_duration = 0.0
    covered_duration = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        active = {spk for s, e, spk in clipped if s < right and e > left}
        dur = right - left
        if active:
            covered_duration += dur
        if len(active) == 1:
            clean_duration[next(iter(active))] += dur
        elif len(active) > 1:
            overlap_duration += dur

    dominant_speaker = "UNK"
    dominant_clean = 0.0
    for speaker, dur in clean_duration.items():
        if dur > dominant_clean:
            dominant_speaker = speaker
            dominant_clean = dur
    purity = dominant_clean / span
    overlap_ratio = overlap_duration / span
    covered_purity = (dominant_clean / covered_duration) if covered_duration > 0 else 0.0
    coverage_ratio = covered_duration / span
    return SpeakerMetrics(
        speaker=dominant_speaker,
        purity=purity,
        overlap_ratio=overlap_ratio,
        covered_purity=covered_purity,
        coverage_ratio=coverage_ratio,
    )


def speaker_purity(start: float, end: float, turns: Sequence[DiarTurn]) -> Tuple[str, float, float]:
    metrics = speaker_metrics(start, end, turns)
    return metrics.speaker, metrics.purity, metrics.overlap_ratio


def strong_validation_for_covered_purity(validation: Optional[CandidateValidation]) -> bool:
    return bool(
        validation is not None
        and validation.clip_asr_score >= 0.56
        and validation.force_anchor_mean >= 0.4
    )


def trim_span_for_speaker_scoring(start: float, end: float, trim_seconds: float) -> Tuple[float, float]:
    span = max(0.0, end - start)
    if trim_seconds <= 0 or span <= 0.45:
        return start, end
    trim_each_side = min(trim_seconds, span * 0.18)
    if (span - (2 * trim_each_side)) < 0.25:
        return start, end
    return start + trim_each_side, end - trim_each_side


def dominant_speaker_intervals(
    start: float,
    end: float,
    speaker: str,
    turns: Sequence[DiarTurn],
    gap_tolerance: float,
) -> List[Tuple[float, float]]:
    spans: List[List[float]] = []
    for turn in turns:
        if turn.end <= start:
            continue
        if turn.start >= end:
            break
        if turn.speaker != speaker:
            continue
        left = max(start, turn.start)
        right = min(end, turn.end)
        if right > left:
            spans.append([left, right])

    if not spans:
        return []

    merged = [spans[0]]
    for left, right in spans[1:]:
        prev = merged[-1]
        if left - prev[1] <= gap_tolerance:
            prev[1] = max(prev[1], right)
        else:
            merged.append([left, right])
    return [(left, right) for left, right in merged]


def snap_segment_to_dominant_speaker(
    start: float,
    end: float,
    score_start: float,
    score_end: float,
    speaker: str,
    turns: Sequence[DiarTurn],
    trim_seconds: float,
    gap_tolerance: float,
    min_keep_ratio: float,
    min_duration: float,
    min_purity: float,
    max_overlap_ratio: float,
) -> Optional[Tuple[float, float, str, float, float]]:
    if speaker == "UNK":
        return None

    spans = dominant_speaker_intervals(start, end, speaker, turns, gap_tolerance=gap_tolerance)
    if not spans:
        return None

    midpoint = (score_start + score_end) / 2.0
    ranked = sorted(
        spans,
        key=lambda span: (
            span[0] <= midpoint <= span[1],
            max(0.0, min(span[1], score_end) - max(span[0], score_start)),
            span[1] - span[0],
        ),
        reverse=True,
    )

    original_duration = max(0.0, end - start)
    for snap_start, snap_end in ranked:
        snapped_duration = snap_end - snap_start
        if snapped_duration < min_duration:
            continue
        if original_duration > 0 and (snapped_duration / original_duration) < min_keep_ratio:
            continue
        trimmed_start, trimmed_end = trim_span_for_speaker_scoring(snap_start, snap_end, trim_seconds)
        snapped_speaker, snapped_purity, snapped_overlap = speaker_purity(trimmed_start, trimmed_end, turns)
        if snapped_speaker != speaker:
            continue
        if snapped_purity < min_purity:
            continue
        if snapped_overlap > max_overlap_ratio:
            continue
        return snap_start, snap_end, snapped_speaker, snapped_purity, snapped_overlap
    return None


def lock_span_to_speaker_core(
    start: float,
    end: float,
    core_start: float,
    core_end: float,
    speaker: str,
    turns: Sequence[DiarTurn],
    trim_seconds: float,
    gap_tolerance: float,
    min_duration: float,
    min_purity: float,
    max_overlap_ratio: float,
) -> Optional[Tuple[float, float, str, float, float]]:
    if speaker == "UNK":
        return None

    spans = dominant_speaker_intervals(start, end, speaker, turns, gap_tolerance=gap_tolerance)
    if not spans:
        return None

    midpoint = (core_start + core_end) / 2.0
    ranked = sorted(
        spans,
        key=lambda span: (
            max(0.0, min(span[1], core_end) - max(span[0], core_start)),
            span[0] <= midpoint <= span[1],
            span[1] - span[0],
        ),
        reverse=True,
    )

    for locked_start, locked_end in ranked:
        if locked_end <= locked_start:
            continue
        if (locked_end - locked_start) < min_duration:
            continue
        if max(0.0, min(locked_end, core_end) - max(locked_start, core_start)) <= 0:
            continue
        trimmed_start, trimmed_end = trim_span_for_speaker_scoring(locked_start, locked_end, trim_seconds)
        locked_speaker, locked_purity, locked_overlap = speaker_purity(trimmed_start, trimmed_end, turns)
        if locked_speaker != speaker:
            continue
        if locked_purity < min_purity:
            continue
        if locked_overlap > max_overlap_ratio:
            continue
        return locked_start, locked_end, locked_speaker, locked_purity, locked_overlap
    return None


def process_recording(
    job: RecordingJob,
    transcript: TranscriptDoc,
    whisper_model: WhisperModel,
    diarization_pipeline: Optional[Pipeline],
    force_aligner: ForceAlignResources,
    precomputed_diar_turns: Optional[Sequence[DiarTurn]],
    asr_cache_paths: Optional[AsrCachePaths],
    language: Optional[str],
    num_speakers: Optional[int],
    sentence_gap: float,
    lookahead: int,
    max_merge: int,
    min_align_score: float,
    short_turn_bonus: float,
    min_purity: float,
    max_overlap_ratio: float,
    max_end: Optional[float],
    min_duration: float,
    checkpoint_every: int,
    align_mode: str,
    speaker_trim: float,
    speaker_snap_gap: float,
    speaker_snap_min_ratio: float,
    min_covered_purity: float,
    min_speaker_coverage_ratio: float,
    covered_purity_max_overlap: float,
    max_text_rate: float,
    force_window_pad: float,
    force_align_slack: float,
    force_refine_pad: float,
    clip_asr_pad: float,
) -> Dict:
    section = transcript.sections.get(job.section_name)
    if section is None:
        raise KeyError(f"Transcript section '{job.section_name}' not found for {job.docx_path}")

    job.out_dir.mkdir(parents=True, exist_ok=True)
    accepted_ndjson = job.out_dir / "accepted_segments.ndjson"
    rejected_ndjson = job.out_dir / "rejections.ndjson"
    segments_tsv = job.out_dir / "segments.tsv"
    accepted_ndjson.write_text("", encoding="utf-8")
    rejected_ndjson.write_text("", encoding="utf-8")
    segments_tsv.write_text("", encoding="utf-8")

    process_limit_s = max_end
    if process_limit_s is None and job.verbatim_until_s is not None:
        process_limit_s = job.verbatim_until_s

    params = {
        "language": language,
        "asr_cache_sentences": str(asr_cache_paths.sentences) if asr_cache_paths is not None else None,
        "asr_cache_words": str(asr_cache_paths.words) if asr_cache_paths is not None else None,
        "sentence_gap": sentence_gap,
        "lookahead": lookahead,
        "max_merge": max_merge,
        "min_align_score": min_align_score,
        "short_turn_bonus": short_turn_bonus,
        "min_purity": min_purity,
        "max_overlap_ratio": max_overlap_ratio,
        "verbatim_until_s": job.verbatim_until_s,
        "process_limit_s": process_limit_s,
        "min_duration": min_duration,
        "checkpoint_every": checkpoint_every,
        "align_mode": align_mode,
        "speaker_trim": speaker_trim,
        "speaker_snap_gap": speaker_snap_gap,
        "speaker_snap_min_ratio": speaker_snap_min_ratio,
        "min_covered_purity": min_covered_purity,
        "min_speaker_coverage_ratio": min_speaker_coverage_ratio,
        "covered_purity_max_overlap": covered_purity_max_overlap,
        "max_text_rate": max_text_rate,
        "force_window_pad": force_window_pad,
        "force_align_slack": force_align_slack,
        "force_refine_pad": force_refine_pad,
        "clip_asr_pad": clip_asr_pad,
        "force_align_language": force_aligner.language_code,
    }

    write_progress(
        job.out_dir,
        stage="starting",
        recording=job.wav_path,
        docx=job.docx_path,
        section_name=job.section_name,
        accepted=0,
        rejected=0,
        total_turns=len(section.turns),
        processed_turns=0,
        extra={"params": params},
    )
    write_checkpoint(
        job.out_dir,
        recording=job.wav_path,
        docx=job.docx_path,
        section_name=job.section_name,
        accepted_rows=[],
        reject_rows=[],
        params=params,
    )

    excerpt_path: Optional[Path] = None
    process_wav = job.wav_path
    if process_limit_s is not None:
        write_progress(
            job.out_dir,
            stage="preparing_excerpt",
            recording=job.wav_path,
            docx=job.docx_path,
            section_name=job.section_name,
            accepted=0,
            rejected=0,
            total_turns=len(section.turns),
            processed_turns=0,
            extra={"process_limit_s": process_limit_s},
        )
        fd, tmp_name = tempfile.mkstemp(prefix=f"{job.rec_code}_", suffix=".wav")
        os.close(fd)
        excerpt_path = Path(tmp_name)
        ffmpeg_excerpt(job.wav_path, excerpt_path, end=process_limit_s)
        process_wav = excerpt_path

    recording_audio: Optional[np.ndarray] = None
    try:
        write_progress(
            job.out_dir,
            stage="transcribing",
            recording=job.wav_path,
            docx=job.docx_path,
            section_name=job.section_name,
            accepted=0,
            rejected=0,
            total_turns=len(section.turns),
            processed_turns=0,
        )
        if (
            asr_cache_paths is not None
            and asr_cache_paths.sentences.exists()
        ):
            asr_sentences, words = read_asr_cache(asr_cache_paths.sentences, asr_cache_paths.words)
        else:
            asr_sentences, words = transcribe_to_sentences(whisper_model, process_wav, language=language, sentence_gap=sentence_gap)
        write_json(
            job.out_dir / "asr_sentences.json",
            [
                {
                    "idx": sent.idx,
                    "start": round(sent.start, 3),
                    "end": round(sent.end, 3),
                    "text": sent.text,
                    "word_start": sent.word_start,
                    "word_end": sent.word_end,
                }
                for sent in asr_sentences
            ],
        )
        write_json(
            job.out_dir / "asr_words.json",
            [
                {
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "text": word.text,
                }
                for word in words
            ],
        )
        if precomputed_diar_turns is None:
            if diarization_pipeline is None:
                raise RuntimeError("Missing diarization pipeline")
            write_progress(
                job.out_dir,
                stage="diarizing",
                recording=job.wav_path,
                docx=job.docx_path,
                section_name=job.section_name,
                accepted=0,
                rejected=0,
                total_turns=len(section.turns),
                processed_turns=0,
                extra={"asr_sentences": len(asr_sentences), "asr_words": len(words)},
            )
            diar_turns = diarize_full_recording(diarization_pipeline, process_wav, num_speakers=num_speakers)
        else:
            diar_turns = list(precomputed_diar_turns)
        write_json(
            job.out_dir / "diarization_turns.json",
            [
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": turn.speaker,
                }
                for turn in diar_turns
            ],
        )
        write_progress(
            job.out_dir,
            stage="loading_audio",
            recording=job.wav_path,
            docx=job.docx_path,
            section_name=job.section_name,
            accepted=0,
            rejected=0,
            total_turns=len(section.turns),
            processed_turns=0,
        )
        recording_audio = load_recording_audio(process_wav)
    finally:
        if excerpt_path is not None and excerpt_path.exists():
            excerpt_path.unlink()

    if recording_audio is None:
        raise RuntimeError(f"Failed to load audio for validation: {job.wav_path}")

    weights = token_weights(section.turns, asr_sentences)
    turn_matches, rejects = align_turns(
        turns=section.turns,
        asr_sentences=asr_sentences,
        weights=weights,
        lookahead=lookahead,
        max_merge=max_merge,
        min_score=min_align_score,
        short_turn_bonus=short_turn_bonus,
        mode=align_mode,
    )

    accepted_rows = []
    reject_rows = list(rejects)
    accepted_span_keys: Dict[Tuple[int, int, str], str] = {}
    seg_index = 1
    for reject in rejects:
        append_ndjson(rejected_ndjson, reject)

    write_progress(
        job.out_dir,
        stage="aligning",
        recording=job.wav_path,
        docx=job.docx_path,
        section_name=job.section_name,
        accepted=0,
        rejected=len(reject_rows),
        total_turns=len(turn_matches),
        processed_turns=0,
        extra={"initial_unmatched_turns": len(rejects), "diar_turns": len(diar_turns)},
    )
    write_checkpoint(
        job.out_dir,
        recording=job.wav_path,
        docx=job.docx_path,
        section_name=job.section_name,
        accepted_rows=accepted_rows,
        reject_rows=reject_rows,
        params=params,
    )

    for turn_idx, (turn, match) in enumerate(turn_matches, start=1):
        candidates = split_turn_to_sentences(
            turn=turn,
            match=match,
            asr_sentences=asr_sentences,
            words=words,
            weights=weights,
            min_score=min_align_score,
            short_turn_bonus=short_turn_bonus,
        )
        for cand in candidates:
            low_trust = cand.sentence.low_trust or job.global_low_trust
            beyond_verbatim = job.verbatim_until_s is not None and cand.end > job.verbatim_until_s
            reason = None
            validation: Optional[CandidateValidation] = None
            dominant_speaker = "UNK"
            purity = 0.0
            overlap_ratio = 1.0
            covered_purity = 0.0
            coverage_ratio = 0.0
            speaker_gate = "primary"
            if cand.end <= cand.start:
                reason = "non_positive_duration"
            elif cand.alignment_score < min_align_score:
                reason = "low_alignment"
            elif beyond_verbatim:
                reason = "beyond_verbatim_limit"
            else:
                validation = validate_candidate_segment(
                    cand=cand,
                    audio=recording_audio,
                    whisper_model=whisper_model,
                    force_aligner=force_aligner,
                    language=language,
                    weights=weights,
                    max_text_rate=max_text_rate,
                    force_window_pad=force_window_pad,
                    force_align_slack=force_align_slack,
                    force_refine_pad=force_refine_pad,
                    clip_asr_pad=clip_asr_pad,
                )
                cand = CandidateSegment(
                    sentence=cand.sentence,
                    start=validation.start,
                    end=validation.end,
                    alignment_score=cand.alignment_score,
                    matched_asr_text=cand.matched_asr_text,
                    used_proportional_fallback=cand.used_proportional_fallback,
                )
                if not validation.ok:
                    reason = validation.reason

            if reason is None:
                speaker_probe_start = cand.start
                speaker_probe_end = cand.end
                if validation is not None:
                    speaker_probe_start = max(cand.start, min(validation.force_core_start, cand.end))
                    speaker_probe_end = min(cand.end, max(validation.force_core_end, cand.start))
                    if speaker_probe_end <= speaker_probe_start:
                        speaker_probe_start = cand.start
                        speaker_probe_end = cand.end

                score_start, score_end = trim_span_for_speaker_scoring(speaker_probe_start, speaker_probe_end, speaker_trim)
                metrics = speaker_metrics(score_start, score_end, diar_turns)
                dominant_speaker = metrics.speaker
                purity = metrics.purity
                overlap_ratio = metrics.overlap_ratio
                covered_purity = metrics.covered_purity
                coverage_ratio = metrics.coverage_ratio

                if validation is not None and purity >= min_purity and overlap_ratio <= max_overlap_ratio:
                    locked = lock_span_to_speaker_core(
                        start=cand.start,
                        end=cand.end,
                        core_start=speaker_probe_start,
                        core_end=speaker_probe_end,
                        speaker=dominant_speaker,
                        turns=diar_turns,
                        trim_seconds=speaker_trim,
                        gap_tolerance=speaker_snap_gap,
                        min_duration=min_duration,
                        min_purity=min_purity,
                        max_overlap_ratio=max_overlap_ratio,
                    )
                    if locked is not None:
                        locked_start, locked_end, dominant_speaker, purity, overlap_ratio = locked
                        cand = CandidateSegment(
                            sentence=cand.sentence,
                            start=locked_start,
                            end=locked_end,
                            alignment_score=cand.alignment_score,
                            matched_asr_text=cand.matched_asr_text,
                            used_proportional_fallback=cand.used_proportional_fallback,
                        )
                    else:
                        cand = CandidateSegment(
                            sentence=cand.sentence,
                            start=speaker_probe_start,
                            end=speaker_probe_end,
                            alignment_score=cand.alignment_score,
                            matched_asr_text=cand.matched_asr_text,
                            used_proportional_fallback=cand.used_proportional_fallback,
                        )
                        speaker_gate = "core"

                snapped = None
                if purity < min_purity or overlap_ratio > max_overlap_ratio:
                    snapped = snap_segment_to_dominant_speaker(
                        start=cand.start,
                        end=cand.end,
                        score_start=score_start,
                        score_end=score_end,
                        speaker=dominant_speaker,
                        turns=diar_turns,
                        trim_seconds=speaker_trim,
                        gap_tolerance=speaker_snap_gap,
                        min_keep_ratio=speaker_snap_min_ratio,
                        min_duration=min_duration,
                        min_purity=min_purity,
                        max_overlap_ratio=max_overlap_ratio,
                    )
                    if snapped is None and (speaker_probe_start != cand.start or speaker_probe_end != cand.end):
                        snapped = snap_segment_to_dominant_speaker(
                            start=speaker_probe_start,
                            end=speaker_probe_end,
                            score_start=score_start,
                            score_end=score_end,
                            speaker=dominant_speaker,
                            turns=diar_turns,
                            trim_seconds=speaker_trim,
                            gap_tolerance=speaker_snap_gap,
                            min_keep_ratio=speaker_snap_min_ratio,
                            min_duration=min_duration,
                            min_purity=min_purity,
                            max_overlap_ratio=max_overlap_ratio,
                        )
                    if snapped is not None:
                        snap_start, snap_end, dominant_speaker, purity, overlap_ratio = snapped
                        cand = CandidateSegment(
                            sentence=cand.sentence,
                            start=snap_start,
                            end=snap_end,
                            alignment_score=cand.alignment_score,
                            matched_asr_text=cand.matched_asr_text,
                            used_proportional_fallback=cand.used_proportional_fallback,
                        )
                        speaker_gate = "snap"
                    elif (
                        strong_validation_for_covered_purity(validation)
                        and dominant_speaker != "UNK"
                        and covered_purity >= min_covered_purity
                        and coverage_ratio >= min_speaker_coverage_ratio
                        and overlap_ratio <= covered_purity_max_overlap
                    ):
                        cand = CandidateSegment(
                            sentence=cand.sentence,
                            start=speaker_probe_start,
                            end=speaker_probe_end,
                            alignment_score=cand.alignment_score,
                            matched_asr_text=cand.matched_asr_text,
                            used_proportional_fallback=cand.used_proportional_fallback,
                        )
                        speaker_gate = "covered"
                if purity < min_purity:
                    if speaker_gate != "covered":
                        reason = "mixed_speaker"
                elif overlap_ratio > max_overlap_ratio:
                    if speaker_gate != "covered":
                        reason = "overlap_detected"

            if reason is not None:
                reject = {
                    "type": reason,
                    "sentence_id": cand.sentence.sentence_id,
                    "turn_id": cand.sentence.turn_id,
                    "text": cand.sentence.text,
                    "start": round(cand.start, 3),
                    "end": round(cand.end, 3),
                    "alignment_score": round(cand.alignment_score, 3),
                    "speaker": dominant_speaker,
                    "speaker_purity": round(purity, 3),
                    "overlap_ratio": round(overlap_ratio, 3),
                    "speaker_covered_purity": round(covered_purity, 3),
                    "speaker_coverage_ratio": round(coverage_ratio, 3),
                    "speaker_gate": speaker_gate,
                    "matched_asr_text": cand.matched_asr_text,
                    "low_trust": low_trust,
                    "used_proportional_fallback": cand.used_proportional_fallback,
                    "clip_asr_text": validation.clip_asr_text if validation else "",
                    "clip_asr_score": round(validation.clip_asr_score, 3) if validation else None,
                    "text_rate": round(validation.text_rate, 3) if validation else None,
                    "force_mean_score": round(validation.force_mean_score, 3) if validation else None,
                    "force_anchor_mean": round(validation.force_anchor_mean, 3) if validation else None,
                    "force_anchor_max": round(validation.force_anchor_max, 3) if validation else None,
                    "force_keep_ratio": round(validation.force_keep_ratio, 3) if validation else None,
                }
                reject_rows.append(reject)
                append_ndjson(rejected_ndjson, reject)
                continue

            if (cand.end - cand.start) < min_duration:
                reject = {
                    "type": "too_short",
                    "sentence_id": cand.sentence.sentence_id,
                    "turn_id": cand.sentence.turn_id,
                    "text": cand.sentence.text,
                    "start": round(cand.start, 3),
                    "end": round(cand.end, 3),
                    "alignment_score": round(cand.alignment_score, 3),
                    "speaker": dominant_speaker,
                    "speaker_purity": round(purity, 3),
                    "overlap_ratio": round(overlap_ratio, 3),
                    "speaker_covered_purity": round(covered_purity, 3),
                    "speaker_coverage_ratio": round(coverage_ratio, 3),
                    "speaker_gate": speaker_gate,
                    "matched_asr_text": cand.matched_asr_text,
                    "low_trust": low_trust,
                    "used_proportional_fallback": cand.used_proportional_fallback,
                    "clip_asr_text": validation.clip_asr_text if validation else "",
                    "clip_asr_score": round(validation.clip_asr_score, 3) if validation else None,
                    "text_rate": round(validation.text_rate, 3) if validation else None,
                    "force_mean_score": round(validation.force_mean_score, 3) if validation else None,
                    "force_anchor_mean": round(validation.force_anchor_mean, 3) if validation else None,
                    "force_anchor_max": round(validation.force_anchor_max, 3) if validation else None,
                    "force_keep_ratio": round(validation.force_keep_ratio, 3) if validation else None,
                }
                reject_rows.append(reject)
                append_ndjson(rejected_ndjson, reject)
                continue

            span_key = (int(round(cand.start * 100)), int(round(cand.end * 100)), dominant_speaker)
            if span_key in accepted_span_keys and accepted_span_keys[span_key] != cand.sentence.text:
                reject = {
                    "type": "duplicate_span_conflict",
                    "sentence_id": cand.sentence.sentence_id,
                    "turn_id": cand.sentence.turn_id,
                    "text": cand.sentence.text,
                    "start": round(cand.start, 3),
                    "end": round(cand.end, 3),
                    "alignment_score": round(cand.alignment_score, 3),
                    "speaker": dominant_speaker,
                    "speaker_purity": round(purity, 3),
                    "overlap_ratio": round(overlap_ratio, 3),
                    "speaker_covered_purity": round(covered_purity, 3),
                    "speaker_coverage_ratio": round(coverage_ratio, 3),
                    "speaker_gate": speaker_gate,
                    "matched_asr_text": cand.matched_asr_text,
                    "low_trust": low_trust,
                    "used_proportional_fallback": cand.used_proportional_fallback,
                    "clip_asr_text": validation.clip_asr_text if validation else "",
                    "clip_asr_score": round(validation.clip_asr_score, 3) if validation else None,
                    "text_rate": round(validation.text_rate, 3) if validation else None,
                    "force_mean_score": round(validation.force_mean_score, 3) if validation else None,
                    "force_anchor_mean": round(validation.force_anchor_mean, 3) if validation else None,
                    "force_anchor_max": round(validation.force_anchor_max, 3) if validation else None,
                    "force_keep_ratio": round(validation.force_keep_ratio, 3) if validation else None,
                    "conflicts_with_text": accepted_span_keys[span_key],
                }
                reject_rows.append(reject)
                append_ndjson(rejected_ndjson, reject)
                continue

            name = f"20260328_{job.rec_code}_s{seg_index:04d}.wav"
            ffmpeg_cut(job.wav_path, job.out_dir / name, cand.start, cand.end)
            accepted = {
                "file": name,
                "start": round(cand.start, 3),
                "end": round(cand.end, 3),
                "duration": round(cand.end - cand.start, 3),
                "speaker": dominant_speaker,
                "speaker_purity": round(purity, 3),
                "overlap_ratio": round(overlap_ratio, 3),
                "speaker_covered_purity": round(covered_purity, 3),
                "speaker_coverage_ratio": round(coverage_ratio, 3),
                "speaker_gate": speaker_gate,
                "alignment_score": round(cand.alignment_score, 3),
                "text": cand.sentence.text,
                "sentence_id": cand.sentence.sentence_id,
                "turn_id": cand.sentence.turn_id,
                "matched_asr_text": cand.matched_asr_text,
                "low_trust": low_trust,
                "used_proportional_fallback": cand.used_proportional_fallback,
                "clip_asr_text": validation.clip_asr_text if validation else "",
                "clip_asr_score": round(validation.clip_asr_score, 3) if validation else None,
                "text_rate": round(validation.text_rate, 3) if validation else None,
                "force_mean_score": round(validation.force_mean_score, 3) if validation else None,
                "force_median_score": round(validation.force_median_score, 3) if validation else None,
                "force_max_score": round(validation.force_max_score, 3) if validation else None,
                "force_ge20_ratio": round(validation.force_ge20_ratio, 3) if validation else None,
                "force_anchor_hit_ratio": round(validation.force_anchor_hit_ratio, 3) if validation else None,
                "force_anchor_mean": round(validation.force_anchor_mean, 3) if validation else None,
                "force_anchor_max": round(validation.force_anchor_max, 3) if validation else None,
                "force_keep_ratio": round(validation.force_keep_ratio, 3) if validation else None,
                "validation_quality": round(
                    validation_quality(
                        {
                            "force_anchor_mean": validation.force_anchor_mean,
                            "force_anchor_max": validation.force_anchor_max,
                            "force_mean_score": validation.force_mean_score,
                        },
                        validation.clip_asr_score,
                    ),
                    3,
                ) if validation else None,
            }
            accepted_rows.append(accepted)
            accepted_span_keys[span_key] = cand.sentence.text
            append_ndjson(accepted_ndjson, accepted)
            with segments_tsv.open("a", encoding="utf-8") as handle:
                handle.write(f"{accepted['file']}\t{accepted['text']}\n")
            seg_index += 1

        if turn_idx % checkpoint_every == 0 or turn_idx == len(turn_matches):
            write_progress(
                job.out_dir,
                stage="aligning",
                recording=job.wav_path,
                docx=job.docx_path,
                section_name=job.section_name,
                accepted=len(accepted_rows),
                rejected=len(reject_rows),
                total_turns=len(turn_matches),
                processed_turns=turn_idx,
            )
            write_checkpoint(
                job.out_dir,
                recording=job.wav_path,
                docx=job.docx_path,
                section_name=job.section_name,
                accepted_rows=accepted_rows,
                reject_rows=reject_rows,
                params=params,
            )
            print(
                f"    progress: turns={turn_idx}/{len(turn_matches)} "
                f"accepted={len(accepted_rows)} rejected={len(reject_rows)}",
                flush=True,
            )

    write_progress(
        job.out_dir,
        stage="done",
        recording=job.wav_path,
        docx=job.docx_path,
        section_name=job.section_name,
        accepted=len(accepted_rows),
        rejected=len(reject_rows),
        total_turns=len(turn_matches),
        processed_turns=len(turn_matches),
    )
    write_checkpoint(
        job.out_dir,
        recording=job.wav_path,
        docx=job.docx_path,
        section_name=job.section_name,
        accepted_rows=accepted_rows,
        reject_rows=reject_rows,
        params=params,
    )

    return {
        "rec_code": job.rec_code,
        "recording": str(job.wav_path),
        "out_dir": str(job.out_dir),
        "segments": len(accepted_rows),
        "rejected": len(reject_rows),
        "transcript_section": job.section_name,
        "turns": len(section.turns),
        "asr_sentences": len(asr_sentences),
        "diar_turns": len(diar_turns),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create transcript-aligned, single-speaker sentence clips from .docx + audio")
    ap.add_argument("--base-dir", type=Path, default=Path("recordings-tuke"))
    ap.add_argument("--recording", action="append", help="Only process these rec_codes, e.g. r01 or r02a")
    ap.add_argument("--out-prefix", default="segments_transcript_aligned")
    ap.add_argument("--whisper-model-path", type=Path, default=None)
    ap.add_argument("--pyannote-config", type=Path, default=None)
    ap.add_argument("--language", default="sk")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--align-mode", choices=["global", "greedy"], default="global")
    ap.add_argument("--sentence-gap", type=float, default=0.9)
    ap.add_argument("--lookahead", type=int, default=12)
    ap.add_argument("--max-merge", type=int, default=4)
    ap.add_argument("--min-align-score", type=float, default=0.52)
    ap.add_argument("--short-turn-bonus", type=float, default=0.02)
    ap.add_argument("--min-purity", type=float, default=0.90)
    ap.add_argument("--max-overlap-ratio", type=float, default=0.05)
    ap.add_argument("--max-end", type=float, default=None, help="Hard cap in seconds for processing each recording")
    ap.add_argument("--min-duration", type=float, default=0.35)
    ap.add_argument("--speaker-trim", type=float, default=0.12)
    ap.add_argument("--speaker-snap-gap", type=float, default=0.12)
    ap.add_argument("--speaker-snap-min-ratio", type=float, default=0.45)
    ap.add_argument("--min-covered-purity", type=float, default=0.90)
    ap.add_argument("--min-speaker-coverage-ratio", type=float, default=0.60)
    ap.add_argument("--covered-purity-max-overlap", type=float, default=0.08)
    ap.add_argument("--max-text-rate", type=float, default=6.5)
    ap.add_argument("--force-align-model-path", type=Path, default=None)
    ap.add_argument("--force-align-language", default=None)
    ap.add_argument("--force-align-device", default="cpu")
    ap.add_argument("--force-window-pad", type=float, default=0.45)
    ap.add_argument("--force-align-slack", type=float, default=0.30)
    ap.add_argument("--force-refine-pad", type=float, default=0.05)
    ap.add_argument("--clip-asr-pad", type=float, default=0.08)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--diarization-device", default="auto")
    ap.add_argument("--diarization-json", type=Path, default=None, help="Reuse diarization_turns.json for a single recording rerun")
    ap.add_argument("--reuse-asr-dir", type=Path, default=None, help="Reuse asr_sentences.json + asr_words.json from an earlier run")
    args = ap.parse_args()

    jobs = build_jobs(args.base_dir, out_prefix=args.out_prefix)
    if args.recording:
        allowed = set(args.recording)
        jobs = [job for job in jobs if job.rec_code in allowed]
    if not jobs:
        raise SystemExit("No matching recordings found.")

    doc_cache: Dict[Path, TranscriptDoc] = {}
    whisper_model = load_whisper_model(args.whisper_model_path, compute_type=args.compute_type)
    diarization_device = args.diarization_device
    if diarization_device == "auto":
        diarization_device = "mps" if torch.backends.mps.is_available() else "cpu"
    precomputed_diar_turns = None
    asr_cache_paths: Optional[AsrCachePaths] = None
    diarization_pipeline: Optional[Pipeline] = None
    if args.diarization_json is not None:
        if len(jobs) != 1:
            raise SystemExit("--diarization-json can only be used when processing a single recording")
        precomputed_diar_turns = read_diarization_turns(args.diarization_json)
    else:
        diarization_pipeline = load_diarization_pipeline(args.pyannote_config, device=diarization_device)
    if args.reuse_asr_dir is not None:
        if len(jobs) != 1:
            raise SystemExit("--reuse-asr-dir can only be used when processing a single recording")
        asr_cache_paths = AsrCachePaths(
            sentences=args.reuse_asr_dir / "asr_sentences.json",
            words=args.reuse_asr_dir / "asr_words.json",
        )
    force_aligner = load_force_aligner(
        args.force_align_model_path,
        language_code=args.force_align_language or args.language,
        device=args.force_align_device,
    )

    summaries = []
    print(
        f"Diarization device: {diarization_device}"
        + (" (reused turns)" if precomputed_diar_turns is not None else ""),
        flush=True,
    )
    print(f"Force-align device: {force_aligner.device}", flush=True)
    for job in jobs:
        if job.docx_path not in doc_cache:
            doc_cache[job.docx_path] = parse_docx(job.docx_path)
        transcript = doc_cache[job.docx_path]
        effective_end = args.max_end
        if effective_end is None and job.verbatim_until_s is not None:
            effective_end = job.verbatim_until_s
        print(
            f"Processing {job.rec_code}: {job.wav_path.name} [{job.section_name}]"
            + (f" up to {effective_end:.1f}s" if effective_end is not None else ""),
            flush=True,
        )
        summary = process_recording(
            job=job,
            transcript=transcript,
            whisper_model=whisper_model,
            diarization_pipeline=diarization_pipeline,
            force_aligner=force_aligner,
            precomputed_diar_turns=precomputed_diar_turns,
            asr_cache_paths=asr_cache_paths,
            language=args.language,
            num_speakers=args.num_speakers,
            sentence_gap=args.sentence_gap,
            lookahead=args.lookahead,
            max_merge=args.max_merge,
            min_align_score=args.min_align_score,
            short_turn_bonus=args.short_turn_bonus,
            min_purity=args.min_purity,
            max_overlap_ratio=args.max_overlap_ratio,
            max_end=args.max_end,
            min_duration=args.min_duration,
            checkpoint_every=args.checkpoint_every,
            align_mode=args.align_mode,
            speaker_trim=args.speaker_trim,
            speaker_snap_gap=args.speaker_snap_gap,
            speaker_snap_min_ratio=args.speaker_snap_min_ratio,
            min_covered_purity=args.min_covered_purity,
            min_speaker_coverage_ratio=args.min_speaker_coverage_ratio,
            covered_purity_max_overlap=args.covered_purity_max_overlap,
            max_text_rate=args.max_text_rate,
            force_window_pad=args.force_window_pad,
            force_align_slack=args.force_align_slack,
            force_refine_pad=args.force_refine_pad,
            clip_asr_pad=args.clip_asr_pad,
        )
        summaries.append(summary)
        print(f"  segments={summary['segments']} rejected={summary['rejected']}", flush=True)

    print("=" * 80, flush=True)
    print("DONE", flush=True)
    print("=" * 80, flush=True)
    for summary in summaries:
        print(
            f"{summary['rec_code']}: segments={summary['segments']} rejected={summary['rejected']} "
            f"turns={summary['turns']} asr_sentences={summary['asr_sentences']} diar_turns={summary['diar_turns']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
