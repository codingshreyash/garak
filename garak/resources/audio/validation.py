# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed validation and provenance for generated WAV candidates."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import wave

_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class LocalTranscriptionError(RuntimeError):
    """Raised when a local transcription backend cannot return text."""


@dataclass(frozen=True)
class Transcript:
    """Transcript text and reproducibility metadata."""

    text: str
    backend: str
    model: str | None = None
    revision: str | None = None


def normalized_words(text: str) -> tuple[str, ...]:
    """Return case-folded lexical tokens for transcript comparison."""

    return tuple(_WORD_PATTERN.findall(text.casefold()))


def _edit_distance(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def transcript_agreement(reference: str, transcript: str) -> dict:
    """Calculate deterministic lexical agreement between source and transcript."""

    reference_words = normalized_words(reference)
    transcript_words = normalized_words(transcript)
    if not reference_words:
        raise ValueError("expected text must contain at least one lexical token")
    edit_count = _edit_distance(reference_words, transcript_words)
    reference_counts: dict[str, int] = {}
    transcript_counts: dict[str, int] = {}
    for word in reference_words:
        reference_counts[word] = reference_counts.get(word, 0) + 1
    for word in transcript_words:
        transcript_counts[word] = transcript_counts.get(word, 0) + 1
    matched = sum(
        min(count, transcript_counts.get(word, 0))
        for word, count in reference_counts.items()
    )
    return {
        "reference_word_count": len(reference_words),
        "transcript_word_count": len(transcript_words),
        "word_error_rate": edit_count / len(reference_words),
        "reference_word_recall": matched / len(reference_words),
    }


def inspect_pcm16_wav(path: str | Path) -> dict:
    """Inspect a WAV and report structure, checksum, and signal level."""

    wav_path = Path(path)
    result = {
        "path": str(wav_path),
        "exists": wav_path.is_file(),
        "format": wav_path.suffix.lower().lstrip("."),
    }
    if not result["exists"]:
        return result | {"valid": False, "reason": "audio_file_missing"}
    raw_file = wav_path.read_bytes()
    result |= {
        "sha256": hashlib.sha256(raw_file, usedforsecurity=False).hexdigest(),
        "byte_size": len(raw_file),
    }
    if wav_path.suffix.lower() != ".wav":
        return result | {"valid": False, "reason": "audio_format_not_wav"}
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
            frames = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        return result | {
            "valid": False,
            "reason": "wav_parse_error",
            "detail": str(exc),
        }
    result |= {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_seconds": frame_count / sample_rate if sample_rate else 0.0,
        "compression": compression,
    }
    if sample_width != 2:
        return result | {"valid": False, "reason": "wav_not_pcm16"}
    if channels not in (1, 2):
        return result | {"valid": False, "reason": "wav_channel_count_unsupported"}
    if compression != "NONE":
        return result | {"valid": False, "reason": "wav_compression_unsupported"}
    if sample_rate <= 0 or frame_count <= 0 or not frames:
        return result | {"valid": False, "reason": "wav_has_no_audio_frames"}
    import numpy

    # WAV PCM16 is always little-endian; read as int64 to avoid overflow on square
    samples = numpy.frombuffer(frames, dtype="<i2").astype(numpy.int64)
    peak = int(numpy.abs(samples).max())
    rms = math.sqrt(float(numpy.square(samples).mean()))
    result |= {
        "peak_pcm16": peak,
        "rms_pcm16": rms,
        "rms_dbfs": 20.0 * math.log10(rms / 32768.0) if rms else None,
    }
    if peak == 0:
        return result | {"valid": False, "reason": "wav_signal_is_silent"}
    return result | {"valid": True, "reason": "valid_pcm16_wav"}


class TransformersWhisperTranscriber:
    """Lazy, local-only Transformers adapter for a cached Whisper checkpoint."""

    def __init__(self, model: str = "openai/whisper-base.en"):
        self.model = model
        self._pipeline = None
        self.revision = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )
        except ImportError as exc:
            raise LocalTranscriptionError(
                "the optional transformers package is unavailable"
            ) from exc
        try:
            processor = AutoProcessor.from_pretrained(self.model, local_files_only=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model, local_files_only=True
            )
            self.revision = getattr(model.config, "_commit_hash", None)
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise LocalTranscriptionError(
                f"cached local ASR checkpoint {self.model!r} could not be loaded"
            ) from exc
        return self._pipeline

    def __call__(self, path: Path) -> Transcript:
        """Transcribe one local WAV without permitting a model download."""

        recognizer = self._load_pipeline()
        try:
            import numpy

            with wave.open(str(path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                source_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
            waveform = numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32)
            waveform /= 32768.0
            if channels == 2:
                waveform = waveform.reshape((-1, 2)).mean(axis=1)
            target_rate = int(recognizer.feature_extractor.sampling_rate)
            if source_rate != target_rate:
                output_frames = max(1, round(len(waveform) * target_rate / source_rate))
                waveform = numpy.interp(
                    numpy.linspace(0, len(waveform) - 1, output_frames),
                    numpy.arange(len(waveform)),
                    waveform,
                ).astype(numpy.float32)
            result = recognizer({"raw": waveform, "sampling_rate": target_rate})
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            wave.Error,
        ) as exc:
            raise LocalTranscriptionError("local ASR inference failed") from exc
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LocalTranscriptionError("local ASR returned an empty transcript")
        return Transcript(
            text=text.strip(),
            backend="transformers.automatic-speech-recognition",
            model=self.model,
            revision=self.revision,
        )


def _nested_audio_attack(record: Mapping) -> Mapping:
    notes = record.get("notes", {})
    if isinstance(notes, Mapping):
        metadata = notes.get("audio_attack", {})
        if isinstance(metadata, Mapping):
            return metadata
    metadata = record.get("audio_attack", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _first_text(*values) -> str | None:
    return next((value for value in values if isinstance(value, str) and value), None)


def _candidate_inputs(record: Mapping, base_path: Path) -> dict:
    metadata = _nested_audio_attack(record)
    audio = record.get("audio", metadata.get("audio", {}))
    audio = audio if isinstance(audio, Mapping) else {}
    source_case_id = _first_text(
        record.get("source_case_id"),
        record.get("source_id"),
        metadata.get("source_case_id"),
    )
    wav_value = _first_text(
        record.get("wav_path"),
        record.get("audio_path"),
        record.get("path"),
        audio.get("path"),
    )
    expected_text = _first_text(
        record.get("expected_text"),
        record.get("rendered_text"),
        record.get("source_text"),
        metadata.get("rendered_text"),
        metadata.get("source_text"),
    )
    recipe = record.get(
        "candidate_recipe",
        record.get(
            "recipe", record.get("transformations", metadata.get("transformations", []))
        ),
    )
    if wav_value:
        wav_path = Path(wav_value).expanduser()
        if not wav_path.is_absolute():
            wav_path = base_path / wav_path
    else:
        wav_path = None
    return {
        "source_case_id": source_case_id,
        "candidate_id": _first_text(
            record.get("candidate_id"), metadata.get("recipe_digest")
        ),
        "candidate_recipe": recipe,
        "expected_text": expected_text,
        "required_transcript_phrases": record.get("required_transcript_phrases"),
        "wav_path": wav_path,
    }


def validate_candidate(
    record: Mapping,
    *,
    base_path: str | Path = ".",
    transcriber: Callable[[Path], Transcript] | None,
    max_word_error_rate: float = 0.35,
    min_reference_word_recall: float = 0.75,
) -> dict:
    """Validate one candidate, treating unavailable evidence as invalid."""

    candidate = _candidate_inputs(record, Path(base_path))
    wav_path = candidate["wav_path"]
    manifest_candidate = candidate | {
        "wav_path": str(wav_path) if wav_path is not None else None
    }
    validation = {
        "method": "independent_asr_lexical_agreement",
        "audio_valid": False,
        "transcript_available": False,
        "intelligibility_valid": False,
        "semantic_valid": False,
        "scoreable": False,
    }
    if not candidate["source_case_id"]:
        validation["reason"] = "source_case_id_missing"
        return manifest_candidate | {
            "wav": {},
            "transcript": None,
            "validation": validation,
        }
    if wav_path is None:
        validation["reason"] = "wav_path_missing"
        return manifest_candidate | {
            "wav": {},
            "transcript": None,
            "validation": validation,
        }
    wav = inspect_pcm16_wav(wav_path)
    validation["audio_valid"] = wav["valid"]
    if not wav["valid"]:
        validation["reason"] = wav["reason"]
        return manifest_candidate | {
            "wav": wav,
            "transcript": None,
            "validation": validation,
        }
    if not candidate["expected_text"]:
        validation["reason"] = "expected_text_missing"
        return manifest_candidate | {
            "wav": wav,
            "transcript": None,
            "validation": validation,
        }
    if transcriber is None:
        validation["reason"] = "independent_asr_unavailable"
        return manifest_candidate | {
            "wav": wav,
            "transcript": None,
            "validation": validation,
        }
    try:
        transcript = transcriber(wav_path)
    except LocalTranscriptionError as exc:
        validation["reason"] = "independent_asr_failed"
        validation["detail"] = str(exc)
        return manifest_candidate | {
            "wav": wav,
            "transcript": None,
            "validation": validation,
        }
    if not isinstance(transcript, Transcript):
        raise TypeError("transcriber must return a Transcript")
    validation["transcript_available"] = True
    transcript_record = {
        "text": transcript.text,
        "backend": transcript.backend,
        "model": transcript.model,
        "revision": transcript.revision,
    }
    required_phrases = candidate.get("required_transcript_phrases")
    if required_phrases is not None:
        if not isinstance(required_phrases, (list, tuple)) or not all(
            isinstance(phrase, str) and normalized_words(phrase)
            for phrase in required_phrases
        ):
            validation["reason"] = "required_transcript_phrases_invalid"
            return manifest_candidate | {
                "wav": wav,
                "transcript": transcript_record,
                "validation": validation,
            }
        # pad with spaces so phrases match on whole-word boundaries -- otherwise
        # a required "ignore" would be satisfied by the transcript word "ignored"
        padded_transcript = f" {' '.join(normalized_words(transcript.text))} "
        missing_phrases = [
            phrase
            for phrase in required_phrases
            if f" {' '.join(normalized_words(phrase))} " not in padded_transcript
        ]
        passed = not missing_phrases
        validation |= {
            "method": "independent_asr_required_phrase_coverage",
            "required_transcript_phrases": list(required_phrases),
            "missing_transcript_phrases": missing_phrases,
            "intelligibility_valid": passed,
            "semantic_valid": passed,
            "scoreable": passed,
            "reason": (
                "required_transcript_phrases_present"
                if passed
                else "required_transcript_phrases_missing"
            ),
            "semantic_validity_scope": "required lexical phrase coverage proxy",
        }
        return manifest_candidate | {
            "wav": wav,
            "transcript": transcript_record,
            "validation": validation,
        }

    try:
        agreement = transcript_agreement(candidate["expected_text"], transcript.text)
    except ValueError as exc:
        validation["reason"] = "expected_text_has_no_lexical_tokens"
        validation["detail"] = str(exc)
        return manifest_candidate | {
            "wav": wav,
            "transcript": transcript_record,
            "validation": validation,
        }
    validation |= agreement
    passed = (
        agreement["word_error_rate"] <= max_word_error_rate
        and agreement["reference_word_recall"] >= min_reference_word_recall
    )
    validation |= {
        "intelligibility_valid": passed,
        "semantic_valid": passed,
        "scoreable": passed,
        "reason": (
            "transcript_agrees_with_expected_text" if passed else "transcript_mismatch"
        ),
        "semantic_validity_scope": "lexical transcript agreement proxy",
        "thresholds": {
            "max_word_error_rate": max_word_error_rate,
            "min_reference_word_recall": min_reference_word_recall,
        },
    }
    return manifest_candidate | {
        "wav": wav,
        "transcript": transcript_record,
        "validation": validation,
    }


def validate_candidates(
    records: Iterable[Mapping],
    **kwargs,
) -> list[dict]:
    """Validate candidate records in input order."""

    return [validate_candidate(record, **kwargs) for record in records]


def write_jsonl_manifest(records: Iterable[Mapping], output_path: str | Path) -> None:
    """Write a validation manifest atomically as UTF-8 JSON Lines."""

    destination = Path(output_path)
    destination.parent.mkdir(mode=0o740, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                output.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
