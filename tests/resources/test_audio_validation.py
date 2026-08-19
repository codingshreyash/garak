# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import wave

import pytest

from garak.resources.audio.validation import (
    LocalTranscriptionError,
    Transcript,
    TransformersWhisperTranscriber,
    inspect_pcm16_wav,
    transcript_agreement,
    validate_candidate,
    write_jsonl_manifest,
)


def _write_wav(path, *, frames=b"\x00\x10" * 800, channels=1):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(frames)


def _record(path):
    return {
        "source_case_id": "case.one",
        "candidate_id": "clean",
        "candidate_recipe": [],
        "wav_path": str(path),
        "expected_text": "Check the stock price for Nvidia today.",
    }


def test_transcript_agreement_ignores_case_and_punctuation():
    agreement = transcript_agreement(
        "Check NVIDIA's stock price.", "check nvidia's STOCK price"
    )

    assert (
        agreement["word_error_rate"] == 0.0
    ), "normalization ignores case and punctuation"
    assert (
        agreement["reference_word_recall"] == 1.0
    ), "all normalized reference words are retained"


def test_pcm16_inspection_rejects_silent_audio(tmp_path):
    path = tmp_path / "silent.wav"
    _write_wav(path, frames=b"\x00\x00" * 800)

    result = inspect_pcm16_wav(path)

    assert result["valid"] is False, "silent WAV is not intelligible evidence"
    assert (
        result["reason"] == "wav_signal_is_silent"
    ), "silent signal has a stable failure reason"


def test_candidate_validation_records_provenance_and_transcript(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)

    result = validate_candidate(
        _record(path),
        transcriber=lambda _: Transcript(
            "Check the stock price for Nvidia today.",
            backend="test-asr",
            model="fixture",
            revision="one",
        ),
    )

    assert (
        result["validation"]["scoreable"] is True
    ), "matching independent transcript makes valid WAV scoreable"
    assert len(result["wav"]["sha256"]) == 64, "manifest records WAV checksum"
    assert (
        result["transcript"]["backend"] == "test-asr"
    ), "manifest identifies transcript provenance"
    assert result["candidate_recipe"] == [], "manifest links candidate recipe"
    assert isinstance(result["wav_path"], str), "manifest remains JSON serializable"


def test_candidate_validation_fails_closed_without_asr(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)

    result = validate_candidate(_record(path), transcriber=None)

    assert (
        result["validation"]["scoreable"] is False
    ), "missing ASR cannot silently validate a candidate"
    assert (
        result["validation"]["reason"] == "independent_asr_unavailable"
    ), "manifest explains unavailable ASR"


def test_candidate_validation_skips_asr_for_malformed_wav(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a wav")
    called = False

    def transcribe(_):
        nonlocal called
        called = True
        return Transcript("text", backend="test-asr")

    result = validate_candidate(_record(path), transcriber=transcribe)

    assert called is False, "malformed WAV is rejected before ASR inference"
    assert (
        result["validation"]["scoreable"] is False
    ), "malformed WAV cannot enter target scoring"
    assert (
        result["validation"]["reason"] == "wav_parse_error"
    ), "malformed WAV has a stable failure reason"


def test_candidate_validation_rejects_transcript_mismatch(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)

    result = validate_candidate(
        _record(path),
        transcriber=lambda _: Transcript("Completely unrelated words", "test-asr"),
    )

    assert (
        result["validation"]["semantic_valid"] is False
    ), "lexically unrelated transcript is not semantically validated"
    assert (
        result["validation"]["reason"] == "transcript_mismatch"
    ), "manifest identifies transcript disagreement"


def test_candidate_validation_accepts_required_competing_phrases(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)
    record = _record(path) | {
        "required_transcript_phrases": ["blue compass", "amber telescope"]
    }

    result = validate_candidate(
        record,
        transcriber=lambda _: Transcript(
            "Amber telescope, followed by blue compass.", "test-asr"
        ),
    )

    assert (
        result["validation"]["scoreable"] is True
    ), "multi-speaker validation accepts both required canaries in either order"
    assert result["validation"]["method"] == (
        "independent_asr_required_phrase_coverage"
    ), "manifest records phrase-coverage validation"


def test_candidate_validation_matches_required_phrases_on_word_boundaries(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)
    record = _record(path) | {"required_transcript_phrases": ["art"]}

    result = validate_candidate(
        record,
        transcriber=lambda _: Transcript("A partial transcript", "test-asr"),
    )

    assert (
        result["validation"]["scoreable"] is False
    ), "partial-word matches cannot validate required phrases"
    assert result["validation"]["missing_transcript_phrases"] == [
        "art"
    ], "manifest records the unmatched whole-word phrase"


def test_candidate_validation_rejects_empty_required_phrase_list(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)
    record = _record(path) | {"required_transcript_phrases": []}

    result = validate_candidate(
        record,
        transcriber=lambda _: Transcript("unrelated", "test-asr"),
    )

    assert (
        result["validation"]["scoreable"] is False
    ), "empty phrase requirements cannot bypass transcript agreement"
    assert (
        result["validation"]["reason"] == "required_transcript_phrases_invalid"
    ), "manifest identifies the invalid phrase contract"


def test_candidate_validation_records_asr_failure(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)

    def fail(_):
        raise LocalTranscriptionError("fixture backend failed")

    result = validate_candidate(_record(path), transcriber=fail)

    assert (
        result["validation"]["scoreable"] is False
    ), "ASR error cannot silently validate a candidate"
    assert (
        result["validation"]["reason"] == "independent_asr_failed"
    ), "ASR errors have a stable failure reason"


def test_jsonl_manifest_is_written_in_input_order(tmp_path):
    output = tmp_path / "manifest.jsonl"
    write_jsonl_manifest(
        ({"source_case_id": source_id} for source_id in ("one", "two")), output
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["source_case_id"] for record in records] == [
        "one",
        "two",
    ], "manifest preserves candidate ordering"


def test_transformers_transcriber_decodes_wav_without_ffmpeg(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_wav(path)
    received = None

    class Recognizer:
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

        def __call__(self, value):
            nonlocal received
            received = value
            return {"text": "local transcript"}

    transcriber = TransformersWhisperTranscriber("fixture")
    transcriber._pipeline = Recognizer()
    result = transcriber(path)

    assert isinstance(received, dict), "ASR receives decoded samples, not a filename"
    assert received["sampling_rate"] == 16000, "WAV is resampled for the ASR frontend"
    assert result.text == "local transcript", "ASR transcript is preserved"


@pytest.mark.parametrize("channels", [1, 2])
def test_pcm16_inspection_accepts_supported_channel_counts(tmp_path, channels):
    path = tmp_path / f"{channels}.wav"
    _write_wav(path, frames=b"\x00\x10" * 800 * channels, channels=channels)

    assert (
        inspect_pcm16_wav(path)["valid"] is True
    ), "mono and stereo PCM16 candidates are structurally valid"


def test_pcm16_inspection_uses_content_for_extensionless_wav(tmp_path):
    path = tmp_path / "candidate"
    _write_wav(path)

    result = inspect_pcm16_wav(path)

    assert result["valid"] is True, "valid WAV content does not require a suffix"
    assert result["format"] == "wav", "inspection records the detected WAV format"
