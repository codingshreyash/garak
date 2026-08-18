# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import wave

import pytest

from garak.attempt import Attempt, Message
from garak.resources.audio.attack import (
    AudioAttackMetadata,
    attach_audio_attack_metadata,
    audio_file_metadata,
    recipe_digest,
    summarize_audio_groups,
    summarize_audio_records,
)


def _attempt(group_id: str, primary, quality) -> Attempt:
    attempt = Attempt(prompt=Message("test"))
    attempt.outputs = [Message("response")]
    attach_audio_attack_metadata(
        attempt,
        AudioAttackMetadata(
            source_case_id="case.one",
            group_id=group_id,
            source_text="test source",
        ),
    )
    attempt.detector_results = {
        "audio.Primary": [primary],
        "audio.AudioOutputQualityDetector": [quality],
    }
    return attempt


def test_recipe_digest_is_order_independent():
    assert recipe_digest({"b": 2, "a": 1}) == recipe_digest(
        {"a": 1, "b": 2}
    ), "canonical recipes have stable digests"


def test_audio_file_metadata_reads_wav_properties(tmp_path):
    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)

    metadata = audio_file_metadata(audio_path)

    assert metadata["sample_rate"] == 8000, "records the WAV sample rate"
    assert metadata["channels"] == 1, "records the WAV channel count"
    assert metadata["duration_seconds"] == pytest.approx(0.1), "records WAV duration"
    assert len(metadata["sha256"]) == 64, "records the audio checksum"


def test_audio_file_metadata_inspects_extensionless_wav(tmp_path):
    audio_path = tmp_path / "sample"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)

    metadata = audio_file_metadata(audio_path)

    assert metadata["format"] == "wav", "detects WAV content without a suffix"
    assert metadata["sample_rate"] == 8000, "inspects detected WAV properties"


def test_attach_audio_attack_metadata_validates_indices():
    attempt = Attempt(prompt=Message("test"))
    metadata = AudioAttackMetadata(
        source_case_id="case.one",
        group_id="group.one",
        source_text="test source",
        candidate_index=2,
        candidate_count=1,
    )

    with pytest.raises(ValueError, match="candidate_index"):
        attach_audio_attack_metadata(attempt, metadata)


def test_audio_group_summary_excludes_invalid_candidates():
    attempts = [
        _attempt("group.one", 0.0, 0.0),
        _attempt("group.one", 1.0, 1.0),
    ]

    summary = summarize_audio_groups(attempts, "audio.Primary")["group.one"]

    assert summary["scoreable_candidates"] == 1, "counts valid outputs"
    assert summary["invalid_candidates"] == 1, "separates invalid outputs"
    assert (
        summary["group_attack_success"] is False
    ), "invalid detector hits cannot make a group successful"


def test_audio_group_summary_reports_any_scoreable_success():
    attempts = [
        _attempt("group.one", 0.0, 0.0),
        _attempt("group.one", 1.0, 0.0),
    ]

    summary = summarize_audio_groups(attempts, "audio.Primary")["group.one"]

    assert summary["attack_successes"] == 1, "counts primary detector hits"
    assert summary["group_attack_success"] is True, "uses Best-of-N semantics"
    assert summary["status"] == "attack_success", "labels group status"


def test_audio_group_summary_includes_default_threshold_boundary():
    summary = summarize_audio_groups(
        [_attempt("group.one", 0.5, 0.0)], "audio.Primary"
    )["group.one"]

    assert summary["attack_successes"] == 1, "score 0.5 meets the default threshold"
    assert (
        summary["group_attack_success"] is True
    ), "live summary matches JSONL semantics"


def test_audio_group_summary_marks_all_invalid_group_unscored():
    summary = summarize_audio_groups(
        [_attempt("group.one", 1.0, 1.0)], "audio.Primary"
    )["group.one"]

    assert summary["group_attack_success"] is None, "does not pass invalid groups"
    assert summary["status"] == "unscored", "labels all-invalid groups"


def test_record_summary_separates_invalid_and_best_of_n_success():
    records = [
        {
            "entry_type": "attempt",
            "status": 2,
            "uuid": "one",
            "outputs": [{"text": "garbled"}],
            "notes": {"audio_attack": {"group_id": "case-a"}},
            "detector_results": {
                "primary": [0.0],
                "audio.AudioOutputQualityDetector": [1.0],
            },
        },
        {
            "entry_type": "attempt",
            "status": 2,
            "uuid": "two",
            "outputs": [{"text": "compliance"}],
            "notes": {"audio_attack": {"group_id": "case-a"}},
            "detector_results": {
                "primary": [1.0],
                "audio.AudioOutputQualityDetector": [0.0],
            },
        },
    ]

    summary = summarize_audio_records(records, "primary")

    assert (
        summary["invalid_candidate_count"] == 1
    ), "invalid output is counted separately"
    assert (
        summary["candidate_attack_success_rate"] == 1.0
    ), "attack success rate uses scoreable candidates"
    assert (
        summary["groups"]["case-a"]["any_success"] is True
    ), "group records bounded-search success"


def test_wilson_interval_bounds_and_extremes():
    from garak.resources.audio.attack import wilson_interval

    assert wilson_interval(0, 0) == (None, None)
    lo, hi = wilson_interval(0, 10)
    assert (
        lo == 0.0 and 0.0 < hi < 0.35
    ), "all-fail interval hugs zero but is non-trivial"
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.6 < lo < 1.0, "all-success interval hugs one"
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi, "even split brackets the point estimate"
