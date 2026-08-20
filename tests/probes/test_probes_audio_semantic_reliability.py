# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tempfile

from garak import _config, _plugins
import garak.probes.audio_reliability


def _load_semantic_reliability_probe() -> (
    garak.probes.audio_reliability.SemanticReliability
):
    _config.load_config()
    return garak.probes.audio_reliability.SemanticReliability()


def _configure_probe_for_test(monkeypatch, tmp_path, probe):
    monkeypatch.setattr(_config.transient, "cache_dir", tmp_path)
    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    monkeypatch.setattr(_config.transient, "reportfile", temp_report_file)
    monkeypatch.setattr(_config.transient, "report_filename", temp_report_file.name)
    probe.tts_model_name = "test-tts-model"
    probe.audio_cache_dir = probe._audio_cache_dir()
    probe.audio_cache_dir.mkdir(mode=0o740, parents=True, exist_ok=True)
    monkeypatch.setattr(
        probe,
        "_synthesise_audio",
        lambda prompt_text, audio_path: audio_path.write_bytes(b"RIFF....WAVEfmt "),
    )
    return temp_report_file


def test_semantic_reliability_cases_cover_simple_comprehension():
    cases = garak.probes.audio_reliability.semantic_reliability_cases()

    assert len(cases) >= 3
    assert any(case.case_id == "semantic_reliability.clear_sky" for case in cases)
    assert all(case.required_response_terms for case in cases)


def test_semantic_reliability_builds_precursor_prompts():
    probe = _load_semantic_reliability_probe()

    assert probe.primary_detector == "audio.SemanticReliabilityDetector"
    assert "audio.AudioOutputQualityDetector" in probe.extended_detectors
    assert probe.prompts == [
        case.source_text for case in probe._semantic_reliability_cases()
    ]
    assert probe.prompt_intents == [
        case.case_id for case in probe._semantic_reliability_cases()
    ]


def test_semantic_reliability_records_case_metadata(monkeypatch, tmp_path):
    probe = _load_semantic_reliability_probe()
    temp_report_file = _configure_probe_for_test(monkeypatch, tmp_path, probe)
    generator = _plugins.load_plugin("generators.test.Repeat")
    monkeypatch.setattr(generator, "modality", {"in": {"text", "audio"}})

    try:
        attempts = probe.probe(generator)
    finally:
        temp_report_file.close()

    assert len(attempts) == len(garak.probes.audio_reliability.semantic_reliability_cases())
    sky_attempt = next(
        attempt
        for attempt in attempts
        if attempt.intent == "semantic_reliability.clear_sky"
    )
    metadata = sky_attempt.notes["audio_semantic_reliability"]
    prompt = sky_attempt.prompt.last_message("user")

    assert prompt.text == probe.text_prompt
    assert prompt.data_path is not None
    assert Path(prompt.data_path).is_file()
    assert "clear daytime sky" in metadata["source_text"]
    assert ["blue"] in metadata["required_response_terms"]
    assert metadata["trial_index"] == 1
    assert metadata["trial_count"] == 1


def test_semantic_reliability_rejects_unknown_case_id():
    probe = _load_semantic_reliability_probe()
    probe.semantic_reliability_case_ids = ("semantic_reliability.missing",)

    try:
        probe.build_prompts()
    except ValueError as exc:
        assert "unknown semantic reliability case IDs" in str(exc)
    else:
        raise AssertionError("unknown case IDs should raise ValueError")
