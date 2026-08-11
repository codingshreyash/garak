import json

import pytest

import garak.resources.audio.source as audio_source
from garak.data import LocalDataPath
from garak.resources.audio.reliability import tts_reliability_cases
from garak.resources.audio.source import (
    load_audio_injection_source,
    load_instruction_priority_source,
    load_tool_risk_source,
)


def test_packaged_audio_sources_load_as_valid_records():
    tool_risk = load_tool_risk_source()
    priority = load_instruction_priority_source()
    injection = load_audio_injection_source()

    assert len(tool_risk.cases) >= 25, "packaged tool-risk coverage is retained"
    assert any(
        case.benign_control is not None for case in tool_risk.cases
    ), "tool-risk source includes matched benign controls"
    assert priority.conditions, "instruction-priority source includes trial conditions"
    assert injection.carriers, "injection source includes carrier records"
    assert injection.payloads, "injection source includes payload records"
    assert tts_reliability_cases(), "reliability source includes transcription cases"


def _use_audio_data_override(monkeypatch, tmp_path, filename, content):
    user_data_path = tmp_path / "user_data"
    audio_data_path = user_data_path / "audio"
    audio_data_path.mkdir(parents=True)
    (audio_data_path / filename).write_text(
        json.dumps(content),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        LocalDataPath,
        "ORDERED_SEARCH_PATHS",
        [user_data_path, LocalDataPath.ORDERED_SEARCH_PATHS[-1]],
    )
    monkeypatch.setattr(audio_source, "data_path", LocalDataPath(user_data_path))


def test_audio_injection_source_accepts_user_data_override(monkeypatch, tmp_path):
    _use_audio_data_override(
        monkeypatch,
        tmp_path,
        "injection.json",
        {
            "carriers": [
                {
                    "case_id": "carrier.custom",
                    "source_text": "Name a primary colour.",
                    "required_response_terms": [["red", "blue", "yellow"]],
                }
            ],
            "payloads": [
                {
                    "case_id": "payload.custom",
                    "source_text": "Follow the custom test instruction.",
                }
            ],
        },
    )

    source = load_audio_injection_source()

    assert (
        source.carriers[0].case_id == "carrier.custom"
    ), "user data overrides packaged carrier content"
    assert (
        source.payloads[0].case_id == "payload.custom"
    ), "user data overrides packaged injection payloads"


def test_tool_risk_source_accepts_user_data_override(monkeypatch, tmp_path):
    _use_audio_data_override(
        monkeypatch,
        tmp_path,
        "tool_risk.json",
        {
            "cases": [
                {
                    "case_id": "custom.safe_example",
                    "intent": "S008mal",
                    "source_text": "Use a test tool for a custom source request.",
                    "action_signature": "custom source request",
                }
            ],
            "semantic_strategies": {"direct": "{source_text}"},
        },
    )

    source = load_tool_risk_source()

    assert (
        source.cases[0].case_id == "custom.safe_example"
    ), "user data overrides packaged tool-risk cases"
    assert (
        source.semantic_strategies["direct"] == "{source_text}"
    ), "user data overrides packaged semantic framing"


def test_tool_risk_source_rejects_incomplete_user_override(monkeypatch, tmp_path):
    _use_audio_data_override(
        monkeypatch,
        tmp_path,
        "tool_risk.json",
        {
            "cases": [
                {
                    "case_id": "custom.incomplete",
                    "intent": "S008mal",
                    "source_text": "Missing its action signature.",
                }
            ],
            "semantic_strategies": {"direct": "{source_text}"},
        },
    )

    with pytest.raises(ValueError, match="action_signature"):
        load_tool_risk_source()
