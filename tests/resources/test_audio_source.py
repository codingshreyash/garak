import json

import pytest

from garak.resources.audio.reliability import tts_reliability_cases
from garak.resources.audio.source import (
    load_audio_injection_source,
    load_instruction_priority_source,
    load_tool_risk_source,
)


def test_packaged_audio_sources_load_as_valid_records():
    tool_risk = load_tool_risk_source()
    priority = load_instruction_priority_source()

    assert len(tool_risk.cases) >= 25, "packaged tool-risk coverage is retained"
    assert any(
        case.benign_control is not None for case in tool_risk.cases
    ), "tool-risk source includes matched benign controls"
    assert priority.conditions, "instruction-priority source includes trial conditions"
    assert tts_reliability_cases(), "reliability source includes transcription cases"


def test_audio_injection_source_accepts_user_provided_content(tmp_path):
    source_path = tmp_path / "injection.json"
    source_path.write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )

    source = load_audio_injection_source(source_path)

    assert (
        source.carriers[0].case_id == "carrier.custom"
    ), "custom source replaces packaged carrier content"
    assert (
        source.payloads[0].case_id == "payload.custom"
    ), "custom source replaces packaged injection payloads"


def test_tool_risk_source_accepts_user_provided_content(tmp_path):
    source_path = tmp_path / "tool_risk.json"
    source_path.write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )

    source = load_tool_risk_source(source_path)

    assert (
        source.cases[0].case_id == "custom.safe_example"
    ), "custom source data replaces packaged tool-risk cases"
    assert (
        source.semantic_strategies["direct"] == "{source_text}"
    ), "custom source data controls semantic framing"


def test_tool_risk_source_rejects_incomplete_records(tmp_path):
    source_path = tmp_path / "tool_risk.json"
    source_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "custom.incomplete",
                        "intent": "S008mal",
                        "source_text": "Missing its action signature.",
                    }
                ],
                "semantic_strategies": {"direct": "{source_text}"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_signature"):
        load_tool_risk_source(source_path)
