"""Tests for spoken semantic reliability probing."""

import json

import pytest

from garak.attempt import Conversation, Message, Turn
from garak.probes.audio_reliability import Semantic


def test_semantic_reliability_builds_packaged_cases(loaded_intent_service):
    """Packaged cases remain aligned with their detector metadata."""

    probe = Semantic()

    assert len(probe.prompts) == 4, "the packaged reliability set must be loaded"
    assert len(probe.prompts) == len(
        probe.prompt_intents
    ), "prompts and case identifiers must remain aligned"


def test_semantic_reliability_mints_case_metadata(mocker, loaded_intent_service):
    """Each attempt carries the expectations for its selected case."""

    probe = Semantic()
    prepared = Conversation([Turn("user", Message("prepared audio"))])
    mocker.patch.object(probe, "_prepare_prompt", return_value=prepared)

    attempt = probe._mint_attempt(probe.prompts[0], seq=0)

    assert (
        attempt.intent == "clear_sky"
    ), "the attempt intent must identify the reliability case"
    assert attempt.notes["audio_reliability"]["required_terms"] == [
        ["blue"]
    ], "the detector expectations must follow the selected case"


def test_semantic_reliability_supports_source_override(tmp_path, loaded_intent_service):
    """Users can replace packaged reliability cases with validated JSON."""

    source = tmp_path / "cases.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "custom",
                    "prompt": "Say custom.",
                    "required_terms": [["custom"]],
                    "forbidden_terms": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "probes": {"audio_reliability": {"Semantic": {"source_data_path": str(source)}}}
    }

    probe = Semantic(config_root=config)

    assert probe.prompts == ["Say custom."], "the configured data source must be used"
    assert probe.prompt_intents == ["custom"], "custom case IDs must be retained"


def test_semantic_reliability_rejects_duplicate_ids(tmp_path, loaded_intent_service):
    """Ambiguous case identifiers fail during probe construction."""

    source = tmp_path / "cases.json"
    case = {
        "id": "duplicate",
        "prompt": "Say duplicate.",
        "required_terms": [["duplicate"]],
        "forbidden_terms": [],
    }
    source.write_text(json.dumps([case, case]), encoding="utf-8")
    config = {
        "probes": {"audio_reliability": {"Semantic": {"source_data_path": str(source)}}}
    }

    with pytest.raises(ValueError, match="duplicate audio reliability case ID"):
        Semantic(config_root=config)
