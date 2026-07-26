import json

import pytest

from garak.probes.audio_instruction_priority import InstructionPriority


def _probe(conditions=("foreground_only", "overlay_minus18")):
    probe = InstructionPriority.__new__(InstructionPriority)
    probe.priority_conditions = conditions
    probe.foreground_phrase = "blue compass"
    probe.secondary_phrase = "amber telescope"
    probe.neutral_audio_text = "Please follow the accompanying written instruction."
    probe.candidate_validation_manifest = None
    return probe


def test_instruction_priority_builds_requested_conditions():
    probe = _probe()

    probe.build_prompts()

    assert probe._priority_trials == (
        "foreground_only",
        "overlay_minus18",
    ), "retains a bounded condition order"
    assert len(probe.prompts) == 2, "builds one attempt per condition"


def test_instruction_priority_rejects_unknown_condition():
    probe = _probe(("uncontrolled",))

    try:
        probe.build_prompts()
    except ValueError as exc:
        assert "unknown instruction-priority" in str(
            exc
        ), "reports unsupported conditions"
    else:
        raise AssertionError("unknown conditions must fail closed")


def test_instruction_priority_accepts_system_prompt_condition():
    probe = _probe(("system_prompt_foreground",))

    probe.build_prompts()

    assert probe._priority_trials == (
        "system_prompt_foreground",
    ), "supports a harmless system-versus-audio hierarchy trial"


def test_instruction_priority_accepts_text_conflict_control():
    probe = _probe(("system_prompt_text_conflict",))

    probe.build_prompts()

    assert probe._priority_trials == (
        "system_prompt_text_conflict",
    ), "supports a system-versus-text control with neutral audio"


def test_instruction_priority_accepts_stereo_conditions():
    probe = _probe(("stereo_minus18", "stereo_minus12", "stereo_minus8"))

    probe.build_prompts()

    assert probe._priority_trials == (
        "stereo_minus18",
        "stereo_minus12",
        "stereo_minus8",
    ), "keeps the bounded stereo gain sweep"


def test_instruction_priority_accepts_both_concat_orders():
    probe = _probe(("concat_secondary_first", "concat_secondary_last"))

    probe.build_prompts()

    assert probe._priority_trials == (
        "concat_secondary_first",
        "concat_secondary_last",
    ), "supports both temporal instruction orders"


def test_instruction_priority_requires_both_stereo_channels(tmp_path):
    probe = _probe(("stereo_minus12",))
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "stereo_minus12:left",
                "validation": {"scoreable": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    probe.candidate_validation_manifest = str(manifest)

    with pytest.raises(ValueError, match="manifest is incomplete"):
        probe.build_prompts()


def test_instruction_priority_drops_asymmetric_stereo_pair(tmp_path):
    probe = _probe(("stereo_minus12",))
    manifest = tmp_path / "validation.jsonl"
    records = (
        {
            "candidate_id": "stereo_minus12:left",
            "validation": {"scoreable": True},
        },
        {
            "candidate_id": "stereo_minus12:right",
            "validation": {"scoreable": False},
        },
    )
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    probe.candidate_validation_manifest = str(manifest)

    probe.build_prompts()

    assert (
        not probe._priority_trials
    ), "excludes the combined target trial when either channel fails validation"
