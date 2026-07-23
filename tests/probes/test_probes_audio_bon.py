import json
from pathlib import Path

import pytest

from garak.probes.audio_bon import PairedDirect


def _probe(candidate_names=("clean", "speed_1_12")) -> PairedDirect:
    probe = PairedDirect.__new__(PairedDirect)
    probe.tool_risk_case_ids = ("python.reverse_shell.direct",)
    probe.tool_risk_case_limit = None
    probe.candidate_names = candidate_names
    probe.synthesis_condition = None
    probe.candidate_validation_manifest = None
    return probe


def test_paired_direct_attack_goal_only_on_harmful(monkeypatch):
    """Benign trials must not carry attack_goal (matches audio_acoustic)."""
    from garak.attempt import Attempt, Message
    from garak.probes.audio import PETTS

    probe = _probe()
    probe._candidate_validations = {}
    probe._prepared_bon_trials = [
        (
            "benign",
            "p1",
            "c.benign",
            "T000",
            "benign text",
            "clean",
            0,
            1,
            (),
            (("ok",),),
        ),
        ("harmful", "p1", "c.direct", "T000", "harmful text", "clean", 0, 1, (), ()),
    ]
    monkeypatch.setattr(PETTS, "_attempt_prestore_hook", lambda self, a, s: a)
    monkeypatch.setattr(probe, "_attach_audio_attack_metadata", lambda *a, **k: None)

    benign = probe._attempt_prestore_hook(
        Attempt(probe_classname="audio_bon.PairedDirect", prompt=Message("x")), 0
    )
    harmful = probe._attempt_prestore_hook(
        Attempt(probe_classname="audio_bon.PairedDirect", prompt=Message("x")), 1
    )

    assert "attack_goal" not in benign.notes, "benign trials must not carry attack_goal"
    assert (
        harmful.notes["attack_goal"] == "harmful text"
    ), "harmful trials must retain their attack goal"
    assert (
        benign.notes["is_adversarial"] is False
    ), "benign controls must not be marked adversarial"
    assert (
        harmful.notes["is_adversarial"] is True
    ), "harmful candidates must be marked adversarial"


def test_paired_direct_builds_matched_bounded_candidates():
    probe = _probe()

    probe.build_prompts()

    assert (
        len(probe._selected_bon_trials) == 4
    ), "each harmful and benign request receives every candidate"
    assert {trial[0] for trial in probe._selected_bon_trials} == {
        "harmful",
        "benign",
    }, "retains the paired experimental condition"
    assert {trial[5] for trial in probe._selected_bon_trials} == {
        "clean",
        "speed_1_12",
    }, "records a stable candidate label"
    assert {trial[7] for trial in probe._selected_bon_trials} == {
        2
    }, "records the bounded group size"
    assert all(
        trial[9] for trial in probe._selected_bon_trials if trial[0] == "benign"
    ), "benign candidates preserve semantic expectations"


def test_paired_direct_rejects_unknown_candidate():
    probe = _probe(("uncontrolled_voice",))

    with pytest.raises(ValueError, match="unknown audio candidate"):
        probe.build_prompts()


@pytest.mark.parametrize(
    ("candidate_name", "factor"),
    (
        ("speed_0_90", 0.90),
        ("speed_1_05", 1.05),
        ("speed_1_12", 1.12),
        ("speed_1_18", 1.18),
        ("speed_1_25", 1.25),
    ),
)
def test_paired_direct_speed_candidates_use_named_factors(candidate_name, factor):
    probe = _probe((candidate_name,))

    candidates = probe._selected_candidates()

    assert candidates == (
        (candidate_name, ({"type": "speed", "factor": factor},)),
    ), "named speed candidates preserve the configured dose"


def test_paired_direct_labels_a_second_synthesis_system():
    probe = _probe(("clean",))
    probe.synthesis_condition = "mms_tts"

    probe.build_prompts()

    assert {trial[5] for trial in probe._selected_bon_trials} == {
        "mms_tts:clean"
    }, "distinguishes a second TTS system from the baseline waveform"


def test_paired_direct_rejects_empty_synthesis_label():
    probe = _probe(("clean",))
    probe.synthesis_condition = " "

    with pytest.raises(ValueError, match="synthesis_condition"):
        probe.build_prompts()


def test_paired_direct_clean_candidate_reuses_source(monkeypatch, tmp_path):
    probe = _probe(("clean",))
    source = tmp_path / "source.wav"
    monkeypatch.setattr(probe, "_ensure_audio_file", lambda _: source)

    assert (
        probe._ensure_candidate_audio("request", ()) == source
    ), "clean control does not rewrite the waveform"


def test_paired_direct_transform_cache_depends_on_recipe(tmp_path):
    probe = _probe()
    probe.audio_cache_dir = tmp_path
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")

    speed_path = probe._transformed_audio_path(
        source, ({"type": "speed", "factor": 1.12},)
    )
    noise_path = probe._transformed_audio_path(
        source, ({"type": "noise", "kind": "white", "snr_db": 22},)
    )

    assert isinstance(speed_path, Path), "returns a filesystem cache path"
    assert speed_path != noise_path, "different recipes use different cache identities"


def test_paired_direct_filters_unscoreable_candidates(tmp_path):
    probe = _probe()
    probe.build_prompts()
    records = []
    for trial in probe._selected_bon_trials:
        candidate_id = f"{trial[0]}:{trial[1]}:{trial[5]}"
        records.append(
            {
                "candidate_id": candidate_id,
                "validation": {"scoreable": trial[5] == "clean"},
            }
        )
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    probe.candidate_validation_manifest = str(manifest)

    probe.build_prompts()

    assert (
        len(probe._selected_bon_trials) == 2
    ), "drops invalid audio before target calls"
    assert all(
        trial[5] == "clean" for trial in probe._selected_bon_trials
    ), "retains only independently scoreable candidates"
    assert all(
        trial[6:8] == (1, 1) for trial in probe._selected_bon_trials
    ), "reindexes each bounded group after validation"


def test_paired_direct_rejects_incomplete_validation_manifest(tmp_path):
    probe = _probe()
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "harmful:python.reverse_shell:clean",
                "validation": {"scoreable": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    probe.candidate_validation_manifest = str(manifest)

    with pytest.raises(ValueError, match="manifest is incomplete"):
        probe.build_prompts()


def test_paired_direct_drops_asymmetric_candidate_pair(tmp_path):
    probe = _probe(("clean",))
    probe.build_prompts()
    records = [
        {
            "candidate_id": f"{trial[0]}:{trial[1]}:{trial[5]}",
            "validation": {"scoreable": trial[0] == "harmful"},
        }
        for trial in probe._selected_bon_trials
    ]
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    probe.candidate_validation_manifest = str(manifest)

    probe.build_prompts()

    assert (
        not probe._selected_bon_trials
    ), "a candidate is excluded from both conditions when either matched WAV fails"
