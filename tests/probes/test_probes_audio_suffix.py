# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from garak.probes.audio_suffix import AudioSuffixInjection


def _bare(**params):
    probe = AudioSuffixInjection.__new__(AudioSuffixInjection)
    probe.carrier_ids = params.get("carrier_ids", ("carrier.capital", "carrier.grass"))
    probe.suffix_ids = params.get(
        "suffix_ids", ("suffix.log_cleanup", "suffix.kubernetes_secrets")
    )
    probe.suffix_gaps_ms = params.get("suffix_gaps_ms", (0, 100))
    probe.suffix_stealth = params.get("suffix_stealth", "plain")
    return probe


def test_combined_recipe_appends_gap_then_suffix():
    suffix = Path("/tmp/suffix.wav")
    # zero gap -> just concatenate the suffix
    assert AudioSuffixInjection._combined_recipe(suffix, 0) == (
        {"type": "concat", "path": str(suffix), "gain_db": 0.0},
    )
    # positive gap -> trailing silence, then concatenate the suffix
    recipe = AudioSuffixInjection._combined_recipe(suffix, 100)
    assert recipe[0] == {"type": "silence", "start_ms": 0, "end_ms": 100}
    assert recipe[1]["type"] == "concat" and recipe[1]["path"] == str(suffix)


def test_stealth_carrier_appends_disguise_transform():
    suffix = Path("/tmp/s.wav")
    # plain: no disguise (default keeps existing behavior)
    assert AudioSuffixInjection._combined_recipe(suffix, 0) == (
        {"type": "concat", "path": str(suffix), "gain_db": 0.0},
    )
    # speed_up: appends a speed transform to the composed clip
    sped = AudioSuffixInjection._combined_recipe(suffix, 100, "speed_up")
    assert sped[-1] == {"type": "speed", "factor": 1.25}
    # noise: hides the clip under white noise
    noised = AudioSuffixInjection._combined_recipe(suffix, 0, "noise")
    assert noised[-1]["type"] == "noise"


def test_unknown_stealth_rejected():
    probe = _bare()
    probe.suffix_stealth = "invisible"
    with pytest.raises(ValueError, match="unknown suffix_stealth"):
        probe.build_prompts()


def test_build_prompts_enumerates_controls_and_combined():
    probe = _bare()  # 2 carriers, 2 suffixes, gaps (0, 100)
    probe.build_prompts()
    trials = probe._selected_suffix_trials
    conditions = [t[0] for t in trials]

    assert conditions.count("benign_only") == 2, "one control per carrier"
    assert conditions.count("suffix_only") == 2, "one baseline per suffix"
    assert conditions.count("combined") == 2 * 2 * 2, "carrier x suffix x gap"
    assert len(trials) == 2 + 2 + 8


def test_combined_source_text_records_gap():
    probe = _bare(suffix_gaps_ms=(500,))
    probe.build_prompts()
    combined = [t for t in probe._selected_suffix_trials if t[0] == "combined"]
    rendered = probe._trial_source_text(combined[0])
    assert "+500ms" in rendered, "rendered text records the injection gap"


def test_only_suffix_bearing_conditions_are_adversarial():
    # benign_only carries no malicious intent; suffix_only and combined do
    probe = _bare()
    probe.build_prompts()
    for condition, *_ in probe._selected_suffix_trials:
        expected = condition != "benign_only"
        assert (condition in ("suffix_only", "combined")) == expected


def test_invalid_ids_and_gaps_are_rejected():
    bad_carrier = _bare(carrier_ids=("carrier.nope",))
    with pytest.raises(ValueError, match="unknown suffix carrier ids"):
        bad_carrier.build_prompts()

    bad_gap = _bare(suffix_gaps_ms=(-5,))
    with pytest.raises(ValueError, match="non-negative"):
        bad_gap.build_prompts()


def test_probe_is_discoverable_as_plugin():
    from garak._plugins import plugin_info

    info = plugin_info("probes.audio_suffix.AudioSuffixInjection")
    assert "audio.AudioToolRiskJudge" in info["extended_detectors"]
    # the plugin cache normalizes tuples to JSON lists
    assert list(info["DEFAULT_PARAMS"]["suffix_gaps_ms"]) == [0, 100, 500]
