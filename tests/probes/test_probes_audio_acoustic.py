# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from garak.probes.audio_acoustic import AcousticVoiceBestOfN
from garak.resources.audio.synthesis import (
    SynthesisRequest,
    TransformersSynthesisProvider,
    validate_synthesis_request,
)

# ---- synthesis provider: voice is forwarded, not dropped ----


def test_provider_advertises_and_validates_voices():
    prov = TransformersSynthesisProvider("suno/bark-small", voices=("v2/en_speaker_0",))
    assert prov.capabilities().voices == ("v2/en_speaker_0",)
    validate_synthesis_request(
        SynthesisRequest("hi", voice="v2/en_speaker_0"), prov.capabilities()
    )
    with pytest.raises(ValueError, match="does not support voice"):
        validate_synthesis_request(
            SynthesisRequest("hi", voice="v2/en_speaker_1"), prov.capabilities()
        )


def test_provider_forwards_voice_to_pipeline(mocker):
    prov = TransformersSynthesisProvider("suno/bark-small", voices=("v2/en_speaker_6",))
    fake_pipe = mocker.MagicMock(
        return_value={"audio": [0.0, 0.1], "sampling_rate": 24000}
    )
    mocker.patch.object(prov, "_load_pipeline", return_value=fake_pipe)

    result = prov.synthesize(
        SynthesisRequest("go", voice="v2/en_speaker_6", sample_rate=24000)
    )

    # the voice preset must reach the model, not be silently dropped
    _, kwargs = fake_pipe.call_args
    assert kwargs["forward_params"] == {"history_prompt": "v2/en_speaker_6"}
    assert result.effective_options == {"voice": "v2/en_speaker_6"}


def test_provider_omits_forward_params_when_no_voice(mocker):
    prov = TransformersSynthesisProvider("suno/bark-small")
    fake_pipe = mocker.MagicMock(return_value={"audio": [0.0], "sampling_rate": 24000})
    mocker.patch.object(prov, "_load_pipeline", return_value=fake_pipe)
    prov.synthesize(SynthesisRequest("go", sample_rate=24000))
    _, kwargs = fake_pipe.call_args
    assert "forward_params" not in kwargs


# ---- probe: enumerates one trial per voice and groups Best-of-N ----


def _bare(voices=("v2/en_speaker_0", "v2/en_speaker_6")):
    p = AcousticVoiceBestOfN.__new__(AcousticVoiceBestOfN)
    p.tool_risk_case_ids = ("bash.exfil_s3.direct", "bash.kubernetes_secrets.direct")
    p.tool_risk_case_limit = None
    p.semantic_strategies = ("direct",)
    p.voice_candidates = voices
    return p


def test_build_prompts_one_trial_per_voice_and_condition():
    probe = _bare()  # 2 cases x 2 conditions x 2 voices
    probe.build_prompts()
    trials = probe._selected_voice_trials
    assert len(trials) == 2 * 2 * 2
    assert {t[5] for t in trials} == {"v2/en_speaker_0", "v2/en_speaker_6"}
    # candidate_count reflects the number of voices (for Best-of-N grouping)
    assert all(t[7] == 2 for t in trials)


def test_empty_voice_list_rejected():
    probe = _bare(voices=())
    with pytest.raises(ValueError, match="at least one voice"):
        probe.build_prompts()


# ---- cache key: different voices must not collide ----


def test_audio_cache_key_differs_per_voice():
    probe = _bare()
    probe.tts_model_name = "suno/bark-small"
    probe.tts_model_revision = None
    probe.tts_sample_rate = 24000
    probe.tts_audio_format = "WAV"
    probe.tts_audio_subtype = "PCM_16"
    probe.tts_audio_stereo = False
    from pathlib import Path

    probe.audio_cache_dir = Path("/tmp")

    probe.tts_voice = "v2/en_speaker_0"
    path_a = probe._audio_file_path("list secrets")
    probe.tts_voice = "v2/en_speaker_6"
    path_b = probe._audio_file_path("list secrets")
    assert path_a != path_b, "different voices must cache to different files"


def test_probe_is_discoverable_as_plugin():
    from garak._plugins import plugin_info

    info = plugin_info("probes.audio_acoustic.AcousticVoiceBestOfN")
    assert list(info["DEFAULT_PARAMS"]["voice_candidates"]) == [
        "v2/en_speaker_0",
        "v2/en_speaker_6",
        "v2/en_speaker_9",
    ]
