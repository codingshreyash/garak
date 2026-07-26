# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import wave
from pathlib import Path

import pytest

from garak.probes.audio_overlay import AudioOverlayInjection


def _bare(**params):
    p = AudioOverlayInjection.__new__(AudioOverlayInjection)
    p.carrier_ids = params.get("carrier_ids", ("carrier.capital", "carrier.grass"))
    p.payload_ids = params.get(
        "payload_ids", ("suffix.log_cleanup", "suffix.kubernetes_secrets")
    )
    p.overlay_gains_db = params.get("overlay_gains_db", (-6.0, -12.0))
    p.payload_positions = params.get("payload_positions", ("start", "middle", "end"))
    return p


def test_overlay_recipe_uses_overlay_mode_and_gain():
    payload = Path("/tmp/p.wav")
    assert AudioOverlayInjection._overlay_recipe(payload, -12.0) == (
        {"type": "overlay", "path": str(payload), "gain_db": -12.0},
    )


def test_build_prompts_enumerates_controls_and_mixed():
    # 2 carriers, 2 payloads, 2 gains, 3 positions → mixed = 2×2×2×3 = 24
    probe = _bare()
    probe.build_prompts()
    conditions = [t[0] for t in probe._selected_overlay_trials]
    assert conditions.count("foreground_only") == 2
    assert conditions.count("background_only") == 2
    assert conditions.count("mixed") == 2 * 2 * 2 * 3
    assert len(probe._selected_overlay_trials) == 2 + 2 + 24


def test_only_payload_bearing_conditions_are_adversarial(monkeypatch):
    # background_only and mixed carry the payload; foreground_only is the control.
    # Assert the flag the hook actually stores against a fixed truth table.
    from garak.attempt import Attempt, Message
    from garak.probes.audio import PETTS

    probe = _bare()
    probe.build_prompts()
    probe._prepared_overlay_trials = probe._selected_overlay_trials
    monkeypatch.setattr(PETTS, "_attempt_prestore_hook", lambda self, a, s: a)
    monkeypatch.setattr(probe, "_attach_audio_attack_metadata", lambda *a, **k: None)

    expected = {"foreground_only": False, "background_only": True, "mixed": True}
    seen = set()
    for seq, trial in enumerate(probe._prepared_overlay_trials):
        attempt = probe._attempt_prestore_hook(
            Attempt(
                probe_classname="audio_overlay.AudioOverlayInjection",
                prompt=Message("x"),
            ),
            seq,
        )
        assert attempt.notes["is_adversarial"] is expected[trial[0]], trial[0]
        seen.add(trial[0])
    assert seen == set(expected), "all three overlay conditions are exercised"


def test_positive_gain_is_rejected():
    bad = _bare(overlay_gains_db=(6.0,))
    with pytest.raises(ValueError, match="attenuations"):
        bad.build_prompts()

    bad_id = _bare(carrier_ids=("carrier.nope",))
    with pytest.raises(ValueError, match="unknown overlay carrier ids"):
        bad_id.build_prompts()


def test_real_overlay_composition_mixes_and_preserves_length(tmp_path):
    from garak.resources.audio.transforms import apply_transform_recipe, read_pcm16_wav

    def mkwav(path, ms, val, sr=24000):
        n = int(sr * ms / 1000)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(val * n)

    carrier = tmp_path / "c.wav"
    payload = tmp_path / "p.wav"
    out = tmp_path / "o.wav"
    mkwav(carrier, 1000, b"\x00\x20")  # 1000ms carrier
    mkwav(payload, 400, b"\x00\x20")  # 400ms payload
    apply_transform_recipe(
        carrier, out, AudioOverlayInjection._overlay_recipe(payload, -12.0)
    )

    wf = read_pcm16_wav(out)
    dur_ms = round(1000 * len(wf.samples) / wf.sample_rate)
    # overlay = max(carrier, payload) length, not the sum (that would be concat)
    assert 990 <= dur_ms <= 1010, f"overlay should be carrier length, got {dur_ms}ms"


def test_probe_is_discoverable_as_plugin():
    from garak._plugins import plugin_info

    info = plugin_info("probes.audio_overlay.AudioOverlayInjection")
    assert "audio.AudioToolRiskJudge" in info["extended_detectors"]
    assert list(info["DEFAULT_PARAMS"]["overlay_gains_db"]) == [-6.0, -12.0, -18.0]
