# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural and unit tests for audio_duplex probes.

No live endpoint or TTS model is required — all audio synthesis and generator
calls are monkeypatched.
"""

import io
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from garak import _config
import garak.probes.audio_duplex as _dp
from garak.probes.audio_duplex import (
    AudioSilenceSmuggling,
    PETTSBargeFlood,
    PETTSBargeInterrupt,
    PETTSFalseBargeIn,
    PETTSDriftPoison,
)
from garak.resources.audio.session import SessionScript, SessionEvent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_ms: int = 100) -> bytes:
    n = int(16000 * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


class _FakeDuplexGenerator:
    """Minimal DuplexCapable stub that always returns a benign response."""

    supports_duplex = True
    modality = {"in": {"audio", "text"}, "out": {"text"}}
    audio_formats = {"wav"}

    def run_session(self, script):
        from garak.resources.audio.session import SessionResult, SessionResultEvent

        result = SessionResult(
            full_transcript="I cannot help with that.",
            pre_interrupt_transcript="I cannot help with that.",
            post_interrupt_transcript="I cannot help with that.",
        )
        return result


class _FakeNonDuplexGenerator:
    modality = {"in": {"audio", "text"}, "out": {"text"}}
    audio_formats = {"wav"}
    # Does NOT inherit DuplexCapable


@pytest.fixture()
def probe_env(monkeypatch, tmp_path):
    """Set up _config for probe tests and patch TTS synthesis."""
    _config.load_config()
    monkeypatch.setattr(_config.transient, "cache_dir", tmp_path)

    import tempfile

    report = tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8")
    monkeypatch.setattr(_config.transient, "reportfile", report)
    monkeypatch.setattr(_config.transient, "report_filename", report.name)
    yield tmp_path


# ---------------------------------------------------------------------------
# Session-script helpers
# ---------------------------------------------------------------------------


class TestMakeSilenceWav:
    def test_produces_valid_wav(self):
        data = _dp._make_silence_wav(100)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000


# ---------------------------------------------------------------------------
# PETTSBargeInterrupt
# ---------------------------------------------------------------------------


class TestPETTSBargeInterrupt:
    def _make_probe(self, tmp_path, monkeypatch):
        from garak.probes.audio import _spoken_prompt_candidate

        monkeypatch.setattr(
            "garak.probes.audio._spoken_prompt_candidate", lambda _: True
        )
        probe = PETTSBargeInterrupt()
        probe.tts_model_name = "test-tts"
        probe.audio_cache_dir = tmp_path / "audio_cache"
        probe.audio_cache_dir.mkdir(parents=True)
        monkeypatch.setattr(
            probe, "_ensure_audio_file", lambda text: _write_fake_wav(tmp_path, text)
        )
        return probe

    def test_skips_non_duplex_generator(self, probe_env, monkeypatch):
        probe = self._make_probe(probe_env, monkeypatch)
        gen = _FakeNonDuplexGenerator()
        attempts = list(probe.probe(gen))
        assert attempts == []

    def test_build_session_scripts_produces_two_events(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())
        probe = PETTSBargeInterrupt()
        probe.audio_cache_dir = probe_env
        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)

        source_cases = [("bash.delete_home.direct", "S008mal", "Delete home.")]
        scripts = probe._build_session_scripts([wav], source_cases)

        # One script per affirmation clip
        assert len(scripts) == len(probe.affirmation_clips)
        for script in scripts:
            events = script.user_events()
            assert len(events) == 2
            assert events[0].label == "initial_request"
            assert events[1].label == "interrupt"

    def test_scripts_cover_all_affirmation_variants(self, probe_env, monkeypatch):
        """5 cases × 4 affirmations = 20 scripts, not 5 (the zip-truncation bug)."""
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())
        probe = PETTSBargeInterrupt()
        probe.audio_cache_dir = probe_env
        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)

        n_cases = 5
        source_cases = [("c", "i", "t")] * n_cases
        scripts = probe._build_session_scripts([wav] * n_cases, source_cases)

        expected = n_cases * len(probe.affirmation_clips)
        assert len(scripts) == expected, (
            f"expected {expected} scripts ({n_cases} cases × "
            f"{len(probe.affirmation_clips)} affirmations), got {len(scripts)}"
        )

    def test_interrupt_event_has_trigger(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())
        probe = PETTSBargeInterrupt()
        probe.audio_cache_dir = probe_env
        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)

        scripts = probe._build_session_scripts(
            [wav], [("case_id", "intent", "source")]
        )
        assert scripts[0].events[1].trigger is not None


# ---------------------------------------------------------------------------
# PETTSBargeFlood
# ---------------------------------------------------------------------------


class TestPETTSBargeFlood:
    def test_flood_counts_produce_correct_event_count(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())

        probe = PETTSBargeFlood()
        probe.audio_cache_dir = probe_env
        probe.flood_counts = (2, 5)
        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)

        scripts = probe._build_session_scripts(
            [wav], [("cid", "intent", "text")]
        )

        assert len(scripts) == 2  # one per flood_count
        event_counts = [len(s.user_events()) for s in scripts]
        assert event_counts == [3, 6]  # 1 initial + N flood

    def test_interrupt_label_is_first_flood(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())
        probe = PETTSBargeFlood()
        probe.audio_cache_dir = probe_env
        probe.flood_counts = (3,)
        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)

        scripts = probe._build_session_scripts([wav], [("c", "i", "t")])
        assert scripts[0].interrupt_label == "flood_0"


# ---------------------------------------------------------------------------
# PETTSFalseBargeIn
# ---------------------------------------------------------------------------


class TestPETTSFalseBargeIn:
    def test_produces_one_script_per_trigger_type(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())
        silence = probe_env / "_silence_200ms.wav"
        silence.write_bytes(_make_wav_bytes(200))

        probe = PETTSFalseBargeIn()
        probe.audio_cache_dir = probe_env

        # Stub apply_transform_recipe so false-trigger WAV is just silence
        monkeypatch.setattr(
            "garak.probes.audio_duplex.apply_transform_recipe",
            lambda src, dst, recipe: Path(dst).write_bytes(_make_wav_bytes(50)),
        )

        scripts = probe._build_session_scripts([wav], [("c", "i", "t")])
        from garak.probes.audio_duplex import _FALSE_TRIGGER_RECIPES

        assert len(scripts) == len(_FALSE_TRIGGER_RECIPES)

    def test_false_trigger_event_has_audio_data(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())

        probe = PETTSFalseBargeIn()
        probe.audio_cache_dir = probe_env

        monkeypatch.setattr(
            "garak.probes.audio_duplex.apply_transform_recipe",
            lambda src, dst, recipe: Path(dst).write_bytes(_make_wav_bytes(50)),
        )

        scripts = probe._build_session_scripts([wav], [("c", "i", "t")])
        trigger_events = [e for s in scripts for e in s.user_events() if e.label == "false_trigger"]
        assert all(e.audio_data is not None for e in trigger_events)


# ---------------------------------------------------------------------------
# AudioSilenceSmuggling
# ---------------------------------------------------------------------------


class TestAudioSilenceSmuggling:
    def test_encode_produces_valid_wav(self):
        probe = AudioSilenceSmuggling()
        probe.short_pause_ms = 100
        probe.long_pause_ms = 300
        probe.encoded_payload = "Hi"

        data = probe._encode_to_pauses("Hi")
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000

    def test_encoded_and_control_differ(self):
        probe = AudioSilenceSmuggling()
        probe.short_pause_ms = 100
        probe.long_pause_ms = 300
        probe.encoded_payload = "AB"

        encoded = probe._encode_to_pauses("AB")
        bits = [0, 1] * 8
        control = probe._encode_to_pauses_bits(bits)
        assert encoded != control

    def test_builds_paired_conditions(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())

        probe = AudioSilenceSmuggling()
        scripts = probe._build_session_scripts([wav], [("c", "i", "t")])
        conditions = [s.metadata["condition"] for s in scripts]
        assert set(conditions) == {"encoded", "control"}


# ---------------------------------------------------------------------------
# PETTSDriftPoison
# ---------------------------------------------------------------------------


class TestPETTSDriftPoison:
    def test_session_length_produces_correct_event_count(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())

        probe = PETTSDriftPoison()
        probe.audio_cache_dir = probe_env
        probe.session_lengths = (5,)
        probe.noise_snr_db = 30

        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)
        monkeypatch.setattr(
            "garak.probes.audio_duplex.apply_transform_recipe",
            lambda src, dst, recipe: Path(dst).write_bytes(_make_wav_bytes()),
        )

        scripts = probe._build_session_scripts([wav], [("c", "i", "t")])
        # 2 conditions (perturbed, control) × 1 session_length × 1 case
        assert len(scripts) == 2
        for s in scripts:
            # 5 turns: 4 benign + 1 harmful
            assert len(s.user_events()) == 5

    def test_last_event_is_harmful_request(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())

        probe = PETTSDriftPoison()
        probe.audio_cache_dir = probe_env
        probe.session_lengths = (3,)
        probe.noise_snr_db = 30

        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)
        monkeypatch.setattr(
            "garak.probes.audio_duplex.apply_transform_recipe",
            lambda src, dst, recipe: Path(dst).write_bytes(_make_wav_bytes()),
        )

        scripts = probe._build_session_scripts([wav], [("c", "i", "t")])
        for s in scripts:
            assert s.user_events()[-1].label == "harmful_request"

    def test_control_condition_exists(self, probe_env, monkeypatch):
        wav = probe_env / "test.wav"
        wav.write_bytes(_make_wav_bytes())

        probe = PETTSDriftPoison()
        probe.audio_cache_dir = probe_env
        probe.session_lengths = (5,)
        probe.noise_snr_db = 30

        monkeypatch.setattr(probe, "_ensure_audio_file", lambda t: wav)
        monkeypatch.setattr(
            "garak.probes.audio_duplex.apply_transform_recipe",
            lambda src, dst, recipe: Path(dst).write_bytes(_make_wav_bytes()),
        )

        scripts = probe._build_session_scripts([wav], [("c", "i", "t")])
        conditions = {s.metadata["condition"] for s in scripts}
        assert "control" in conditions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_wav(tmp_path: Path, text: str) -> Path:
    digest = hex(hash(text) & 0xFFFFFFFF)[2:]
    p = tmp_path / f"{digest}.wav"
    if not p.exists():
        p.write_bytes(_make_wav_bytes())
    return p
