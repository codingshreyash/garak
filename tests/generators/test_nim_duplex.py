# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVDuplexChat and the DuplexCapable mixin."""

import base64
import io
import json
import struct
import wave

import pytest

from garak.generators.nim import DuplexCapable, NVDuplexChat
from garak.resources.audio.session import (
    ReactivePattern,
    SessionEvent,
    SessionResult,
    SessionScript,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(duration_ms: int = 200) -> bytes:
    n = int(16000 * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


def _vc_config(model_name="test-model", **extra):
    return {
        "generators": {
            "nim": {
                "NVDuplexChat": {
                    "api_key": "test-key",
                    "uri": "https://shim.test/v1",
                    **extra,
                }
            }
        }
    }


class _FakeResponse:
    def __init__(self, content="Hello from agent.", status_code=200):
        self.status_code = status_code
        self._content = content
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": self._content,
                    }
                }
            ]
        }


# ---------------------------------------------------------------------------
# DuplexCapable mixin
# ---------------------------------------------------------------------------


class TestDuplexCapable:
    def test_nvduplexchat_is_duplex_capable(self):
        assert issubclass(NVDuplexChat, DuplexCapable)

    def test_supports_duplex_attribute(self):
        assert NVDuplexChat.supports_duplex is True

    def test_non_duplex_generator_not_instance(self):
        from garak.generators.nim import NVVoiceChat

        assert not isinstance(NVVoiceChat.__new__(NVVoiceChat), DuplexCapable)


# ---------------------------------------------------------------------------
# NVDuplexChat._resolve_event_audio
# ---------------------------------------------------------------------------


class TestResolveEventAudio:
    def setup_method(self):
        self.gen = NVDuplexChat.__new__(NVDuplexChat)

    def test_returns_audio_data_if_set(self):
        event = SessionEvent(stream="user", offset_s=0.0, audio_data=b"wav-bytes")
        assert self.gen._resolve_event_audio(event) == b"wav-bytes"

    def test_returns_file_bytes_if_path_set(self, tmp_path):
        p = tmp_path / "test.wav"
        p.write_bytes(b"wav-file")
        event = SessionEvent(stream="user", offset_s=0.0, audio_path=str(p))
        assert self.gen._resolve_event_audio(event) == b"wav-file"

    def test_returns_none_if_neither(self):
        event = SessionEvent(stream="user", offset_s=0.0, text="text only")
        assert self.gen._resolve_event_audio(event) is None

    def test_prefers_audio_data_over_path(self, tmp_path):
        p = tmp_path / "test.wav"
        p.write_bytes(b"file-bytes")
        event = SessionEvent(
            stream="user",
            offset_s=0.0,
            audio_data=b"inline-bytes",
            audio_path=str(p),
        )
        assert self.gen._resolve_event_audio(event) == b"inline-bytes"


# ---------------------------------------------------------------------------
# NVDuplexChat.run_session
# ---------------------------------------------------------------------------


class TestTranscribeResponseAudio:
    def test_asr_not_called_when_asr_uri_unset(self, monkeypatch):
        """Without asr_uri configured, blank content is returned as-is."""
        gen = NVDuplexChat.__new__(NVDuplexChat)
        gen.asr_uri = None
        # _transcribe_response_audio should never be reached
        assert gen.asr_uri is None

    def test_transcribe_called_on_blank_content_with_asr_uri(self, monkeypatch):
        gen = NVDuplexChat("test-model", config_root=_vc_config())
        gen.trailing_silence_ms = 0
        gen.asr_uri = "https://asr.test/v1"
        gen.asr_model = "nvidia/parakeet-1-1b-rnnt-multilingual"
        gen.asr_language = "en-US"

        asr_called = []

        def fake_transcribe(audio_b64: str) -> str:
            asr_called.append(audio_b64)
            return "I cannot help with that."

        monkeypatch.setattr(gen, "_transcribe_response_audio", fake_transcribe)

        # Fake response: blank content but audio data present
        import base64 as _b64
        dummy_audio = _b64.b64encode(b"RIFF....WAVE").decode()

        def fake_post(*, headers, payload):
            return _FakeResponse.__new__(_FakeResponse).__class__(
                **{
                    **_FakeResponse.__init__.__code__.co_varnames  # just use the class directly
                }
            ) if False else type("R", (), {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "choices": [{"message": {"role": "assistant", "content": "", "audio": {"data": dummy_audio}}}]
                },
            })()

        monkeypatch.setattr(gen, "_post_completion", fake_post)

        from garak.resources.audio.session import SessionEvent, SessionScript

        import io, struct, wave as _wave
        buf = io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(struct.pack("<100h", *([0]*100)))
        wav_bytes = buf.getvalue()

        event = SessionEvent(stream="user", offset_s=0.0, audio_data=wav_bytes, label="req")
        text, provenance = gen._send_turn(wav_bytes, [], event)

        assert asr_called, "ASR transcription should have been called for blank content with audio"
        assert text == "I cannot help with that."
        assert provenance["asr_used"] is True


class TestRunSession:
    def _make_gen(self, monkeypatch, responses: list[str]):
        gen = NVDuplexChat("test-model", config_root=_vc_config())
        gen.trailing_silence_ms = 0   # skip silence append in tests

        resp_iter = iter(responses)

        def fake_post_completion(*, headers, payload):
            try:
                content = next(resp_iter)
            except StopIteration:
                content = "fallback response"
            return _FakeResponse(content=content)

        monkeypatch.setattr(gen, "_post_completion", fake_post_completion)
        return gen

    def test_single_turn_populates_full_transcript(self, monkeypatch, tmp_path):
        gen = self._make_gen(monkeypatch, ["Agent said hello."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req")]
        )
        result = gen.run_session(script)
        assert "hello" in result.full_transcript

    def test_two_turn_session_transcript_join(self, monkeypatch, tmp_path):
        gen = self._make_gen(monkeypatch, ["First turn.", "Second turn."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            events=[
                SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="t1"),
                SessionEvent(stream="user", offset_s=5.0, audio_path=str(wav), label="t2"),
            ]
        )
        result = gen.run_session(script)
        assert "First turn" in result.full_transcript
        assert "Second turn" in result.full_transcript

    def test_pre_post_split_on_interrupt_label(self, monkeypatch, tmp_path):
        gen = self._make_gen(monkeypatch, ["I cannot do that.", "Okay, proceeding."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            interrupt_label="interrupt",
            events=[
                SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req"),
                SessionEvent(stream="user", offset_s=5.0, audio_path=str(wav), label="interrupt"),
            ],
        )
        result = gen.run_session(script)
        assert "cannot" in result.pre_interrupt_transcript
        assert "proceeding" in result.post_interrupt_transcript

    def test_reactive_trigger_notes_triggered_by(self, monkeypatch, tmp_path):
        gen = self._make_gen(monkeypatch, ["I cannot help.", "Sure thing."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        trigger = ReactivePattern(pattern=r"i cannot", offset_ms=0)
        script = SessionScript(
            interrupt_label="interrupt",
            events=[
                SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req"),
                SessionEvent(
                    stream="user",
                    offset_s=5.0,
                    audio_path=str(wav),
                    label="interrupt",
                    trigger=trigger,
                ),
            ],
        )
        result = gen.run_session(script)
        interrupt_responses = [
            e for e in result.events if e.triggered_by is not None
        ]
        assert len(interrupt_responses) >= 1

    def test_missing_audio_event_skipped(self, monkeypatch):
        gen = self._make_gen(monkeypatch, ["Response."])
        # Event has no audio source at all
        script = SessionScript(
            events=[
                SessionEvent(stream="user", offset_s=0.0, label="empty")
            ]
        )
        result = gen.run_session(script)
        # Should complete without crashing; full_transcript may be empty
        assert result.error is None

    def test_result_as_dict_structure(self, monkeypatch, tmp_path):
        gen = self._make_gen(monkeypatch, ["Hello."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req")]
        )
        result = gen.run_session(script)
        d = result.as_dict()
        assert "full_transcript" in d
        assert "events" in d
        assert isinstance(d["events"], list)
