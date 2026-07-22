# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVDuplexChat and the DuplexCapable mixin.

NVDuplexChat now extends NVVoiceChat (OpenAI SDK path) and returns TEXT only.
run_session drives self._call_model per turn, so these tests stub _call_model
rather than any HTTP layer. No audio-output / ASR-fallback behaviour exists.
"""

import io
import struct
import wave

import pytest

from garak.attempt import Conversation, Message, Turn
from garak.generators.nim import DuplexCapable, NVDuplexChat, NVVoiceChat
from garak.resources.audio.session import (
    ReactivePattern,
    SessionEvent,
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


def _dc_config(**extra):
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


def _make_gen(monkeypatch, responses, capture=None):
    """Build an NVDuplexChat whose _call_model returns queued text responses."""
    gen = NVDuplexChat("test-model", config_root=_dc_config(trailing_silence_ms=0))
    resp_iter = iter(responses)

    def fake_call_model(prompt, generations_this_call=1):
        if capture is not None:
            capture.append(prompt)
        try:
            text = next(resp_iter)
        except StopIteration:
            text = "fallback"
        return [Message(text=text)]

    monkeypatch.setattr(gen, "_call_model", fake_call_model)
    return gen


# ---------------------------------------------------------------------------
# DuplexCapable mixin
# ---------------------------------------------------------------------------


class TestDuplexCapable:
    def test_nvduplexchat_is_duplex_capable(self):
        assert issubclass(NVDuplexChat, DuplexCapable)

    def test_supports_duplex_attribute(self):
        assert NVDuplexChat.supports_duplex is True

    def test_nvvoicechat_not_duplex_capable(self):
        assert not issubclass(NVVoiceChat, DuplexCapable)

    def test_output_modality_text_only(self):
        assert NVDuplexChat.modality["out"] == {"text"}

    def test_no_asr_params(self):
        assert "asr_uri" not in NVDuplexChat.DEFAULT_PARAMS
        assert "asr_model" not in NVDuplexChat.DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# _resolve_event_audio
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
            stream="user", offset_s=0.0, audio_data=b"inline-bytes", audio_path=str(p)
        )
        assert self.gen._resolve_event_audio(event) == b"inline-bytes"


# ---------------------------------------------------------------------------
# run_session
# ---------------------------------------------------------------------------


class TestRunSession:
    def test_single_turn_populates_full_transcript(self, monkeypatch, tmp_path):
        gen = _make_gen(monkeypatch, ["Agent said hello."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req")]
        )
        result = gen.run_session(script)
        assert "hello" in result.full_transcript

    def test_two_turn_session_transcript_join(self, monkeypatch, tmp_path):
        gen = _make_gen(monkeypatch, ["First turn.", "Second turn."])
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

    def test_history_accumulates_across_turns(self, monkeypatch, tmp_path):
        """Each turn's Conversation must include all prior turns."""
        capture = []
        gen = _make_gen(monkeypatch, ["r1", "r2", "r3"], capture=capture)
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            events=[
                SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="t1"),
                SessionEvent(stream="user", offset_s=1.0, audio_path=str(wav), label="t2"),
                SessionEvent(stream="user", offset_s=2.0, audio_path=str(wav), label="t3"),
            ]
        )
        gen.run_session(script)
        # turn 1: 1 turn; turn 2: prior user+assistant + new user = 3; turn 3: 5
        assert [len(c.turns) for c in capture] == [1, 3, 5]

    def test_pre_post_split_on_interrupt_label(self, monkeypatch, tmp_path):
        gen = _make_gen(monkeypatch, ["I cannot do that.", "Okay, proceeding."])
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

    def test_reactive_trigger_records_triggered_by(self, monkeypatch, tmp_path):
        gen = _make_gen(monkeypatch, ["I cannot help.", "Sure thing."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        trigger = ReactivePattern(pattern=r"i cannot", offset_ms=0)
        script = SessionScript(
            interrupt_label="interrupt",
            events=[
                SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req"),
                SessionEvent(
                    stream="user", offset_s=5.0, audio_path=str(wav), label="interrupt", trigger=trigger
                ),
            ],
        )
        result = gen.run_session(script)
        triggered = [e for e in result.events if e.triggered_by is not None]
        assert len(triggered) >= 1

    def test_missing_audio_event_skipped(self, monkeypatch):
        gen = _make_gen(monkeypatch, ["Response."])
        script = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, label="empty")]
        )
        result = gen.run_session(script)
        assert result.error is None

    def test_call_model_failure_yields_empty_turn(self, monkeypatch, tmp_path):
        gen = NVDuplexChat("test-model", config_root=_dc_config(trailing_silence_ms=0))

        def boom(prompt, generations_this_call=1):
            raise RuntimeError("endpoint down")

        monkeypatch.setattr(gen, "_call_model", boom)
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req")]
        )
        result = gen.run_session(script)
        # empty transcript, but no crash
        assert result.full_transcript.strip() == ""

    def test_result_as_dict_structure(self, monkeypatch, tmp_path):
        gen = _make_gen(monkeypatch, ["Hello."])
        wav = tmp_path / "r.wav"
        wav.write_bytes(_make_wav())
        script = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req")]
        )
        result = gen.run_session(script)
        d = result.as_dict()
        assert "full_transcript" in d
        assert isinstance(d["events"], list)


# ---------------------------------------------------------------------------
# run_session integration with real _call_model (SDK create stubbed)
# ---------------------------------------------------------------------------


class TestRunSessionThroughCallModel:
    def test_audio_flows_through_to_sdk_create(self, monkeypatch, tmp_path):
        """End-to-end: run_session -> _call_model -> _conversation_to_list -> create."""
        import base64

        wav = tmp_path / "r.wav"
        wav_bytes = _make_wav()
        wav.write_bytes(wav_bytes)

        gen = NVDuplexChat("test-model", config_root=_dc_config(trailing_silence_ms=0))

        captured = []

        class _Msg:
            content = "I cannot help with that."
            tool_calls = None

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        def fake_create(**kwargs):
            captured.append(kwargs)
            return _Resp()

        stub = type("_C", (), {"create": staticmethod(fake_create)})()
        monkeypatch.setattr(gen, "generator", stub)

        script = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, audio_path=str(wav), label="req")]
        )
        result = gen.run_session(script)

        assert result.full_transcript == "I cannot help with that."
        # the SDK request carried an input_audio block with our WAV
        messages = captured[0]["messages"]
        audio_parts = [
            part
            for m in messages
            if isinstance(m.get("content"), list)
            for part in m["content"]
            if part.get("type") == "input_audio"
        ]
        assert audio_parts, "duplex turn must send input_audio"
        assert base64.b64decode(audio_parts[0]["input_audio"]["data"]) == wav_bytes
