"""Tests for NVVoiceChat (audio-in / text-out S2S chat target over the OpenAI SDK)."""

import io
import struct
import wave

import pytest

from garak.attempt import Conversation, Message, Turn
from garak.exception import GarakException
from garak.generators.nim import NVVoiceChat


def _make_wav(num_samples=1600, framerate=16000, nchannels=1, sampwidth=2) -> bytes:
    """Return a minimal valid WAV file as bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return buf.getvalue()


def _vc_config(api_key="test-key", **extra):
    return {
        "generators": {
            "nim": {
                "NVVoiceChat": {
                    "api_key": api_key,
                    "uri": "https://shim.nvcf.nvidia.com/v1",
                    **extra,
                }
            }
        }
    }


class _FakeSDKMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeSDKChoice:
    def __init__(self, message):
        self.message = message


class _FakeSDKResponse:
    def __init__(self, content="Hello, I can help with that.", tool_calls=None):
        self.choices = [_FakeSDKChoice(_FakeSDKMessage(content, tool_calls))]


class _FakeToolCall:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def _stub_create(gen, monkeypatch, *, response=None, capture=None, error=None):
    """Replace gen.generator with a stub exposing .create()."""

    def fake_create(**kwargs):
        if capture is not None:
            capture.append(kwargs)
        if error is not None:
            raise error
        return response if response is not None else _FakeSDKResponse()

    stub = type("_StubCompletions", (), {"create": staticmethod(fake_create)})()
    monkeypatch.setattr(gen, "generator", stub)
    return gen


def _audio_block(messages):
    """Return the input_audio dict from the last user message's content list."""
    for msg in reversed(messages):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for part in msg["content"]:
                if part.get("type") == "input_audio":
                    return part["input_audio"]
    return None


def test_nv_voice_chat_reports_supported_audio_formats():
    assert NVVoiceChat.supported_formats("audio") == {
        "wav"
    }, "NVVoiceChat must advertise WAV input"
    assert (
        NVVoiceChat.supported_formats("image") == set()
    ), "NVVoiceChat must not advertise image input"


def test_nv_voice_chat_output_modality_is_text_only():
    assert NVVoiceChat.modality["out"] == {
        "text"
    }, "detectors must receive only target-provided text"


def test_nv_voice_chat_requires_target_name():
    with pytest.raises(ValueError, match="requires a target name"):
        NVVoiceChat(config_root=_vc_config())


def test_nv_voice_chat_sends_input_audio_and_reads_text(monkeypatch, tmp_path):
    import base64

    wav_bytes = _make_wav()
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(wav_bytes)

    capture = []
    gen = NVVoiceChat("voice-chat", config_root=_vc_config(trailing_silence_ms=0))
    _stub_create(gen, monkeypatch, capture=capture)

    result = gen._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    assert (
        result[0].text == "Hello, I can help with that."
    ), "the target transcript must become the generation"
    block = _audio_block(capture[0]["messages"])
    assert block is not None, "request must carry an input_audio block"
    assert block["format"] == "wav", "the request must declare WAV input"
    assert (
        base64.b64decode(block["data"]) == wav_bytes
    ), "the request must preserve the source audio"


def test_nv_voice_chat_forwards_explicit_shim_options(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    capture = []
    gen = NVVoiceChat(
        "voice-chat",
        config_root=_vc_config(
            trailing_silence_ms=0,
            extra_body={"generate_audio": True},
        ),
    )
    _stub_create(gen, monkeypatch, capture=capture)

    gen._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )
    assert (
        capture[0]["extra_body"]["generate_audio"] is True
    ), "explicit endpoint extensions must be forwarded"


def test_nv_voice_chat_has_no_implicit_audio_output_request():
    assert (
        "generate_audio" not in NVVoiceChat.DEFAULT_PARAMS
    ), "a non-standard shim field must not be enabled by default"


def test_nv_voice_chat_forwards_system_text_tools(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    capture = []
    gen = NVVoiceChat(
        "voice-chat",
        config_root=_vc_config(
            trailing_silence_ms=0,
            system_prompt="You are a helper.",
            text_prompt="Answer the audio.",
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto",
        ),
    )
    _stub_create(gen, monkeypatch, capture=capture)

    gen._call_model(
        Conversation([Turn("user", Message("ignored", data_path=str(audio_path)))])
    )

    messages = capture[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "You are a helper.",
    }, "the configured system prompt must be forwarded"
    # user turn text part uses the configured text_prompt, overriding msg text
    text_parts = [
        p["text"]
        for p in messages[-1]["content"]
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    assert text_parts == [
        "Answer the audio."
    ], "configured text must replace the message text"
    assert (
        capture[0]["tools"][0]["function"]["name"] == "f"
    ), "tool definitions must be forwarded"
    assert capture[0]["tool_choice"] == "auto", "tool choice must be forwarded"


def test_nv_voice_chat_appends_trailing_silence(monkeypatch, tmp_path):
    import base64
    import io
    import wave

    wav_bytes = _make_wav(num_samples=1600, framerate=16000)
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(wav_bytes)
    capture = []
    gen = NVVoiceChat("voice-chat", config_root=_vc_config(trailing_silence_ms=1000))
    _stub_create(gen, monkeypatch, capture=capture)

    gen._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    block = _audio_block(capture[0]["messages"])
    sent = base64.b64decode(block["data"])
    with wave.open(io.BytesIO(sent)) as wf:
        frames = wf.getnframes()
    # original 1600 frames + 1000ms * 16000Hz = 16000 frames appended
    assert frames == 1600 + 16000, "one second of silence must add one second of frames"


def test_nv_voice_chat_audio_only_response_is_unscored(monkeypatch, tmp_path):
    """An audio-only response cannot be interpreted by text detectors."""
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    gen = NVVoiceChat("voice-chat", config_root=_vc_config(trailing_silence_ms=0))
    _stub_create(gen, monkeypatch, response=_FakeSDKResponse(content=""))

    result = gen._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )
    assert result == [
        None
    ], "audio-only output must not be represented as a scored blank response"


def test_nv_voice_chat_serializes_tool_calls(monkeypatch, tmp_path):
    import json as _json

    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    tool_call = _FakeToolCall(
        {"function": {"name": "get_stock_price", "arguments": '{"ticker":"NVDA"}'}}
    )
    gen = NVVoiceChat("voice-chat", config_root=_vc_config(trailing_silence_ms=0))
    _stub_create(
        gen,
        monkeypatch,
        response=_FakeSDKResponse(content=None, tool_calls=[tool_call]),
    )

    result = gen._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )
    parsed = _json.loads(result[0].text)
    assert (
        parsed["tool_calls"][0]["function"]["name"] == "get_stock_price"
    ), "the tool name must be preserved"
    assert (
        parsed["tool_calls"][0]["function"]["arguments"] == '{"ticker":"NVDA"}'
    ), "the tool arguments must be preserved"


def test_nv_voice_chat_rejects_missing_audio(monkeypatch):
    gen = NVVoiceChat("voice-chat", config_root=_vc_config())
    _stub_create(gen, monkeypatch)
    with pytest.raises(GarakException, match="expected a prompt containing audio"):
        gen._call_model(Conversation([Turn("user", Message("no audio here"))]))


def test_nv_voice_chat_rejects_non_wav_input(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.mp3"
    audio_path.write_bytes(b"ID3")
    gen = NVVoiceChat("voice-chat", config_root=_vc_config())
    _stub_create(gen, monkeypatch)
    with pytest.raises(GarakException, match="expected one of"):
        gen._call_model(
            Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
        )


def test_nv_voice_chat_rejects_oversize_audio(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    gen = NVVoiceChat(
        "voice-chat", config_root=_vc_config(trailing_silence_ms=0, max_audio_bytes=1)
    )
    _stub_create(gen, monkeypatch)
    with pytest.raises(GarakException, match="exceeds"):
        gen._call_model(
            Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
        )


def test_nv_voice_chat_has_no_audio_output_params():
    """Audio-output handling was removed; these params must not exist."""
    assert (
        "response_audio_dir" not in NVVoiceChat.DEFAULT_PARAMS
    ), "the generator must not store response audio"
    assert (
        "asr_uri" not in NVVoiceChat.DEFAULT_PARAMS
    ), "the generator must not own transcription"
