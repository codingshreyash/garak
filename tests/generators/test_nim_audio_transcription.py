# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
from pathlib import Path
import struct
import wave

import pytest
import requests

from garak.attempt import Conversation, Message, Turn
from garak.exception import GarakException
from garak.generators.nim import NVAudioTranscription, NVVoiceChat


def _make_wav(num_samples=1600, framerate=16000, nchannels=1, sampwidth=2) -> bytes:
    """Return a minimal valid WAV file as bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return buf.getvalue()


def _config(api_key="test-key"):
    return {
        "generators": {
            "nim": {
                "NVAudioTranscription": {
                    "api_key": api_key,
                    "uri": "https://inference-api.nvidia.com/v1",
                }
            }
        }
    }


class _FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload or {
            "text": "What is natural language processing? ",
            "language_code": "en-US",
        }

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_nim_audio_transcription_reports_supported_audio_formats():
    assert NVAudioTranscription.supported_formats("audio") == {"wav"}
    assert NVAudioTranscription.supported_formats("image") == set()


def test_nim_audio_transcription_posts_audio_file(monkeypatch, tmp_path):
    audio_path = tmp_path / "prompt.wav"
    audio_path.write_bytes(b"RIFF....WAVE")
    requests = []

    def fake_post(url, *, headers, data, files, timeout):
        requests.append(
            {
                "url": url,
                "headers": headers,
                "data": data,
                "filename": files["file"][0],
                "content_type": files["file"][2],
                "timeout": timeout,
            }
        )
        return _FakeResponse()

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)

    generator = NVAudioTranscription(config_root=_config())
    result = generator._call_model(
        Conversation([Turn("user", Message("transcribe", data_path=str(audio_path)))])
    )

    assert result[0].text == "What is natural language processing? "
    assert result[0].lang == "en-US"
    assert requests == [
        {
            "url": (
                "https://inference-api.nvidia.com/v1/audio/"
                "nvidia/parakeet-1-1b-rnnt-multilingual/transcriptions"
            ),
            "headers": {"Authorization": "Bearer test-key"},
            "data": {"language": "en-US"},
            "filename": "prompt.wav",
            "content_type": "audio/wav",
            "timeout": 60,
        }
    ]


def test_nim_audio_transcription_uses_configured_language(monkeypatch, tmp_path):
    audio_path = tmp_path / "prompt.wav"
    audio_path.write_bytes(b"RIFF....WAVE")
    requests = []

    def fake_post(url, *, headers, data, files, timeout):
        requests.append(data)
        return _FakeResponse({"text": "bonjour", "language_code": "fr-FR"})

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)

    config = _config()
    config["generators"]["nim"]["NVAudioTranscription"]["language"] = "fr-FR"
    generator = NVAudioTranscription(config_root=config)
    result = generator._call_model(
        Conversation([Turn("user", Message("transcribe", data_path=str(audio_path)))])
    )

    assert requests == [{"language": "fr-FR"}]
    assert result[0].lang == "fr-FR"


def test_nim_audio_transcription_rejects_missing_audio():
    generator = NVAudioTranscription(config_root=_config())

    with pytest.raises(GarakException, match="expected a prompt containing audio data"):
        generator._call_model(Conversation([Turn("user", Message("no audio"))]))


def test_nim_audio_transcription_rejects_oversize_file(tmp_path):
    audio_path = tmp_path / "prompt.wav"
    audio_path.write_bytes(b"RIFF....WAVE")
    config = _config()
    config["generators"]["nim"]["NVAudioTranscription"]["max_audio_bytes"] = 1
    generator = NVAudioTranscription(config_root=config)

    with pytest.raises(GarakException, match="audio file exceeds"):
        generator._call_model(
            Conversation(
                [Turn("user", Message("transcribe", data_path=str(audio_path)))]
            )
        )


def test_nim_audio_transcription_rejects_unsupported_audio_format(tmp_path):
    audio_path = tmp_path / "prompt.mp3"
    audio_path.write_bytes(b"ID3")
    generator = NVAudioTranscription(config_root=_config())

    with pytest.raises(GarakException, match="expected one of"):
        generator._call_model(
            Conversation(
                [Turn("user", Message("transcribe", data_path=str(audio_path)))]
            )
        )


def test_nim_audio_transcription_posts_inline_wav_bytes(monkeypatch):
    captured = []

    def fake_post(url, *, headers, data, files, timeout):
        captured.append(files["file"])
        return _FakeResponse()

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)

    msg = Message("transcribe", data_type=("audio/wav", None))
    msg.data = b"RIFF....WAVE"
    generator = NVAudioTranscription(config_root=_config())
    generator._call_model(Conversation([Turn("user", msg)]))

    filename, payload, content_type = captured[0]
    assert content_type == "audio/wav", "inline WAV bytes keep the wav mime type"
    assert payload == b"RIFF....WAVE"


def test_nim_audio_transcription_rejects_inline_non_wav_bytes():
    msg = Message("transcribe", data_type=("audio/mpeg", None))
    msg.data = b"\xff\xfb\x90"
    generator = NVAudioTranscription(config_root=_config())

    with pytest.raises(GarakException, match="expected one of"):
        generator._call_model(Conversation([Turn("user", msg)]))


# ---------------------------------------------------------------------------
# NVVoiceChat tests
# ---------------------------------------------------------------------------


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


class _VCFakeResponse:
    def __init__(
        self,
        text="Hello, I can help with that.",
        payload=None,
        *,
        headers=None,
        status_code=200,
    ):
        self._text = text
        self._payload = payload
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        if self._payload is not None:
            return self._payload
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": self._text,
                        "audio": {"data": "AAAA"},
                    }
                }
            ]
        }


def test_nv_voice_chat_reports_supported_audio_formats():
    assert NVVoiceChat.supported_formats("audio") == {"wav"}
    assert NVVoiceChat.supported_formats("image") == set()


def test_nv_voice_chat_posts_audio_file(monkeypatch, tmp_path):
    import base64

    wav_bytes = _make_wav()
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(wav_bytes)
    captured = []

    def fake_post(url, *, headers, json, timeout):
        captured.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return _VCFakeResponse()

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)

    generator = NVVoiceChat(config_root=_vc_config(trailing_silence_ms=0))
    result = generator._call_model(
        Conversation([Turn("user", Message("say hello", data_path=str(audio_path)))])
    )

    assert result[0].text == "Hello, I can help with that."
    req = captured[0]
    assert req["url"] == "https://shim.nvcf.nvidia.com/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer test-key"
    assert req["json"]["model"] == "nemotron-voice-chat"
    assert req["json"]["generate_audio"] is True
    content = req["json"]["messages"][0]["content"]
    assert content[0] == {
        "type": "text",
        "text": "say hello",
    }, "per-attempt text accompanies its audio when no override is configured"
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"
    assert content[1]["input_audio"]["data"] == base64.b64encode(wav_bytes).decode()


def test_nv_voice_chat_preserves_sanitised_response_provenance(monkeypatch, tmp_path):
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(_make_wav())
    response = _VCFakeResponse(
        payload={
            "id": "completion-one",
            "model": "nemotron-voice-chat",
            "choices": [
                {
                    "message": {
                        "content": "Natural language processing handles language.",
                        "audio": {"data": "AAAA"},
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
        headers={
            "nvcf-reqid": "request-one",
            "Authorization": "must-not-be-recorded",
        },
    )
    monkeypatch.setattr(
        "garak.generators.nim.requests.post", lambda *args, **kwargs: response
    )

    generator = NVVoiceChat(config_root=_vc_config(trailing_silence_ms=0))
    result = generator._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    provenance = result[0].notes["nvvoicechat_response"]
    assert provenance["http_status"] == 200, "HTTP status is retained"
    assert provenance["response_identifiers"] == {
        "nvcf-reqid": "request-one"
    }, "NVCF request IDs are retained"
    assert provenance["audio_present"] is True, "response audio presence is retained"
    assert provenance["usage"]["total_tokens"] == 0, "usage diagnostics are retained"
    assert (
        "Authorization" not in provenance["response_identifiers"]
    ), "sensitive and unrelated headers are not retained"


def test_nv_voice_chat_optionally_saves_response_audio(monkeypatch, tmp_path):
    import base64

    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(_make_wav())
    response_wav = _make_wav(num_samples=3200, framerate=24000)
    response = _VCFakeResponse(
        payload={
            "choices": [
                {
                    "message": {
                        "content": "Natural language processing handles language.",
                        "audio": {
                            "data": base64.b64encode(response_wav).decode("ascii")
                        },
                    }
                }
            ]
        }
    )
    monkeypatch.setattr(
        "garak.generators.nim.requests.post", lambda *args, **kwargs: response
    )

    output_dir = tmp_path / "responses"
    generator = NVVoiceChat(
        config_root=_vc_config(
            trailing_silence_ms=0,
            response_audio_dir=str(output_dir),
        )
    )
    result = generator._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    provenance = result[0].notes["nvvoicechat_response"]
    saved_path = Path(provenance["audio_path"])
    assert saved_path.read_bytes() == response_wav, "response WAV is preserved"
    assert provenance["audio_byte_size"] == len(
        response_wav
    ), "response byte size is retained"
    assert len(provenance["audio_sha256"]) == 64, "response digest is retained"


def test_nv_voice_chat_response_audio_dir_with_audio_disabled_does_not_raise(
    monkeypatch, tmp_path
):
    """response_audio_dir + extra_body disabling audio must not raise."""
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(_make_wav())
    # response carries no audio because generation was disabled
    no_audio = _VCFakeResponse(
        payload={"choices": [{"message": {"content": "Text-only reply."}}]}
    )
    monkeypatch.setattr(
        "garak.generators.nim.requests.post", lambda *args, **kwargs: no_audio
    )

    output_dir = tmp_path / "responses"
    generator = NVVoiceChat(
        config_root=_vc_config(
            trailing_silence_ms=0,
            response_audio_dir=str(output_dir),
            extra_body={"generate_audio": False},
        )
    )
    result = generator._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    assert result[0].text == "Text-only reply."
    provenance = result[0].notes["nvvoicechat_response"]
    assert "audio_path" not in provenance, "nothing saved when audio was not requested"


def test_nv_voice_chat_forwards_text_system_tools_and_extra_body(monkeypatch, tmp_path):
    wav_bytes = _make_wav()
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(wav_bytes)
    captured = []

    def fake_post(url, *, headers, json, timeout):
        captured.append({"headers": headers, "json": json})
        return _VCFakeResponse()

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "get_stock_price"}}
    generator = NVVoiceChat(
        config_root=_vc_config(
            trailing_silence_ms=0,
            generate_audio=True,
            system_prompt="You are a tool-using assistant.",
            text_prompt="Use get_stock_price for the attached audio request.",
            extra_body={"generate_audio": False, "shim_option": {"enabled": True}},
            extra_headers={"Accept": "*/*", "User-Agent": "curl/8.7.1"},
            tools=tools,
            tool_choice=tool_choice,
        )
    )
    generator._call_model(
        Conversation([Turn("user", Message("say hello", data_path=str(audio_path)))])
    )

    req = captured[0]["json"]
    headers = captured[0]["headers"]
    assert (
        req["generate_audio"] is False
    ), "extra_body should override base payload keys"
    assert req["shim_option"] == {
        "enabled": True
    }, "extra_body keys should be forwarded"
    assert req["tools"] == tools, "configured tools should be forwarded"
    assert (
        req["tool_choice"] == tool_choice
    ), "configured tool choice should be forwarded"
    assert req["messages"][0] == {
        "role": "system",
        "content": "You are a tool-using assistant.",
    }
    assert headers["Accept"] == "*/*", "extra headers should be forwarded"
    assert (
        headers["User-Agent"] == "curl/8.7.1"
    ), "custom user agent should be forwarded"
    assert (
        headers["Authorization"] == "Bearer test-key"
    ), "configured API key should still set authorization"
    user_content = req["messages"][1]["content"]
    assert user_content[0] == {
        "type": "text",
        "text": "Use get_stock_price for the attached audio request.",
    }
    assert user_content[1]["type"] == "input_audio"


def test_nv_voice_chat_appends_trailing_silence(monkeypatch, tmp_path):
    wav_bytes = _make_wav(num_samples=1600, framerate=16000)
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(wav_bytes)
    captured = []

    def fake_post(url, *, headers, json, timeout):
        captured.append(json)
        return _VCFakeResponse()

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)

    generator = NVVoiceChat(config_root=_vc_config(trailing_silence_ms=1000))
    generator._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    import base64

    audio_content = next(
        item
        for item in captured[0]["messages"][0]["content"]
        if item["type"] == "input_audio"
    )
    sent_wav = base64.b64decode(audio_content["input_audio"]["data"])
    with wave.open(io.BytesIO(sent_wav)) as wf:
        total_frames = wf.getnframes()
        framerate = wf.getframerate()

    # original 1600 samples + 1000ms * 16000Hz = 16000 silence samples
    assert total_frames == 1600 + 16000


def test_nv_voice_chat_omits_auth_header_without_api_key(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    captured = []

    def fake_post(url, *, headers, json, timeout):
        captured.append(headers)
        return _VCFakeResponse()

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)

    generator = NVVoiceChat(config_root=_vc_config(api_key="", trailing_silence_ms=0))
    generator._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    assert "Authorization" not in captured[0]


def test_nv_voice_chat_retries_transient_server_error(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    responses = []

    def fake_post(*args, **kwargs):
        responses.append(kwargs)
        if len(responses) == 1:
            failed = requests.Response()
            failed.status_code = 500
            failed.url = "https://shim.nvcf.nvidia.com/v1/chat/completions"
            return failed
        return _VCFakeResponse()

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)
    monkeypatch.setattr("garak.generators.nim.time.sleep", lambda _: None)

    generator = NVVoiceChat(
        config_root=_vc_config(
            trailing_silence_ms=0,
            request_retries=1,
            retry_delay_seconds=0,
        )
    )
    result = generator._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )

    assert result[0].text == "Hello, I can help with that."
    assert len(responses) == 2, "retries one transient HTTP 500 response"


def test_nv_voice_chat_does_not_retry_client_error(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    request_count = 0

    def fake_post(*args, **kwargs):
        nonlocal request_count
        request_count += 1
        failed = requests.Response()
        failed.status_code = 400
        failed.url = "https://shim.nvcf.nvidia.com/v1/chat/completions"
        return failed

    monkeypatch.setattr("garak.generators.nim.requests.post", fake_post)
    generator = NVVoiceChat(
        config_root=_vc_config(trailing_silence_ms=0, request_retries=3)
    )

    with pytest.raises(GarakException, match="request failed"):
        generator._call_model(
            Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
        )

    assert request_count == 1, "fails closed on non-retryable client errors"


def test_nv_voice_chat_rejects_missing_audio():
    generator = NVVoiceChat(config_root=_vc_config())

    with pytest.raises(GarakException, match="expected a prompt containing audio data"):
        generator._call_model(Conversation([Turn("user", Message("no audio here"))]))


def test_nv_voice_chat_rejects_non_wav_input(tmp_path):
    """Non-WAV bytes must fail cleanly as a GarakException, not a raw wave.Error."""
    audio_path = tmp_path / "not_audio.wav"
    audio_path.write_bytes(b"this is not a RIFF/WAVE file")
    generator = NVVoiceChat(config_root=_vc_config())  # trailing_silence_ms default > 0

    with pytest.raises(GarakException, match="could not parse audio as WAV"):
        generator._call_model(
            Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
        )


def test_nv_voice_chat_rejects_oversize_audio(tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    generator = NVVoiceChat(
        config_root=_vc_config(max_audio_bytes=5, trailing_silence_ms=0)
    )

    # oversize files are rejected by the stat pre-check before being read
    with pytest.raises(GarakException, match="audio file exceeds"):
        generator._call_model(
            Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
        )


def test_nv_voice_chat_rejects_audio_oversize_after_silence(tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_bytes = _make_wav()
    audio_path.write_bytes(audio_bytes)
    generator = NVVoiceChat(
        config_root=_vc_config(
            max_audio_bytes=len(audio_bytes) + 10,
            trailing_silence_ms=1000,
        )
    )

    with pytest.raises(GarakException, match="after appending trailing silence"):
        generator._call_model(
            Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
        )


def test_nv_voice_chat_raises_on_missing_content_field(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())

    class _BadResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"role": "assistant"}}]}

    monkeypatch.setattr(
        "garak.generators.nim.requests.post", lambda *a, **kw: _BadResponse()
    )

    generator = NVVoiceChat(config_root=_vc_config(trailing_silence_ms=0))
    with pytest.raises(GarakException, match="response missing expected fields"):
        generator._call_model(
            Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
        )


def test_nv_voice_chat_serializes_tool_call_response(monkeypatch, tmp_path):
    audio_path = tmp_path / "q.wav"
    audio_path.write_bytes(_make_wav())
    tool_call_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_stock_price",
                                "arguments": '{"ticker":"NVDA"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    monkeypatch.setattr(
        "garak.generators.nim.requests.post",
        lambda *a, **kw: _VCFakeResponse(payload=tool_call_response),
    )

    generator = NVVoiceChat(config_root=_vc_config(trailing_silence_ms=0))
    result = generator._call_model(
        Conversation([Turn("user", Message("ask", data_path=str(audio_path)))])
    )
    parsed = json.loads(result[0].text)

    assert parsed["tool_calls"][0]["function"]["name"] == "get_stock_price"
    assert (
        parsed["tool_calls"][0]["function"]["arguments"] == '{"ticker":"NVDA"}'
    ), "tool-call arguments should be preserved for downstream detectors"
