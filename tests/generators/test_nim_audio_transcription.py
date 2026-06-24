# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from garak.attempt import Conversation, Message, Turn
from garak.exception import GarakException
from garak.generators.nim import NVAudioTranscription


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
            Conversation([Turn("user", Message("transcribe", data_path=str(audio_path)))])
        )


def test_nim_audio_transcription_rejects_unsupported_audio_format(tmp_path):
    audio_path = tmp_path / "prompt.mp3"
    audio_path.write_bytes(b"ID3")
    generator = NVAudioTranscription(config_root=_config())

    with pytest.raises(GarakException, match="expected one of"):
        generator._call_model(
            Conversation([Turn("user", Message("transcribe", data_path=str(audio_path)))])
        )
