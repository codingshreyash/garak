# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA NIM Microservice LLM Interface"""

import hashlib
import json
import logging
import mimetypes
from pathlib import Path
import time
from typing import List, Union

import openai
import requests

from garak import _config
from garak.attempt import Message, Turn, Conversation
from garak.exception import GarakException
from garak.generators.base import Generator
from garak.generators.openai import OpenAICompatible


class NVAudioTranscription(Generator):
    """Wrapper for NVIDIA hosted audio transcription endpoints.

    Connects to ``/v1/audio/{model}/transcriptions`` and returns the
    transcribed text. This supports audio-in, text-out targets such as
    Parakeet ASR endpoints on NVIDIA's inference API.

    You must set the ``NIM_API_KEY`` environment variable. Run garak with
    ``--target_type nim.NVAudioTranscription`` and optionally set
    ``--target_name`` to the NVIDIA audio transcription endpoint name.
    """

    ENV_VAR = "NIM_API_KEY"
    DEFAULT_MODEL = "nvidia/parakeet-1-1b-rnnt-multilingual"
    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "uri": "https://inference-api.nvidia.com/v1",
        "language": "en-US",
        "request_timeout": 60,
        "max_audio_bytes": 25_000_000,
    }
    active = True
    supports_multiple_generations = False
    generator_family_name = "NVIDIAAudioTranscription"
    modality = {"in": {"audio"}, "out": {"text"}}
    audio_formats = {"wav"}

    def __init__(self, name="", config_root=_config):
        super().__init__(name or self.DEFAULT_MODEL, config_root=config_root)

    def _transcription_url(self) -> str:
        return f"{self.uri.rstrip('/')}/audio/" f"{self.name.strip('/')}/transcriptions"

    def _audio_message(self, prompt: Conversation) -> Message:
        if not isinstance(prompt, Conversation):
            raise GarakException(
                f"{self.__class__.__name__} expected a Conversation prompt."
            )

        for turn in reversed(prompt.turns):
            message = turn.content
            if message.data_path is not None:
                return message
            if message.data is not None:
                return message

        raise GarakException(
            f"{self.__class__.__name__} expected a prompt containing audio data."
        )

    def _validate_audio_path(self, audio_path: Path) -> None:
        if not audio_path.is_file():
            raise GarakException(
                f"{self.__class__.__name__} audio file not found: {audio_path}"
            )
        audio_format = audio_path.suffix.lower().lstrip(".")
        if audio_format not in self.audio_formats:
            raise GarakException(
                f"{self.__class__.__name__} expected one of "
                f"{sorted(self.audio_formats)} audio formats: {audio_path}"
            )
        if audio_path.stat().st_size > self.max_audio_bytes:
            raise GarakException(
                f"{self.__class__.__name__} audio file exceeds "
                f"{self.max_audio_bytes} bytes: {audio_path}"
            )

    @staticmethod
    def _mime_type(audio_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(audio_path)
        if mime_type == "audio/x-wav":
            return "audio/wav"
        return mime_type or "audio/wav"

    def _post_transcription(self, file_payload) -> dict:
        try:
            response = requests.post(
                self._transcription_url(),
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"language": self.language},
                files={"file": file_payload},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as exc:
            raise GarakException(
                f"{self.__class__.__name__} transcription request failed."
            ) from exc
        except ValueError as exc:
            raise GarakException(
                f"{self.__class__.__name__} transcription response was not JSON."
            ) from exc

        if not isinstance(response_json.get("text"), str):
            raise GarakException(
                f"{self.__class__.__name__} transcription response omitted text."
            )
        return response_json

    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Union[Message, None]]:
        assert (
            generations_this_call == 1
        ), "generations_per_call / n > 1 is not supported"

        message = self._audio_message(prompt)
        if message.data_path is not None:
            audio_path = Path(message.data_path)
            self._validate_audio_path(audio_path)
            with audio_path.open("rb") as audio_file:
                response_json = self._post_transcription(
                    (audio_path.name, audio_file, self._mime_type(audio_path))
                )
        else:
            audio_data = message.data
            if len(audio_data) > self.max_audio_bytes:
                raise GarakException(
                    f"{self.__class__.__name__} audio data exceeds "
                    f"{self.max_audio_bytes} bytes."
                )
            # validate raw bytes via the mime the message carries
            mime_type = (message.data_type or (None, None))[0] or "audio/wav"
            audio_format = mime_type.split("/")[-1]
            if audio_format == "x-wav":
                audio_format = "wav"
            if audio_format not in self.audio_formats:
                raise GarakException(
                    f"{self.__class__.__name__} expected one of "
                    f"{sorted(self.audio_formats)} audio formats: {mime_type}"
                )
            response_json = self._post_transcription(
                (f"audio.{audio_format}", audio_data, f"audio/{audio_format}")
            )

        return [
            Message(
                text=response_json["text"],
                lang=response_json.get("language_code", self.language),
            )
        ]


class NVVoiceChat(Generator):
    """Speech-to-speech target via an OpenAI-compatible chat-completions shim.

    Connects to a ``/v1/chat/completions`` endpoint that accepts base64-encoded
    WAV audio and returns a text transcript (and optionally a WAV audio reply).

    Works against any OpenAI-compatible proxy that accepts ``input_audio``
    content.  Point ``uri`` at the shim's ``/v1`` base URL, set ``--target_name``
    to the model the shim serves, and set ``NIM_API_KEY`` (leave it blank if the
    endpoint requires no authentication).

    Some targets need trailing silence at the end of the audio so they have time
    to finish the response before the stream closes.  ``trailing_silence_ms``
    controls how much is appended; set to 0 to disable.

    Request payload shape::

        {
          "model": "<model-name>",
          "messages": [{
            "role": "user",
            "content": [{
              "type": "input_audio",
              "input_audio": {"data": "<base64-wav>", "format": "wav"}
            }]
          }],
          "generate_audio": true
        }

    The text reply is taken from ``choices[0].message.content``; the audio
    reply (base64 WAV) is available in ``choices[0].message.audio.data`` but
    is not surfaced by default.
    """

    ENV_VAR = "NIM_API_KEY"
    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "uri": "https://integrate.api.nvidia.com/v1",
        "audio_format": "wav",
        "generate_audio": True,
        "request_timeout": 120,
        "request_retries": 0,
        "retry_delay_seconds": 5.0,
        "retry_max_delay_seconds": 30.0,
        "retry_status_codes": (500, 502, 503, 504),
        "max_audio_bytes": 25_000_000,
        "trailing_silence_ms": 2000,
        "system_prompt": None,
        "text_prompt": None,
        "tools": None,
        "tool_choice": None,
        "extra_body": {},
        "extra_headers": {},
        "response_audio_dir": None,
    }
    active = True
    supports_multiple_generations = False
    generator_family_name = "NVVoiceChat"
    modality = {"in": {"audio", "text"}, "out": {"text"}}
    audio_formats = {"wav"}

    def __init__(self, name="", config_root=_config):
        super().__init__(name, config_root=config_root)
        if self.name in ("", None):
            raise ValueError(
                f"{self.generator_family_name} requires model name to be set, "
                "e.g. --target_name <model-served-by-the-shim>"
            )

    def _completions_url(self) -> str:
        return f"{self.uri.rstrip('/')}/chat/completions"

    def _post_completion(self, *, headers: dict, payload: dict):
        retries = int(self.request_retries)
        if retries < 0:
            raise GarakException(
                f"{self.__class__.__name__} request_retries must not be negative."
            )
        retry_status_codes = {int(code) for code in self.retry_status_codes}
        for request_index in range(retries + 1):
            try:
                response = requests.post(
                    self._completions_url(),
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                status_code = (
                    exc.response.status_code if exc.response is not None else None
                )
                retryable = status_code in retry_status_codes or isinstance(
                    exc,
                    (
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                    ),
                )
                if not retryable or request_index >= retries:
                    raise GarakException(
                        f"{self.__class__.__name__} request failed."
                    ) from exc
                delay = min(
                    float(self.retry_delay_seconds) * (2**request_index),
                    float(self.retry_max_delay_seconds),
                )
                logging.warning(
                    "%s transient request failure%s; retrying %s/%s in %.1fs.",
                    self.__class__.__name__,
                    f" with HTTP {status_code}" if status_code is not None else "",
                    request_index + 1,
                    retries,
                    delay,
                )
                time.sleep(delay)

        raise GarakException(f"{self.__class__.__name__} request failed.")

    def _audio_message(self, prompt: Conversation) -> Message:
        if not isinstance(prompt, Conversation):
            raise GarakException(
                f"{self.__class__.__name__} expected a Conversation prompt."
            )
        for turn in reversed(prompt.turns):
            msg = turn.content
            if msg.data_path is not None or msg.data is not None:
                return msg
        raise GarakException(
            f"{self.__class__.__name__} expected a prompt containing audio data."
        )

    @staticmethod
    def _response_provenance(response, response_json: dict, response_message: dict):
        response_headers = getattr(response, "headers", {})
        identifiers = (
            {
                key: value
                for key, value in response_headers.items()
                if any(
                    marker in key.lower()
                    for marker in ("request-id", "reqid", "trace", "correlation")
                )
            }
            if hasattr(response_headers, "items")
            else {}
        )
        audio = response_message.get("audio")
        provenance = {
            "http_status": getattr(response, "status_code", None),
            "response_identifiers": identifiers,
            "audio_present": isinstance(audio, dict) and bool(audio.get("data")),
        }
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in response_json:
                provenance[key] = response_json[key]
        if isinstance(response_json.get("usage"), dict):
            provenance["usage"] = response_json["usage"]
        return provenance

    def _audio_requested(self) -> bool:
        """Whether audio output was actually requested for this call.

        ``extra_body`` may override ``generate_audio``; if audio was not
        requested there is nothing to save and no error to raise.
        """
        if isinstance(self.extra_body, dict) and "generate_audio" in self.extra_body:
            return bool(self.extra_body["generate_audio"])
        return bool(self.generate_audio)

    def _save_response_audio(self, response_message: dict) -> dict:
        if self.response_audio_dir is None:
            return {}

        audio = response_message.get("audio")
        encoded_audio = audio.get("data") if isinstance(audio, dict) else None
        if not isinstance(encoded_audio, str) or not encoded_audio:
            if self._audio_requested():
                raise GarakException(
                    f"{self.__class__.__name__} response omitted requested audio."
                )
            return {}

        import base64
        import binascii

        try:
            audio_bytes = base64.b64decode(encoded_audio, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise GarakException(
                f"{self.__class__.__name__} response audio was not valid base64."
            ) from exc
        if not (audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE"):
            raise GarakException(
                f"{self.__class__.__name__} response audio was not a WAV file."
            )

        digest = hashlib.sha256(audio_bytes, usedforsecurity=False).hexdigest()
        output_dir = Path(self.response_audio_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"voicechat-response-{digest[:16]}.wav"
        if not output_path.exists():
            output_path.write_bytes(audio_bytes)
        return {
            "audio_path": str(output_path),
            "audio_sha256": digest,
            "audio_byte_size": len(audio_bytes),
        }

    @staticmethod
    def _append_wav_silence(wav_bytes: bytes, silence_ms: int) -> bytes:
        """Return wav_bytes with silence_ms milliseconds of silence appended."""
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes)) as wf:
            params = wf.getparams()
            original_frames = wf.readframes(wf.getnframes())

        silence_frames = int(params.framerate * silence_ms / 1000)
        silence_bytes = b"\x00" * silence_frames * params.nchannels * params.sampwidth

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(original_frames + silence_bytes)
        return buf.getvalue()

    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Union[Message, None]]:
        import base64
        import wave

        assert (
            generations_this_call == 1
        ), "generations_per_call / n > 1 is not supported"

        message = self._audio_message(prompt)
        if message.data_path is not None:
            audio_path = Path(message.data_path)
            if not audio_path.is_file():
                raise GarakException(
                    f"{self.__class__.__name__} audio file not found: {audio_path}"
                )
            if audio_path.suffix.lower().lstrip(".") not in self.audio_formats:
                raise GarakException(
                    f"{self.__class__.__name__} expected one of "
                    f"{sorted(self.audio_formats)} audio formats: {audio_path}"
                )
            if audio_path.stat().st_size > self.max_audio_bytes:
                raise GarakException(
                    f"{self.__class__.__name__} audio file exceeds "
                    f"{self.max_audio_bytes} bytes: {audio_path}"
                )
            raw = audio_path.read_bytes()
        else:
            raw = message.data

        if len(raw) > self.max_audio_bytes:
            raise GarakException(
                f"{self.__class__.__name__} audio data exceeds "
                f"{self.max_audio_bytes} bytes."
            )

        if self.trailing_silence_ms > 0:
            try:
                raw = self._append_wav_silence(raw, self.trailing_silence_ms)
            except (wave.Error, EOFError) as exc:
                raise GarakException(
                    f"{self.__class__.__name__} could not parse audio as WAV "
                    f"to append trailing silence."
                ) from exc
            if len(raw) > self.max_audio_bytes:
                raise GarakException(
                    f"{self.__class__.__name__} audio data exceeds "
                    f"{self.max_audio_bytes} bytes after appending trailing silence."
                )

        audio_b64 = base64.b64encode(raw).decode()
        messages = []
        if self.system_prompt:
            if not isinstance(self.system_prompt, str):
                raise GarakException(
                    f"{self.__class__.__name__} system_prompt must be a string."
                )
            messages.append({"role": "system", "content": self.system_prompt})

        user_content = []
        effective_text_prompt = (
            self.text_prompt if self.text_prompt is not None else message.text
        )
        if effective_text_prompt:
            if not isinstance(effective_text_prompt, str):
                raise GarakException(
                    f"{self.__class__.__name__} text_prompt must be a string."
                )
            user_content.append({"type": "text", "text": effective_text_prompt})
        user_content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": self.audio_format,
                },
            }
        )
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.name,
            "messages": messages,
            "generate_audio": self.generate_audio,
        }
        if self.extra_body:
            if not isinstance(self.extra_body, dict):
                raise GarakException(
                    f"{self.__class__.__name__} extra_body must be a dictionary."
                )
            payload.update(self.extra_body)
        if self.tools is not None:
            payload["tools"] = self.tools
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice

        headers = {"Content-Type": "application/json"}
        if self.extra_headers:
            if not isinstance(self.extra_headers, dict):
                raise GarakException(
                    f"{self.__class__.__name__} extra_headers must be a dictionary."
                )
            headers.update(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self._post_completion(headers=headers, payload=payload)
            response_json = response.json()
        except ValueError as exc:
            raise GarakException(
                f"{self.__class__.__name__} response was not JSON."
            ) from exc

        try:
            response_message = response_json["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GarakException(
                f"{self.__class__.__name__} response missing expected fields."
            ) from exc

        if "content" not in response_message and "tool_calls" not in response_message:
            raise GarakException(
                f"{self.__class__.__name__} response missing expected fields."
            )

        tool_calls = response_message.get("tool_calls")
        text = response_message.get("content")
        if tool_calls:
            text_payload = {"tool_calls": tool_calls}
            if isinstance(text, str) and text:
                text_payload["content"] = text
            text = json.dumps(text_payload, sort_keys=True)

        if not isinstance(text, str):
            raise GarakException(
                f"{self.__class__.__name__} response content was not a string."
            )

        provenance = self._response_provenance(
            response, response_json, response_message
        )
        provenance.update(self._save_response_audio(response_message))
        return [
            Message(
                text=text,
                notes={"nvvoicechat_response": provenance},
            )
        ]


class NVOpenAIChat(OpenAICompatible):
    """Wrapper for NVIDIA NIM microservices hosted on build.nvidia.com and self-hosted.

    Connects to the v1/chat/completions endpoint.
    You must set the NIM_API_KEY environment variable even if you connect to a self-hosted NIM.

    To get started with this generator:

    #. Visit https://build.nvidia.com/explore/reasoning and find the LLM you'd like to use.
    #. On the page for the LLM you want to use (such as `mixtral-8x7b-instruct <https://build.nvidia.com/mistralai/mixtral-8x7b-instruct>`__),
       click **Get API key** above the code snippet.

       You might need to create an account if you don't have one yet.
       Copy this key.
    #. In your console, set the ``NIM_API_KEY`` variable to this API key.

       On Linux, this might look like ``export NIM_API_KEY="nvapi-xXxXxXx"``.
    #. Run garak, setting ``--target_type 'nim.NVIDIAOpenAIChat'`` and ``--target_name`` to
       the name of the model on build.nvidia.com, such as ``--target_name 'mistralai/mixtral-8x7b-instruct-v0.1'``.
    """

    # per https://docs.nvidia.com/ai-enterprise/nim-llm/latest/openai-api.html
    # 2024.05.02, `n>1` is not supported
    ENV_VAR = "NIM_API_KEY"
    DEFAULT_PARAMS = OpenAICompatible.DEFAULT_PARAMS | {
        "temperature": 0.1,
        "top_p": 0.7,
        "top_k": 0,  # top_k is hard set to zero as of 24.04.30
        "uri": "https://integrate.api.nvidia.com/v1/",
        "vary_seed_each_call": True,  # encourage variation when generations>1. not respected by all NIMs
        "vary_temp_each_call": True,  # encourage variation when generations>1. not respected by all NIMs
        "suppressed_params": {"n", "frequency_penalty", "presence_penalty", "timeout"},
    }
    active = True
    supports_multiple_generations = False
    generator_family_name = "NIM"

    timeout = 60

    def _load_unsafe(self):
        self.client = openai.OpenAI(base_url=self.uri, api_key=self.api_key)
        if self.name in ("", None):
            raise ValueError(
                "NIMs require model name to be set, e.g. --target_name mistralai/mistral-8x7b-instruct-v0.1\nCurrent models:\n"
                + "\n - ".join(
                    sorted([entry.id for entry in self.client.models.list().data])
                )
            )
        self.generator = self.client.chat.completions

    def _prepare_prompt(self, prompt: Conversation) -> Conversation:
        return prompt

    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Union[Message, None]]:
        assert (
            generations_this_call == 1
        ), "generations_per_call / n > 1 is not supported"

        if self.vary_seed_each_call:
            self.seed = self._rng.randint(0, 65535)

        if self.vary_temp_each_call:
            self.temperature = self._rng.random()

        prompt = self._prepare_prompt(prompt)
        if prompt is None:
            # if we didn't get a valid prompt, don't process it, and send the NoneType(s) downstream
            return [None] * generations_this_call

        try:
            result = super()._call_model(prompt, generations_this_call)
        except openai.UnprocessableEntityError as uee:
            msg = "Model call didn't match endpoint expectations, see log"
            logging.critical(msg, exc_info=uee)
            raise GarakException(f"🛑 {msg}") from uee
        except openai.NotFoundError as nfe:
            msg = "NIM endpoint not found. Is the model name spelled correctly and the endpoint URI correct?"
            logging.critical(msg, exc_info=nfe)
            raise GarakException(f"🛑 {msg}") from nfe
        except Exception as oe:
            msg = "NIM generation failed. Is the model name spelled correctly?"
            logging.critical(msg, exc_info=oe)
            raise GarakException(f"🛑 {msg}") from oe

        return result

    def __init__(self, name="", config_root=_config):
        super().__init__(name, config_root=config_root)
        if "/" not in self.name:
            msg = "❓ Is this a valid NIM name? expected a slash-formatted name, e.g. 'org/model'"
            logging.info(msg)
            print(msg)


class NVOpenAICompletion(NVOpenAIChat):
    """Wrapper for NVIDIA NIM microservices hosted on build.nvidia.com and self-hosted.

    Connects to the v1/completions endpoint.
    You must set the NIM_API_KEY environment variable even if you connect to a self-hosted NIM.

    To get started with this generator:

    #. Visit https://build.nvidia.com/explore/reasoning and find the LLM you'd like to use.
    #. On the page for the LLM you want to use (such as `mixtral-8x7b-instruct <https://build.nvidia.com/mistralai/mixtral-8x7b-instruct>`__),
       click **Get API key** above the code snippet.

       You might need to create an account if you don't have one yet.
       Copy this key.
    #. In your console, set the ``NIM_API_KEY`` variable to this API key.

       On Linux, this might look like ``export NIM_API_KEY="nvapi-xXxXxXx"``.
    #. Run garak, setting ``--target_type 'nim.NVIDIAOpenAIChat'`` and ``--target_name`` to
       the name of the model on build.nvidia.com, such as ``--target_name 'mistralai/mixtral-8x7b-instruct-v0.1'``.
    """

    def _load_unsafe(self):
        self.client = openai.OpenAI(base_url=self.uri, api_key=self.api_key)
        self.generator = self.client.completions


class NVMultimodal(NVOpenAIChat):
    """Wrapper for text and image / audio to text NVIDIA NIM microservices hosted on build.nvidia.com and self-hosted.

    You must set the NIM_API_KEY environment variable even if you connect to a self-hosted NIM.

    Expects prompt ``Message`` objects to be have ``text`` (required), and ``data`` (optional) in either ``image`` or ``audio`` format.

    By default the ``embed_data`` parameter is disabled, message preparation is deferred to OpenAICompatible for multimodal format.

    When the ``embed_data`` parameter is enabled, message is sent with ``role`` and ``content`` where ``content`` is structured as text
    followed by ``<img>`` and/or ``<audio>`` tags.
    Refer to https://build.nvidia.com/microsoft/phi-4-multimodal-instruct for an example.

    To get started with this generator:

    #. Visit https://build.nvidia.com/explore/reasoning and find the LLM you'd like to use.
    #. On the page for the LLM you want to use (such as `phi-4-multimodal-instruct <https://build.nvidia.com/microsoft/phi-4-multimodal-instruct>`__),
       click **Get API key** above the code snippet.

       You might need to create an account if you don't have one yet.
       Copy this key.
    #. In your console, set the ``NIM_API_KEY`` variable to this API key.

       On Linux, this might look like ``export NIM_API_KEY="nvapi-xXxXxXx"``.
    #. Run garak, setting ``--target_type 'nim.NVMultimodal'`` and ``--target_name`` to
       the name of the model on build.nvidia.com, such as ``--target_name 'microsoft/phi-4-multimodal-instruct-v0.1'``.
    """

    DEFAULT_PARAMS = NVOpenAIChat.DEFAULT_PARAMS | {
        "suppressed_params": {"n", "frequency_penalty", "presence_penalty", "stop"},
        "max_input_len": 180_000,
        "embed_data": False,
    }

    modality = {"in": {"text", "image", "audio"}, "out": {"text"}}

    def _prepare_prompt(self, conv: Conversation) -> Conversation:
        if not self.embed_data:
            return conv

        from dataclasses import asdict

        prepared_conv = Conversation()

        for turn in conv.turns:
            msg = turn.content
            # only manipulate the copy
            prepared_msg = Message(**asdict(msg))

            text = msg.text

            # guessing a default in the case of direct data
            data_extension = "image/jpg"
            # should this use mime/type detection on the actually data vs a default guess?
            data_tag = "img"

            if msg.data is not None:
                import base64

                if msg.data_path is not None:
                    data_extension, _ = msg.data_type
                    if data_extension.startswith("audio"):
                        data_tag = "audio"

                data_b64 = base64.b64encode(msg.data).decode()

                if len(data_b64) > self.max_input_len:
                    big_img_filename = "<direct data>"
                    if msg.data_path is not None:
                        big_img_filename = msg.data_path
                    logging.error(
                        "Request for %s exceeds length limit. To upload larger files, use the assets API (not yet supported)",
                        big_img_filename,
                    )
                    return None

                text = (
                    text
                    + f' <{data_tag} src="data:{data_extension};base64,{data_b64}" />'
                )
            prepared_msg.text = text

            prepared_conv.turns.append(Turn(turn.role, prepared_msg))

        return prepared_conv


class Vision(NVMultimodal):
    """Wrapper for text and image to text NVIDIA NIM microservices hosted on build.nvidia.com and self-hosted.

    You must set the NIM_API_KEY environment variable even if you connect to a self-hosted NIM.

    Following generators.huggingface.LLaVa, expects prompts to be a ``Message`` with keys
    ``text`` and ``data`` (optional) in ``image`` mimetype format.
    The ``text`` key specifies the text prompt, and the ``image`` key specifies the path to the image.
    """

    modality = {"in": {"text", "image"}, "out": {"text"}}


DEFAULT_CLASS = "NVOpenAIChat"
