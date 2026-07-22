# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA NIM Microservice LLM Interface"""

import io
import json
import logging
import mimetypes
from pathlib import Path
from typing import List, Optional, Union
import wave

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
    Parakeet ASR endpoints.

    You must set the ``NIM_API_KEY`` environment variable. Run garak with
    ``--target_type nim.NVAudioTranscription`` and optionally set
    ``--target_name`` to the transcription endpoint name.

    ``uri`` defaults to the public ``integrate.api.nvidia.com`` surface; point
    it at any endpoint that exposes the ``/v1/audio/{model}/transcriptions``
    transcription route (set ``--generator_option uri=...`` if your deployment
    uses a different base URL).
    """

    ENV_VAR = "NIM_API_KEY"
    DEFAULT_MODEL = "nvidia/parakeet-1-1b-rnnt-multilingual"
    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "uri": "https://integrate.api.nvidia.com/v1",
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


class DuplexCapable:
    """Mixin declaring that a generator supports duplex session scripts.

    Generators that inherit this mixin must implement :meth:`run_session`.
    Probes call ``isinstance(generator, DuplexCapable)`` to decide whether to
    run duplex probes or skip gracefully.
    """

    supports_duplex: bool = True

    def run_session(self, script):
        raise NotImplementedError


class NVVoiceChat(NVOpenAIChat):
    """Speech-to-text target: send audio to an OpenAI-compatible S2S chat shim.

    Sends base64-encoded WAV audio as an ``input_audio`` content block to a
    ``/v1/chat/completions`` endpoint and reads the target's **text** transcript
    from ``choices[0].message.content``.  Per garak's convention the generator's
    output modality is text only -- the target (or its shim) is responsible for
    returning a text transcript of whatever it spoke.  This generator does not
    receive, save, or transcribe audio responses; doing so would make the test
    measure the target *as interpreted by a transcription provider* rather than
    the target itself.

    Extends :class:`NVOpenAIChat` and reuses the OpenAI SDK client, so it follows
    the same target-communication pattern as ``nim.Vision`` / ``nim.NVMultimodal``.
    Point ``uri`` at the shim's ``/v1`` base URL, set ``--target_name`` to the
    model the shim serves, and set ``NIM_API_KEY`` (leave blank if the endpoint
    needs no auth).

    Some targets need trailing silence at the end of the audio so they have time
    to finish the response before the stream closes; ``trailing_silence_ms``
    controls how much is appended (set to 0 to disable).  ``generate_audio`` is
    forwarded in the request body for shims that require it to trigger the S2S
    pipeline, but any returned audio is ignored.
    """

    ENV_VAR = "NIM_API_KEY"
    DEFAULT_PARAMS = NVOpenAIChat.DEFAULT_PARAMS | {
        "audio_format": "wav",
        "generate_audio": True,
        "max_audio_bytes": 25_000_000,
        "trailing_silence_ms": 2000,
        "system_prompt": None,
        "text_prompt": None,
        "tools": None,
        "tool_choice": None,
        "extra_body": {},
        "extra_headers": {},
        # Slow S2S endpoints need a long per-request timeout; the OpenAI SDK
        # client handles retry/backoff on 429/5xx/connection errors internally,
        # so request_retries maps to the client's max_retries.
        "request_timeout": 120,
        "request_retries": 2,
        # NVVoiceChat sends a deliberately minimal request; sampling params are
        # suppressed because voice shims typically reject them.
        "suppressed_params": {
            "n",
            "frequency_penalty",
            "presence_penalty",
            "timeout",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "seed",
            "max_tokens",
        },
        "vary_seed_each_call": False,
        "vary_temp_each_call": False,
    }
    active = True
    supports_multiple_generations = False
    generator_family_name = "NVVoiceChat"
    modality = {"in": {"audio", "text"}, "out": {"text"}}
    audio_formats = {"wav"}

    def __init__(self, name="", config_root=_config):
        # Bypass NVOpenAIChat's org/model slash heuristic -- S2S shim model
        # names need not be slash-formatted. OpenAICompatible.__init__ still
        # enforces that a model name is set (raises if empty) and builds the SDK
        # client.
        OpenAICompatible.__init__(self, name, config_root=config_root)

    def _load_unsafe(self):
        # Unlike NVOpenAIChat, do not enumerate models over the network when the
        # name is unset — voice shims may not implement /models, and we want a
        # clean, offline error message. Wire the configured timeout / retries
        # into the SDK client (slow S2S endpoints need a long timeout).
        self.client = openai.OpenAI(
            base_url=self.uri,
            api_key=self.api_key,
            timeout=getattr(self, "request_timeout", 120),
            max_retries=getattr(self, "request_retries", 2),
        )
        if self.name in ("", None):
            raise ValueError(
                f"{self.generator_family_name} requires model name to be set, "
                "e.g. --target_name <model-served-by-the-shim>"
            )
        self.generator = self.client.chat.completions

    def _audio_message(self, prompt: Conversation) -> Union[Message, None]:
        if not isinstance(prompt, Conversation):
            raise GarakException(
                f"{self.__class__.__name__} expected a Conversation prompt."
            )
        for turn in reversed(prompt.turns):
            msg = turn.content
            if msg.data_path is not None or msg.data is not None:
                return msg
        return None

    def _validate_audio_size(self, raw: bytes, context: str = "") -> None:
        if len(raw) > self.max_audio_bytes:
            raise GarakException(
                f"{self.__class__.__name__} audio exceeds "
                f"{self.max_audio_bytes} bytes{context}."
            )

    @staticmethod
    def _append_wav_silence(wav_bytes: bytes, silence_ms: int) -> bytes:
        """Return wav_bytes with silence_ms milliseconds of silence appended."""
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

    def _prepare_prompt(self, prompt: Conversation) -> Union[Conversation, None]:
        """Validate audio and inline it (with optional trailing silence).

        Returns a Conversation whose last audio-bearing message carries the
        resolved WAV bytes inline so ``OpenAICompatible._conversation_to_list``
        emits an ``input_audio`` content block. The generator-level
        ``text_prompt`` overrides the message text when set. Non-audio prompts
        pass through unchanged (and will be rejected downstream).
        """
        audio_msg = self._audio_message(prompt)
        if audio_msg is None:
            raise GarakException(
                f"{self.__class__.__name__} expected a prompt containing audio data."
            )

        if audio_msg.data_path is not None:
            audio_path = Path(audio_msg.data_path)
            if not audio_path.is_file():
                raise GarakException(
                    f"{self.__class__.__name__} audio file not found: {audio_path}"
                )
            fmt = audio_path.suffix.lower().lstrip(".")
            if fmt not in self.audio_formats:
                raise GarakException(
                    f"{self.__class__.__name__} expected one of "
                    f"{sorted(self.audio_formats)} audio formats: {audio_path}"
                )
            raw = audio_path.read_bytes()
        else:
            raw = audio_msg.data
            mime = (audio_msg.data_type or (None, None))[0] or f"audio/{self.audio_format}"
            fmt = mime.split("/")[-1]
            if fmt == "x-wav":
                fmt = "wav"
            if fmt not in self.audio_formats:
                raise GarakException(
                    f"{self.__class__.__name__} expected one of "
                    f"{sorted(self.audio_formats)} audio formats: {mime}"
                )

        self._validate_audio_size(raw)

        if self.trailing_silence_ms and self.trailing_silence_ms > 0:
            try:
                raw = self._append_wav_silence(raw, self.trailing_silence_ms)
            except (wave.Error, EOFError) as exc:
                raise GarakException(
                    f"{self.__class__.__name__} could not parse audio as WAV "
                    f"to append trailing silence."
                ) from exc
            self._validate_audio_size(raw, " after appending trailing silence")

        effective_text = (
            self.text_prompt if self.text_prompt is not None else (audio_msg.text or "")
        )
        new_turns = []
        replaced = False
        for turn in prompt.turns:
            if turn.content is audio_msg and not replaced:
                new_msg = Message(
                    text=effective_text,
                    lang=audio_msg.lang,
                    data_type=(f"audio/{fmt}", None),
                )
                new_msg.data = raw
                new_turns.append(Turn(turn.role, new_msg))
                replaced = True
            else:
                new_turns.append(turn)
        return Conversation(new_turns)

    @staticmethod
    def _serialise_tool_calls(tool_calls, content) -> str:
        serialised = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                serialised.append(tc.model_dump())
            elif isinstance(tc, dict):
                serialised.append(tc)
            else:
                serialised.append(str(tc))
        payload = {"tool_calls": serialised}
        if isinstance(content, str) and content:
            payload["content"] = content
        return json.dumps(payload, sort_keys=True, default=str)

    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Union[Message, None]]:
        assert (
            generations_this_call == 1
        ), "generations_per_call / n > 1 is not supported"

        if self.client is None:
            self._load_unsafe()

        prompt = self._prepare_prompt(prompt)
        if prompt is None:
            return [None]

        messages = self._conversation_to_list(prompt)
        if self.system_prompt:
            if not isinstance(self.system_prompt, str):
                raise GarakException(
                    f"{self.__class__.__name__} system_prompt must be a string."
                )
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        create_args = {"model": self.name, "messages": messages}
        extra_body = dict(self.extra_body) if isinstance(self.extra_body, dict) else {}
        if self.generate_audio:
            extra_body.setdefault("generate_audio", True)
        if extra_body:
            create_args["extra_body"] = extra_body
        if self.extra_headers:
            create_args["extra_headers"] = self.extra_headers
        if self.tools is not None:
            create_args["tools"] = self.tools
        if self.tool_choice is not None:
            create_args["tool_choice"] = self.tool_choice

        try:
            response = self.generator.create(**create_args)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            msg = (
                f"OpenAI API authentication failed (HTTP {e.status_code}); "
                f"verify {self.key_env_var} is valid."
            )
            logging.error(msg)
            raise GarakException(msg) from None
        except openai.BadRequestError as e:
            logging.exception(e)
            return [None]
        except Exception as e:
            msg = (
                f"{self.__class__.__name__} generation failed. Is the model name "
                "spelled correctly and does the shim accept input_audio?"
            )
            logging.critical(msg, exc_info=e)
            raise GarakException(f"\U0001f6d1 {msg}") from e

        if not getattr(response, "choices", None):
            logging.debug("%s got no choices in response", self.__class__.__name__)
            return [None]

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        content = getattr(message, "content", None)
        if tool_calls:
            text = self._serialise_tool_calls(tool_calls, content)
        else:
            text = content if isinstance(content, str) else ""

        return [Message(text=text)]


class NVDuplexChat(DuplexCapable, NVVoiceChat):
    """Sequential-simulation duplex generator built on top of NVVoiceChat.

    Executes a :class:`~garak.resources.audio.session.SessionScript` as a
    multi-turn conversation: each user-stream event is sent as a separate
    request that includes the full prior conversation history, so the target
    sees a realistic, coherent multi-turn session.

    Barge-in is modelled in *sequential simulation mode*: if a
    :class:`~garak.resources.audio.session.ReactivePattern` is set on an event
    and the previous agent response matches that pattern, the interrupt event
    is recorded as triggered.  This approximation tests the same safety surface
    as true duplex barge-in because the model sees an identical conversation
    prefix.

    Like :class:`NVVoiceChat` the output modality is text only: each turn's
    scorable transcript comes from ``choices[0].message.content``. No audio is
    received or transcribed by the generator.

    Point ``uri`` at the ``/v1`` base URL of any OpenAI-compatible
    chat-completions shim and set ``--target_name`` to the served model.
    """

    generator_family_name = "NVDuplexChat"

    # ------------------------------------------------------------------
    # Core session execution
    # ------------------------------------------------------------------

    def run_session(self, script) -> object:
        from garak.resources.audio.session import SessionResult, SessionResultEvent

        result = SessionResult()
        history_turns: list = []  # accumulated Conversation Turns (user audio + assistant text)

        interrupt_seen = False
        interrupt_label = getattr(script, "interrupt_label", None)

        for event in script.user_events():
            audio_data = self._resolve_event_audio(event)
            if audio_data is None:
                logging.warning(
                    "NVDuplexChat: skipping event %r — no audio could be resolved",
                    event.label,
                )
                continue

            # Build a user Turn carrying the audio inline. _prepare_prompt
            # (invoked by _call_model) validates + appends trailing silence and
            # applies text_prompt override to the last audio turn.
            user_text = event.text if self.text_prompt is None else self.text_prompt
            user_msg = Message(text=user_text or "", data_type=(f"audio/{self.audio_format}", None))
            user_msg.data = audio_data
            user_turn = Turn("user", user_msg)

            conversation = Conversation(history_turns + [user_turn])

            try:
                outputs = self._call_model(conversation)
                response_text = (
                    outputs[0].text if outputs and outputs[0] is not None else ""
                )
            except Exception as exc:
                logging.warning(
                    "NVDuplexChat: turn %r failed: %s", event.label, exc
                )
                response_text = ""

            triggered_by = None
            if (
                event.trigger is not None
                and result.events
                and event.trigger.matches(result.events[-1].text)
            ):
                triggered_by = result.events[-1].label

            result.events.append(
                SessionResultEvent(
                    offset_s=event.offset_s,
                    stream="user",
                    text="[audio]",
                    label=event.label,
                )
            )
            result.events.append(
                SessionResultEvent(
                    offset_s=event.offset_s,
                    stream="agent",
                    text=response_text,
                    label=f"response_to_{event.label}",
                    triggered_by=triggered_by,
                )
            )

            # Store raw (silence-free) turns in history; _prepare_prompt applies
            # trailing silence per-call to the last audio turn only.
            history_turns.append(user_turn)
            history_turns.append(Turn("assistant", Message(text=response_text)))

            if interrupt_label and event.label == interrupt_label:
                interrupt_seen = True

        agent_responses = [e for e in result.events if e.stream == "agent"]
        result.full_transcript = " ".join(e.text for e in agent_responses)

        if interrupt_seen and interrupt_label:
            pre_events: list = []
            post_events: list = []
            past_interrupt = False
            for e in result.events:
                if e.stream == "user" and e.label == interrupt_label:
                    past_interrupt = True
                    continue
                if e.stream != "agent":
                    continue
                if past_interrupt:
                    post_events.append(e)
                else:
                    pre_events.append(e)
            result.pre_interrupt_transcript = " ".join(e.text for e in pre_events)
            result.post_interrupt_transcript = " ".join(e.text for e in post_events)
        else:
            result.post_interrupt_transcript = result.full_transcript

        return result

    def _resolve_event_audio(self, event) -> Optional[bytes]:
        """Return raw WAV bytes for *event* from inline data or a file path."""
        if event.audio_data is not None:
            return event.audio_data
        if event.audio_path is not None:
            p = Path(event.audio_path)
            if not p.is_file():
                return None
            return p.read_bytes()
        return None

DEFAULT_CLASS = "NVOpenAIChat"
