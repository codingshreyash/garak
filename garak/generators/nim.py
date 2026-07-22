# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA NIM Microservice LLM Interface"""

import logging
import io
import json
import wave
from typing import List, Union

from pathlib import Path
import openai

from garak import _config
from garak.attempt import Message, Turn, Conversation
from garak.exception import GarakException
from garak.generators.openai import OpenAICompatible


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
    modality = {"in": {"audio"}, "out": {"text"}}
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
            mime = (audio_msg.data_type or (None, None))[
                0
            ] or f"audio/{self.audio_format}"
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


DEFAULT_CLASS = "NVOpenAIChat"
