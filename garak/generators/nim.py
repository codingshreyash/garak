# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA NIM Microservice LLM Interface"""

import logging
import mimetypes
from pathlib import Path
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
        return (
            f"{self.uri.rstrip('/')}/audio/"
            f"{self.name.strip('/')}/transcriptions"
        )

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
            response_json = self._post_transcription(
                ("audio.wav", audio_data, "audio/wav")
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


DEFAULT_CLASS = "NVOpenAIChat"
