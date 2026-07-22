Audio Synthesis Support
=======================

The audio probes (``probes.audio.PETTS`` and its subclasses) render text prompts
to speech and send the resulting audio to an audio-capable target. The modality
shift is part of the *attack technique* and is owned by the probe -- the
generator only speaks to the target and returns text.

Two synthesis-adjacent steps are involved, and both are configurable:

* **Text-to-speech (TTS)** turns the prompt text into the audio that is sent to
  the target.
* **Speech-to-text (ASR)** is used only by the intelligibility gate, to verify
  that a synthesised clip actually carried the intended request before its
  security outcome is scored. It is never used to interpret the target's
  response.

garak does not couple to a specific synthesis model. The model is selected by
configuration and loaded behind a provider interface
(``garak.resources.audio.synthesis.SynthesisProvider``), so any compatible model
-- or a remote service -- can be used without changing the probe.

Limitations
-----------

Synthesis and transcription quality affect results: a low-fidelity TTS voice can
produce audio the target mishears, and a weak ASR pass can mislabel a valid clip.
The intelligibility gate mitigates this by excluding candidates whose synthesised
audio does not transcribe back to the request, so these artefacts reduce the
scoreable sample rather than silently inflating or deflating outcomes. Higher-
fidelity models give more trustworthy results.

Any model referenced by a default must be publicly available and carry a
permissive, Apache-2.0-compatible licence.

Local synthesis may add significant execution time depending on the model and
available hardware (a CUDA device is used automatically when present).

Supported Synthesis (TTS)
-------------------------

Local (Hugging Face)
    Any Transformers ``text-to-audio`` model can be used by setting
    ``tts_model_name``. The default is `suno/bark-small
    <https://huggingface.co/suno/bark-small>`_ (MIT), chosen as a public,
    permissively licensed model that runs out of the box. Override it with a
    higher-fidelity model for real runs.

Remote services
    A remote text-to-speech service (for example an OpenAI-compatible
    ``/v1/audio/speech`` endpoint, or a provider such as those compared in
    public TTS API roundups) can be used by implementing ``SynthesisProvider``
    for that service. Remote providers are the extension point for higher-
    quality or dual-capability services and follow the same interface as the
    local provider.

Supported Transcription (ASR)
-----------------------------

The intelligibility gate transcribes each synthesised clip with a local model.
The default is `openai/whisper-base.en <https://huggingface.co/openai/whisper-base.en>`_
(MIT). Any locally cached Whisper checkpoint may be used; the model is loaded
locally only and no download is triggered at run time.

Configuration
-------------

TTS settings are configured on the probe using garak's standard configurable
pattern. The relevant keys are:

* ``tts_model_name`` -- the model to synthesise with (default ``suno/bark-small``)
* ``tts_model_revision`` -- optional pinned model revision
* ``tts_voice`` -- optional voice preset forwarded to the model
* ``tts_sample_rate`` -- optional target sample rate

Examples
--------

Default (local, out of the box)
    No configuration is required; the probe synthesises with ``suno/bark-small``::

        python3 -m garak --target_type nim.NVVoiceChat --target_name <model> \
            --probes audio.ToolRiskPairedSemanticPETTS

Custom local TTS model
    Point the probe at a different Transformers ``text-to-audio`` model with a
    YAML config passed via ``--config``::

        probes:
          audio:
            PETTS:
              tts_model_name: <huggingface-text-to-audio-model>
              tts_voice: <optional-voice-preset>

Custom ASR checkpoint
    Select a different locally cached Whisper checkpoint for the intelligibility
    gate by configuring the validation transcriber (see
    ``garak.resources.audio.validation``).
