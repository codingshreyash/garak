Audio Synthesis Support
=======================

``probes.audio.PETTS`` renders text prompts to speech before calling an
audio-capable target. This modality change is part of the probe technique; the
generator remains responsible only for communicating with the target.

Provider interface
------------------

Synthesis is accessed through
``garak.resources.audio.synthesis.SynthesisProvider``. A provider declares its
capabilities and accepts a ``SynthesisRequest``, returning audio together with
the effective sample rate and provider identity.

The built-in ``TransformersSynthesisProvider`` loads a Transformers
``text-to-audio`` pipeline lazily. ``suno/bark-small`` is the default PETTS
checkpoint. It is publicly available under the MIT licence, but local model
download and inference can be substantial. Users should select a checkpoint or
implement a remote provider appropriate to their environment.

Configuration
-------------

PETTS exposes the following synthesis settings through garak's standard plugin
configuration:

* ``tts_model_name`` -- Transformers checkpoint used for synthesis
* ``tts_model_revision`` -- optional pinned checkpoint revision
* ``tts_voice`` -- optional provider-supported voice preset
* ``tts_sample_rate`` -- requested sample rate
* ``tts_audio_format`` -- output container, such as ``WAV``
* ``tts_audio_subtype`` -- optional output subtype
* ``tts_audio_stereo`` -- whether to render stereo output

The provider-reported effective sample rate and generated-file properties are
recorded separately from requested settings so evaluation provenance does not
claim that an unsupported request was honoured.

Audio utilities
---------------

``garak.resources.audio.attack`` records candidate and waveform provenance,
including a streaming SHA-256 digest. ``garak.resources.audio.transforms``
provides bounded PCM WAV transformations used by follow-on audio probes. These
utilities add no new project dependencies.

Licensing
---------

Any checkpoint configured as a project default must be publicly available and
licensed compatibly with garak's Apache-2.0 distribution. Remote services and
user-selected checkpoints may impose separate terms that the user must review.
