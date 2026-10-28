"""Tests for audio instruction injection probes."""

from io import BytesIO

import numpy as np
import pytest
import soundfile

from garak.attempt import Message
from garak.probes.audio_injection import Suffix


class FakeTTS:
    """Render deterministic WAV clips for probe tests."""

    def generate(self, prompt):
        text = prompt.last_message().text
        value = 0.25 if text.startswith("This is") else -0.25
        output = BytesIO()
        soundfile.write(
            output, np.full(8, value, dtype=np.float32), 8_000, format="WAV"
        )
        message = Message(data_type=("audio/wav", None))
        message.data = output.getvalue()
        return [message]


def test_suffix_concatenates_carrier_before_request(mocker, loaded_intent_service):
    """The target receives one WAV containing carrier, gap, then request."""

    mocker.patch("garak.probes.audio._plugins.load_plugin", return_value=FakeTTS())
    config = {"probes": {"audio_injection": {"Suffix": {"suffix_silence_ms": 1}}}}
    probe = Suffix(config_root=config)

    prepared = probe._prepare_prompt("spoken request")
    audio, sample_rate = soundfile.read(BytesIO(prepared.last_message().data))

    assert sample_rate == 8_000, "the helper sample rate must be retained"
    assert len(audio) == 24, "the carrier, gap, and request must all be present"
    assert audio[0] > 0 and audio[-1] < 0, "the carrier must precede the request"
    assert (
        prepared.last_message().notes["audio_injection"] == "suffix"
    ), "reports must identify the injection layout"


def test_suffix_rejects_empty_carrier(loaded_intent_service):
    """A suffix probe requires a meaningful benign carrier."""

    config = {"probes": {"audio_injection": {"Suffix": {"carrier_text": " "}}}}

    with pytest.raises(ValueError, match="carrier_text"):
        Suffix(config_root=config)


@pytest.mark.parametrize("silence_ms", [-1, 1.5, True])
def test_suffix_validates_silence_configuration(silence_ms, loaded_intent_service):
    """Invalid silence configuration fails before loading a TTS helper."""

    config = {
        "probes": {"audio_injection": {"Suffix": {"suffix_silence_ms": silence_ms}}}
    }

    with pytest.raises((TypeError, ValueError), match="suffix_silence_ms"):
        Suffix(config_root=config)
