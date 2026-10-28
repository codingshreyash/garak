"""Tests for shared audio processing helpers."""

from io import BytesIO

import numpy as np
import pytest
import soundfile

from garak.resources.audio import concatenate_wav


def _wav(samples, sample_rate=8_000):
    output = BytesIO()
    soundfile.write(
        output, np.asarray(samples, dtype=np.float32), sample_rate, format="WAV"
    )
    return output.getvalue()


def test_concatenate_wav_preserves_order_and_silence():
    """WAV concatenation retains both clips around the requested gap."""

    combined = concatenate_wav(_wav([0.5] * 8), _wav([-0.5] * 8), silence_ms=1)
    audio, sample_rate = soundfile.read(BytesIO(combined), dtype="float32")

    assert sample_rate == 8_000, "the source sample rate must be retained"
    assert len(audio) == 24, "one millisecond of silence must add eight samples"
    assert audio[0] > 0 and audio[-1] < 0, "clip order must be retained"
    assert np.allclose(audio[8:16], 0), "the configured gap must be silent"


def test_concatenate_wav_rejects_incompatible_rates():
    """Combining mismatched sample rates fails instead of distorting audio."""

    with pytest.raises(ValueError, match="sample rates must match"):
        concatenate_wav(_wav([0], 8_000), _wav([0], 16_000))


@pytest.mark.parametrize("silence_ms", [-1, 1.5, True])
def test_concatenate_wav_validates_silence(silence_ms):
    """Silence duration accepts only non-negative integer milliseconds."""

    with pytest.raises((TypeError, ValueError)):
        concatenate_wav(_wav([0]), _wav([0]), silence_ms=silence_ms)
