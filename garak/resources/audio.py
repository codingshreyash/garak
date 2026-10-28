"""Shared audio processing helpers."""

from io import BytesIO


def concatenate_wav(first: bytes, second: bytes, silence_ms: int = 0) -> bytes:
    """Return two compatible WAV byte strings joined with optional silence."""

    if isinstance(silence_ms, bool) or not isinstance(silence_ms, int):
        raise TypeError("silence_ms must be an integer")
    if silence_ms < 0:
        raise ValueError("silence_ms must not be negative")

    import numpy as np
    import soundfile

    first_audio, first_rate = soundfile.read(BytesIO(first), dtype="float32")
    second_audio, second_rate = soundfile.read(BytesIO(second), dtype="float32")
    if first_rate != second_rate:
        raise ValueError("WAV sample rates must match")
    if first_audio.ndim != second_audio.ndim or (
        first_audio.ndim == 2 and first_audio.shape[1] != second_audio.shape[1]
    ):
        raise ValueError("WAV channel layouts must match")

    silence_shape = (round(first_rate * silence_ms / 1000),)
    if first_audio.ndim == 2:
        silence_shape += (first_audio.shape[1],)
    silence = np.zeros(silence_shape, dtype=np.float32)
    combined = np.concatenate((first_audio, silence, second_audio), axis=0)

    output = BytesIO()
    soundfile.write(output, combined, first_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()
