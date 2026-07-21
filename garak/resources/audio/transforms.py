# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable, dependency-light WAV transformations for audio probes."""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import wave

from garak.resources.audio.attack import audio_file_metadata, recipe_digest


@dataclass(frozen=True)
class Waveform:
    """Normalised audio samples and their sample rate."""

    samples: object
    sample_rate: int


def read_pcm16_wav(path: str | Path) -> Waveform:
    """Read a mono or stereo 16-bit PCM WAV into normalised samples."""

    import numpy

    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("audio transforms require 16-bit PCM WAV input")
        channels = wav_file.getnchannels()
        if channels not in (1, 2):
            raise ValueError("audio transforms support mono or stereo WAV input")
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    if not raw:
        raise ValueError("audio transforms require non-empty WAV input")
    samples = numpy.frombuffer(raw, dtype="<i2").astype(numpy.float64) / 32768.0
    if channels == 2:
        samples = samples.reshape((-1, 2))
    return Waveform(samples=samples, sample_rate=sample_rate)


def write_pcm16_wav(path: str | Path, waveform: Waveform) -> None:
    """Write normalised mono or stereo samples as 16-bit PCM WAV."""

    import numpy

    samples = numpy.asarray(waveform.samples, dtype=numpy.float64)
    if samples.ndim not in (1, 2) or (samples.ndim == 2 and samples.shape[1] != 2):
        raise ValueError("audio transforms produce mono or stereo samples")
    pcm = numpy.rint(numpy.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    output_path = Path(path)
    output_path.parent.mkdir(mode=0o740, parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1 if pcm.ndim == 1 else 2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(waveform.sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _gain(samples, decibels: float):
    return samples * (10.0 ** (float(decibels) / 20.0))


def _speed(samples, factor: float):
    import numpy

    factor = float(factor)
    if factor <= 0:
        raise ValueError("speed factor must be positive")
    frame_count = len(samples)
    if frame_count == 0:
        return samples.copy()
    output_frames = max(1, round(frame_count / factor))
    positions = numpy.linspace(0, frame_count - 1, output_frames)
    source_positions = numpy.arange(frame_count)
    if samples.ndim == 1:
        return numpy.interp(positions, source_positions, samples)
    return numpy.column_stack(
        [
            numpy.interp(positions, source_positions, samples[:, channel])
            for channel in range(samples.shape[1])
        ]
    )


def _noise(samples, sample_rate: int, kind: str, snr_db: float, seed: int):
    import numpy

    generator = numpy.random.default_rng(int(seed))
    shape = samples.shape
    white = generator.normal(0.0, 1.0, size=shape)
    if kind == "white":
        noise = white
    elif kind in ("pink", "brown"):
        frequency = numpy.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
        scale = numpy.ones_like(frequency)
        nonzero = frequency > 0
        exponent = 0.5 if kind == "pink" else 1.0
        scale[nonzero] = 1.0 / numpy.power(frequency[nonzero], exponent)
        scale[~nonzero] = 0.0
        if samples.ndim == 1:
            noise = numpy.fft.irfft(numpy.fft.rfft(white) * scale, n=len(samples))
        else:
            noise = numpy.column_stack(
                [
                    numpy.fft.irfft(
                        numpy.fft.rfft(white[:, channel]) * scale,
                        n=len(samples),
                    )
                    for channel in range(samples.shape[1])
                ]
            )
    else:
        raise ValueError("noise kind must be white, pink, or brown")
    signal_rms = float(numpy.sqrt(numpy.mean(numpy.square(samples))))
    noise_rms = float(numpy.sqrt(numpy.mean(numpy.square(noise))))
    if signal_rms == 0.0 or noise_rms == 0.0:
        return samples.copy()
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    return samples + noise * (target_noise_rms / noise_rms)


def _silence(samples, sample_rate: int, start_ms: int, end_ms: int):
    import numpy

    if start_ms < 0 or end_ms < 0:
        raise ValueError("silence durations must be non-negative")
    tail_shape = samples.shape[1:] if samples.ndim == 2 else ()
    start = numpy.zeros((round(sample_rate * start_ms / 1000),) + tail_shape)
    end = numpy.zeros((round(sample_rate * end_ms / 1000),) + tail_shape)
    return numpy.concatenate((start, samples, end), axis=0)


def _micro_gaps(samples, sample_rate: int, gap_ms: int, interval_ms: int):
    result = samples.copy()
    if gap_ms < 1 or interval_ms < 1:
        raise ValueError("gap and interval durations must be positive")
    gap_frames = round(sample_rate * gap_ms / 1000)
    interval_frames = round(sample_rate * interval_ms / 1000)
    for start in range(interval_frames, len(result), interval_frames):
        result[start : start + gap_frames] = 0.0
    return result


def _echo(samples, sample_rate: int, delay_ms: int, decay: float):
    import numpy

    if delay_ms < 1 or not 0.0 <= float(decay) <= 1.0:
        raise ValueError("echo requires positive delay and decay between zero and one")
    delay_frames = round(sample_rate * delay_ms / 1000)
    tail_shape = samples.shape[1:] if samples.ndim == 2 else ()
    result = numpy.zeros((len(samples) + delay_frames,) + tail_shape)
    result[: len(samples)] += samples
    result[delay_frames:] += samples * float(decay)
    return result


def _bandpass(samples, sample_rate: int, low_hz: float, high_hz: float):
    import numpy

    low_hz = float(low_hz)
    high_hz = float(high_hz)
    if low_hz < 0 or high_hz <= low_hz or high_hz > sample_rate / 2:
        raise ValueError("bandpass frequencies must fit within the Nyquist range")
    frequencies = numpy.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    keep = (frequencies >= low_hz) & (frequencies <= high_hz)
    if samples.ndim == 1:
        spectrum = numpy.fft.rfft(samples)
        spectrum[~keep] = 0.0
        return numpy.fft.irfft(spectrum, n=len(samples))
    return numpy.column_stack(
        [
            numpy.fft.irfft(
                numpy.where(keep, numpy.fft.rfft(samples[:, channel]), 0.0),
                n=len(samples),
            )
            for channel in range(samples.shape[1])
        ]
    )


def _match_channels(samples, target_channels: int):
    import numpy

    if target_channels == 1 and samples.ndim == 2:
        return samples.mean(axis=1)
    if target_channels == 2 and samples.ndim == 1:
        return numpy.column_stack((samples, samples))
    return samples


def _resample(samples, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return samples
    return _speed(samples, source_rate / target_rate)


def _combine(samples, sample_rate: int, path: str, mode: str, gain_db: float):
    import numpy

    auxiliary = read_pcm16_wav(path)
    other = _resample(auxiliary.samples, auxiliary.sample_rate, sample_rate)
    other = _match_channels(other, 2 if samples.ndim == 2 else 1)
    other = _gain(other, gain_db)
    if mode == "concat":
        return numpy.concatenate((samples, other), axis=0)
    frame_count = max(len(samples), len(other))
    tail_shape = samples.shape[1:] if samples.ndim == 2 else ()
    result = numpy.zeros((frame_count,) + tail_shape)
    result[: len(samples)] += samples
    result[: len(other)] += other
    return result


def _stereo_split(samples, sample_rate: int, path: str, gain_db: float):
    import numpy

    primary = samples.mean(axis=1) if samples.ndim == 2 else samples
    auxiliary = read_pcm16_wav(path)
    secondary = _resample(auxiliary.samples, auxiliary.sample_rate, sample_rate)
    if secondary.ndim == 2:
        secondary = secondary.mean(axis=1)
    secondary = _gain(secondary, gain_db)
    frame_count = max(len(primary), len(secondary))
    result = numpy.zeros((frame_count, 2))
    result[: len(primary), 0] = primary
    result[: len(secondary), 1] = secondary
    return result


def _channel(samples, index: int):
    index = int(index)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("channel extraction requires stereo audio")
    if index not in (0, 1):
        raise ValueError("channel index must be zero or one")
    return samples[:, index]


def apply_transform_recipe(
    source_path: str | Path,
    output_path: str | Path,
    transformations: list[dict] | tuple[dict, ...],
    *,
    max_duration_seconds: float | None = None,
    max_byte_size: int | None = None,
) -> dict:
    """Apply an ordered transform recipe and return output provenance."""

    waveform = read_pcm16_wav(source_path)
    samples = waveform.samples
    applied = []
    for operation in transformations:
        if not isinstance(operation, dict) or "type" not in operation:
            raise ValueError("each audio transformation requires a type")
        transform_type = operation["type"]
        parameters = {key: value for key, value in operation.items() if key != "type"}
        if transform_type == "gain":
            samples = _gain(samples, parameters["decibels"])
        elif transform_type == "speed":
            samples = _speed(samples, parameters["factor"])
        elif transform_type == "noise":
            samples = _noise(
                samples,
                waveform.sample_rate,
                parameters["kind"],
                parameters["snr_db"],
                parameters.get("seed", 0),
            )
        elif transform_type == "silence":
            samples = _silence(
                samples,
                waveform.sample_rate,
                parameters.get("start_ms", 0),
                parameters.get("end_ms", 0),
            )
        elif transform_type == "micro_gaps":
            samples = _micro_gaps(
                samples,
                waveform.sample_rate,
                parameters["gap_ms"],
                parameters["interval_ms"],
            )
        elif transform_type == "echo":
            samples = _echo(
                samples,
                waveform.sample_rate,
                parameters["delay_ms"],
                parameters["decay"],
            )
        elif transform_type == "bandpass":
            samples = _bandpass(
                samples,
                waveform.sample_rate,
                parameters["low_hz"],
                parameters["high_hz"],
            )
        elif transform_type in ("concat", "overlay"):
            samples = _combine(
                samples,
                waveform.sample_rate,
                parameters["path"],
                transform_type,
                parameters.get("gain_db", 0.0),
            )
        elif transform_type == "stereo_split":
            samples = _stereo_split(
                samples,
                waveform.sample_rate,
                parameters["path"],
                parameters.get("gain_db", 0.0),
            )
        elif transform_type == "channel":
            samples = _channel(samples, parameters["index"])
        else:
            raise ValueError(f"unknown audio transformation: {transform_type}")
        applied.append({"type": transform_type, **parameters})
    duration_seconds = len(samples) / waveform.sample_rate
    if max_duration_seconds is not None and duration_seconds > max_duration_seconds:
        raise ValueError("transformed audio exceeds the configured duration limit")
    channel_count = 1 if samples.ndim == 1 else samples.shape[1]
    estimated_byte_size = 44 + len(samples) * channel_count * 2
    if max_byte_size is not None and estimated_byte_size > max_byte_size:
        raise ValueError("transformed audio exceeds the configured byte-size limit")

    output_path = Path(output_path)
    output_path.parent.mkdir(mode=0o740, parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        write_pcm16_wav(temporary_path, Waveform(samples, waveform.sample_rate))
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {
        "source": audio_file_metadata(source_path),
        "output": audio_file_metadata(output_path),
        "transformations": applied,
        "recipe_digest": recipe_digest(applied),
    }
