import math
import wave

import numpy
import pytest

from garak.resources.audio.transforms import (
    apply_transform_recipe,
    read_pcm16_wav,
)


def _write_tone(path, *, sample_rate=8000, duration=0.2):
    frame_count = round(sample_rate * duration)
    samples = numpy.array(
        [
            math.sin(2 * math.pi * 440 * index / sample_rate)
            for index in range(frame_count)
        ]
    )
    pcm = numpy.rint(samples * 10000).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def test_transform_recipe_is_ordered_and_records_provenance(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    _write_tone(source)

    provenance = apply_transform_recipe(
        source,
        output,
        [
            {"type": "gain", "decibels": -6},
            {"type": "silence", "start_ms": 50, "end_ms": 100},
        ],
    )

    assert output.is_file(), "writes transformed audio"
    assert (
        provenance["transformations"][0]["type"] == "gain"
    ), "preserves recipe ordering"
    assert provenance["output"]["duration_seconds"] == pytest.approx(
        0.35, abs=0.001
    ), "records transformed duration"
    assert (
        provenance["source"]["sha256"] != provenance["output"]["sha256"]
    ), "records distinct source and output hashes"


def test_noise_transform_is_seeded(tmp_path):
    source = tmp_path / "source.wav"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_tone(source)
    recipe = [{"type": "noise", "kind": "pink", "snr_db": 20, "seed": 7}]

    first_provenance = apply_transform_recipe(source, first, recipe)
    second_provenance = apply_transform_recipe(source, second, recipe)

    assert (
        first_provenance["output"]["sha256"] == second_provenance["output"]["sha256"]
    ), "seeded noise is reproducible"


@pytest.mark.parametrize("kind", ["white", "pink", "brown"])
def test_supported_noise_colours_produce_valid_wav(tmp_path, kind):
    source = tmp_path / "source.wav"
    output = tmp_path / f"{kind}.wav"
    _write_tone(source)

    apply_transform_recipe(
        source,
        output,
        [{"type": "noise", "kind": kind, "snr_db": 18, "seed": 1}],
    )

    assert len(read_pcm16_wav(output).samples) > 0, "noise output remains valid audio"


def test_speed_transform_changes_duration(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "fast.wav"
    _write_tone(source)

    provenance = apply_transform_recipe(
        source, output, [{"type": "speed", "factor": 2.0}]
    )

    assert provenance["output"]["duration_seconds"] == pytest.approx(
        0.1, abs=0.001
    ), "speed-up shortens audio"


def test_echo_and_micro_gaps_compose(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "edited.wav"
    _write_tone(source)

    provenance = apply_transform_recipe(
        source,
        output,
        [
            {"type": "micro_gaps", "gap_ms": 10, "interval_ms": 50},
            {"type": "echo", "delay_ms": 40, "decay": 0.3},
        ],
    )

    assert provenance["output"]["duration_seconds"] == pytest.approx(
        0.24, abs=0.001
    ), "echo extends edited audio"


def test_unknown_transform_fails_closed(tmp_path):
    source = tmp_path / "source.wav"
    _write_tone(source)

    with pytest.raises(ValueError, match="unknown audio transformation"):
        apply_transform_recipe(
            source,
            tmp_path / "output.wav",
            [{"type": "unbounded_magic"}],
        )


def test_concat_and_overlay_accept_resampled_audio(tmp_path):
    source = tmp_path / "source.wav"
    auxiliary = tmp_path / "auxiliary.wav"
    _write_tone(source, sample_rate=8000)
    _write_tone(auxiliary, sample_rate=16000)

    concatenated = tmp_path / "concatenated.wav"
    overlaid = tmp_path / "overlaid.wav"
    apply_transform_recipe(
        source,
        concatenated,
        [{"type": "concat", "path": str(auxiliary), "gain_db": -6.0}],
    )
    apply_transform_recipe(
        source,
        overlaid,
        [{"type": "overlay", "path": str(auxiliary), "gain_db": -6.0}],
    )

    assert (
        len(read_pcm16_wav(concatenated).samples) == 3200
    ), "concat should append the resampled auxiliary frames"
    assert (
        len(read_pcm16_wav(overlaid).samples) == 1600
    ), "overlay should preserve the longer input duration"


def test_stereo_split_preserves_independent_speakers(tmp_path):
    source = tmp_path / "source.wav"
    auxiliary = tmp_path / "auxiliary.wav"
    stereo = tmp_path / "stereo.wav"
    left = tmp_path / "left.wav"
    right = tmp_path / "right.wav"
    _write_tone(source, sample_rate=8000)
    _write_tone(auxiliary, sample_rate=16000)

    apply_transform_recipe(
        source,
        stereo,
        [{"type": "stereo_split", "path": str(auxiliary), "gain_db": -6.0}],
    )
    apply_transform_recipe(stereo, left, [{"type": "channel", "index": 0}])
    apply_transform_recipe(stereo, right, [{"type": "channel", "index": 1}])

    stereo_samples = read_pcm16_wav(stereo).samples
    left_samples = read_pcm16_wav(left).samples
    right_samples = read_pcm16_wav(right).samples
    assert stereo_samples.shape == (1600, 2), "stores one speaker in each channel"
    assert (
        left_samples.ndim == 1 and right_samples.ndim == 1
    ), "channel extraction produces independently validatable mono audio"
    assert numpy.sqrt(numpy.mean(numpy.square(right_samples))) < numpy.sqrt(
        numpy.mean(numpy.square(left_samples))
    ), "applies the secondary-speaker gain only to the right channel"


def test_channel_extraction_requires_stereo_input(tmp_path):
    source = tmp_path / "source.wav"
    _write_tone(source)

    with pytest.raises(ValueError, match="requires stereo"):
        apply_transform_recipe(
            source,
            tmp_path / "channel.wav",
            [{"type": "channel", "index": 0}],
        )


def test_limits_fail_without_replacing_existing_output(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    _write_tone(source)
    output.write_bytes(b"existing")

    with pytest.raises(ValueError, match="duration limit"):
        apply_transform_recipe(
            source,
            output,
            [{"type": "silence", "end_ms": 1000}],
            max_duration_seconds=0.2,
        )

    assert (
        output.read_bytes() == b"existing"
    ), "failed transforms must not replace an existing asset"


def test_empty_wav_is_rejected(tmp_path):
    import wave

    source = tmp_path / "empty.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"")

    with pytest.raises(ValueError, match="non-empty"):
        apply_transform_recipe(source, tmp_path / "output.wav", [])
