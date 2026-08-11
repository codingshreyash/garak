# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from garak.generators.base import Generator


class _FormatAwareGenerator(Generator):
    audio_formats = {"test-audio"}
    image_formats = {"test-image"}


def test_supported_formats_reports_declared_modality_contract():
    formats = _FormatAwareGenerator.supported_formats("audio")

    assert formats == {"test-audio"}, "reports formats declared for the modality"
    formats.add("caller-only")
    assert _FormatAwareGenerator.audio_formats == {
        "test-audio"
    }, "returns a copy so callers cannot mutate generator metadata"


def test_supported_formats_returns_empty_set_for_unknown_modality():
    assert (
        _FormatAwareGenerator.supported_formats("spatial") == set()
    ), "unknown modalities report no supported formats"
