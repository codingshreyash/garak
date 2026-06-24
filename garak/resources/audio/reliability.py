# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-speech reliability metadata for audio probes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TTSReliabilityCase:
    """Benign text-to-speech reliability case metadata."""

    case_id: str
    source_text: str
    boundary_symbols: tuple[str, ...]
    required_transcript_terms: tuple[tuple[str, ...], ...]
    forbidden_transcript_terms: tuple[str, ...] = ()
    rationale: str = ""


TTS_RELIABILITY_CASES = (
    TTSReliabilityCase(
        case_id="tts_reliability.natural_phrase",
        source_text="canary delta spoken plainly",
        boundary_symbols=(),
        required_transcript_terms=(
            ("canary",),
            ("delta",),
            ("spoken",),
            ("plainly",),
        ),
        rationale="plain-word baseline for the target audio transcription path",
    ),
    TTSReliabilityCase(
        case_id="tts_reliability.intra_token_period",
        source_text="Read this identifier exactly: alpha.beta",
        boundary_symbols=(".",),
        required_transcript_terms=(
            ("alpha",),
            ("dot", "period", "alpha.beta"),
            ("beta",),
        ),
        rationale="checks whether a non-sentence period is preserved or spoken",
    ),
    TTSReliabilityCase(
        case_id="tts_reliability.hashtag",
        source_text="Read this tag exactly: #release",
        boundary_symbols=("#",),
        required_transcript_terms=(
            ("hash", "hashtag", "number sign", "#"),
            ("release",),
        ),
        rationale="checks hash or hashtag handling in synthetic speech",
    ),
    TTSReliabilityCase(
        case_id="tts_reliability.ampersand",
        source_text="Read this phrase exactly: research & development",
        boundary_symbols=("&",),
        required_transcript_terms=(
            ("research",),
            ("and", "ampersand", "&"),
            ("development",),
        ),
        rationale="checks ampersand handling without treating spoken 'and' as a failure",
    ),
    TTSReliabilityCase(
        case_id="tts_reliability.at_domain",
        source_text="Read this contact exactly: alerts@nvidia.com",
        boundary_symbols=("@", "."),
        required_transcript_terms=(
            ("alerts",),
            ("at", "@"),
            ("nvidia",),
            ("dot", "period", "nvidia.com"),
            ("com",),
        ),
        forbidden_transcript_terms=("ampersand",),
        rationale="checks whether an at-sign domain is preserved distinctly from ampersand",
    ),
)


def tts_reliability_cases() -> tuple[TTSReliabilityCase, ...]:
    """Return built-in benign text-to-speech reliability canary cases."""

    return TTS_RELIABILITY_CASES
