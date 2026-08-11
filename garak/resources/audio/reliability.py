# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-speech reliability metadata for audio probes."""

from dataclasses import dataclass
from pathlib import Path

from garak.resources.audio.source import (
    load_audio_source,
    required_records,
    required_text,
    text_groups,
)


@dataclass(frozen=True)
class TTSReliabilityCase:
    """Benign text-to-speech reliability case metadata."""

    case_id: str
    source_text: str
    boundary_symbols: tuple[str, ...]
    required_transcript_terms: tuple[tuple[str, ...], ...]
    forbidden_transcript_terms: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class SemanticReliabilityCase:
    """Benign spoken-request comprehension case metadata."""

    case_id: str
    source_text: str
    required_response_terms: tuple[tuple[str, ...], ...]
    forbidden_response_terms: tuple[str, ...] = ()
    rationale: str = ""


def _string_list(record: dict, key: str, context: str) -> tuple[str, ...]:
    value = record.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{context} {key} must contain strings")
    return tuple(value)


def _rationale(record: dict, context: str) -> str:
    value = record.get("rationale", "")
    if not isinstance(value, str):
        raise ValueError(f"{context} rationale must be text")
    return value


def _load_reliability_source(source_data_path: str | Path | None) -> dict:
    return load_audio_source("reliability.json", source_data_path)


def tts_reliability_cases(
    source_data_path: str | Path | None = None,
) -> tuple[TTSReliabilityCase, ...]:
    """Return validated text-to-speech reliability canary cases."""

    source = _load_reliability_source(source_data_path)
    cases = []
    seen_ids = set()
    for index, record in enumerate(
        required_records(source, "tts_cases", "audio reliability source data")
    ):
        context = f"TTS reliability case {index}"
        case_id = required_text(record, "case_id", context)
        if case_id in seen_ids:
            raise ValueError(f"duplicate TTS reliability case_id: {case_id}")
        seen_ids.add(case_id)
        cases.append(
            TTSReliabilityCase(
                case_id=case_id,
                source_text=required_text(record, "source_text", context),
                boundary_symbols=_string_list(record, "boundary_symbols", context),
                required_transcript_terms=text_groups(
                    record.get("required_transcript_terms"),
                    f"{context} required_transcript_terms",
                ),
                forbidden_transcript_terms=_string_list(
                    record, "forbidden_transcript_terms", context
                ),
                rationale=_rationale(record, context),
            )
        )
    return tuple(cases)


def semantic_reliability_cases(
    source_data_path: str | Path | None = None,
) -> tuple[SemanticReliabilityCase, ...]:
    """Return validated spoken-request comprehension cases."""

    source = _load_reliability_source(source_data_path)
    cases = []
    seen_ids = set()
    for index, record in enumerate(
        required_records(source, "semantic_cases", "audio reliability source data")
    ):
        context = f"semantic reliability case {index}"
        case_id = required_text(record, "case_id", context)
        if case_id in seen_ids:
            raise ValueError(f"duplicate semantic reliability case_id: {case_id}")
        seen_ids.add(case_id)
        cases.append(
            SemanticReliabilityCase(
                case_id=case_id,
                source_text=required_text(record, "source_text", context),
                required_response_terms=text_groups(
                    record.get("required_response_terms"),
                    f"{context} required_response_terms",
                ),
                forbidden_response_terms=_string_list(
                    record, "forbidden_response_terms", context
                ),
                rationale=_rationale(record, context),
            )
        )
    return tuple(cases)
