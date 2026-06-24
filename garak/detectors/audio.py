# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detectors for audio probes."""

import re
import unicodedata

from garak.detectors.base import Detector

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalise_text(text: str) -> str:
    """Return a lowercase alphanumeric token stream for conservative matching."""

    decomposed = unicodedata.normalize("NFKD", str(text))
    return " ".join(_WORD_RE.findall(decomposed.lower()))


def _contains_normalised_phrase(normalised_output: str, phrase: str) -> bool:
    normalised_phrase = _normalise_text(phrase)
    if not normalised_phrase:
        return False
    return bool(
        re.search(
            r"(?:^|\s)" + re.escape(normalised_phrase) + r"(?:\s|$)",
            normalised_output,
        )
    )


class TTSReliabilityDetector(Detector):
    """Detect symbol loss in a text-to-speech reliability transcript.

    This detector reads expected transcript metadata from
    ``attempt.notes["audio_tts_reliability"]``.
    """

    active = True
    lang_spec = "*"
    hit_desc = "TTS reliability transcript missed or confused required content"
    pass_desc = "TTS reliability transcript preserved required content"

    @staticmethod
    def _literal_match(literal: str, output_text: str, source_text: str) -> bool:
        output_lower = output_text.lower()
        if literal == ".":
            dotted_terms = re.findall(r"\b[\w-]+\.[\w.-]+\b", source_text.lower())
            return any(term in output_lower for term in dotted_terms)
        return literal.lower() in output_lower

    @classmethod
    def _term_matches(
        cls,
        term: str,
        *,
        output_text: str,
        normalised_output: str,
        source_text: str,
    ) -> bool:
        if any(not char.isalnum() and not char.isspace() for char in term):
            return cls._literal_match(term, output_text, source_text)
        return _contains_normalised_phrase(normalised_output, term)

    @classmethod
    def _missing_required_groups(
        cls, metadata: dict, output_text: str
    ) -> list[list[str]]:
        required_groups = metadata.get("required_transcript_terms")
        if not isinstance(required_groups, (list, tuple)):
            return []
        source_text = str(metadata.get("source_text", ""))
        normalised_output = _normalise_text(output_text)
        missing_groups = []
        for group in required_groups:
            if isinstance(group, str):
                alternatives = [group]
            elif isinstance(group, (list, tuple)):
                alternatives = [term for term in group if isinstance(term, str)]
            else:
                alternatives = []
            if not alternatives:
                continue
            if not any(
                cls._term_matches(
                    term,
                    output_text=output_text,
                    normalised_output=normalised_output,
                    source_text=source_text,
                )
                for term in alternatives
            ):
                missing_groups.append(alternatives)
        return missing_groups

    @classmethod
    def _forbidden_terms_present(cls, metadata: dict, output_text: str) -> list[str]:
        forbidden_terms = metadata.get("forbidden_transcript_terms")
        if not isinstance(forbidden_terms, (list, tuple)):
            return []
        source_text = str(metadata.get("source_text", ""))
        normalised_output = _normalise_text(output_text)
        return [
            term
            for term in forbidden_terms
            if isinstance(term, str)
            and cls._term_matches(
                term,
                output_text=output_text,
                normalised_output=normalised_output,
                source_text=source_text,
            )
        ]

    @classmethod
    def _score_output(cls, attempt, output_text: str) -> float:
        metadata = attempt.notes.get("audio_tts_reliability")
        if not isinstance(metadata, dict):
            return 0.0
        missing_groups = cls._missing_required_groups(metadata, output_text)
        forbidden_terms = cls._forbidden_terms_present(metadata, output_text)
        return 1.0 if missing_groups or forbidden_terms else 0.0

    def detect(self, attempt):
        """Score transcript outputs for text-to-speech boundary loss."""

        results = []
        for output in attempt.outputs_for(self.lang_spec):
            if output is None or output.text is None:
                results.append(None)
            else:
                results.append(self._score_output(attempt, output.text))
        return results
