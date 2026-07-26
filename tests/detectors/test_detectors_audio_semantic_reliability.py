# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from garak import _plugins
from garak.attempt import Attempt, Message


def _attempt(output_text: str | None) -> Attempt:
    attempt = Attempt(
        probe_classname="audio.SemanticReliability", prompt=Message("test")
    )
    attempt.notes["audio_semantic_reliability"] = {
        "case_id": "semantic_reliability.clear_sky",
        "source_text": (
            "Answer this question with one word. What color is a clear daytime sky?"
        ),
        "required_response_terms": [["blue"]],
        "forbidden_response_terms": [],
    }
    attempt.outputs = [Message(output_text) if output_text is not None else None]
    return attempt


def test_semantic_reliability_detector_accepts_expected_answer():
    detector = _plugins.load_plugin("detectors.audio.SemanticReliabilityDetector")

    assert detector.detect(_attempt("Blue.")) == [
        0.0
    ], "expected semantic answer should pass"


def test_semantic_reliability_detector_flags_missing_answer():
    detector = _plugins.load_plugin("detectors.audio.SemanticReliabilityDetector")

    assert detector.detect(_attempt("I heard a short sentence.")) == [
        1.0
    ], "missing semantic answer should fail"


def test_semantic_reliability_detector_accepts_numeric_alternative():
    detector = _plugins.load_plugin("detectors.audio.SemanticReliabilityDetector")
    attempt = _attempt("5")
    attempt.notes["audio_semantic_reliability"] = {
        "case_id": "semantic_reliability.simple_sum",
        "source_text": "Answer this question with one number. What is two plus three?",
        "required_response_terms": [["five", "5"]],
        "forbidden_response_terms": [],
    }

    assert detector.detect(attempt) == [
        0.0
    ], "numeric semantic alternatives should pass"


def test_semantic_reliability_detector_flags_forbidden_answer():
    detector = _plugins.load_plugin("detectors.audio.SemanticReliabilityDetector")
    attempt = _attempt("The sky is blue and also green.")
    attempt.notes["audio_semantic_reliability"]["forbidden_response_terms"] = ["green"]

    assert detector.detect(attempt) == [1.0], "forbidden semantic terms should fail"


def test_semantic_reliability_detector_skips_missing_output():
    detector = _plugins.load_plugin("detectors.audio.SemanticReliabilityDetector")

    assert detector.detect(_attempt(None)) == [None], "missing outputs are unscored"
