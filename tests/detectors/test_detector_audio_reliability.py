"""Tests for audio-channel reliability detection."""

from garak.attempt import Attempt, Conversation, Message, Turn
from garak.detectors.audio_reliability import Semantic


def _attempt(*outputs):
    prompt = Conversation([Turn("user", Message("reliability prompt"))])
    attempt = Attempt(prompt=prompt, notes={})
    attempt.notes["audio_reliability"] = {
        "required_terms": [["canary"], ["delta", "change"]],
        "forbidden_terms": ["echo"],
    }
    attempt.outputs = [
        Message(output) if output is not None else None for output in outputs
    ]
    return attempt


def test_semantic_reliability_passes_expected_meaning():
    """Any alternative in every required group satisfies the case."""

    detector = Semantic()

    assert detector.detect(_attempt("Canary change")) == [
        0.0
    ], "a complete expected response must pass"


def test_semantic_reliability_flags_missing_or_forbidden_terms():
    """Missing expectations and forbidden content are reliability failures."""

    detector = Semantic()

    assert detector.detect(_attempt("canary", "canary delta echo")) == [
        1.0,
        1.0,
    ], "missing or forbidden terms must be reported as hits"


def test_semantic_reliability_preserves_none_outputs():
    """Unavailable target responses remain unscored."""

    detector = Semantic()

    assert detector.detect(_attempt(None)) == [None], "None responses must remain None"


def test_semantic_reliability_requires_probe_metadata():
    """The detector does not score attempts from unrelated probes."""

    prompt = Conversation([Turn("user", Message("unrelated prompt"))])
    attempt = Attempt(prompt=prompt, notes={})
    attempt.outputs = [Message("blue")]

    assert Semantic().detect(attempt) == [
        0.0
    ], "unrelated text responses must remain a non-hit"
