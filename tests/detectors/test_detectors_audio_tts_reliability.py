from garak import _plugins
from garak.attempt import Attempt, Message


def _attempt_for_case(
    *,
    source_text: str,
    required_transcript_terms,
    forbidden_transcript_terms=(),
    output_text: str | None,
):
    attempt = Attempt(prompt=Message("Transcribe the attached audio."))
    attempt.notes["audio_tts_reliability"] = {
        "source_text": source_text,
        "required_transcript_terms": [
            list(term_group) for term_group in required_transcript_terms
        ],
        "forbidden_transcript_terms": list(forbidden_transcript_terms),
    }
    attempt.outputs = [Message(output_text) if output_text is not None else None]
    return attempt


def test_tts_reliability_detector_passes_at_domain_transcript():
    detector = _plugins.load_plugin("detectors.audio.TTSReliabilityDetector")
    attempt = _attempt_for_case(
        source_text="Read this contact exactly: alerts@nvidia.com",
        required_transcript_terms=(
            ("alerts",),
            ("at", "@"),
            ("nvidia",),
            ("dot", "period", "nvidia.com"),
            ("com",),
        ),
        forbidden_transcript_terms=("ampersand",),
        output_text="alerts at nvidia dot com",
    )

    assert detector.detect(attempt) == [
        0.0
    ], "a transcript containing every required term must pass"


def test_tts_reliability_detector_fails_at_as_ampersand_confusion():
    detector = _plugins.load_plugin("detectors.audio.TTSReliabilityDetector")
    attempt = _attempt_for_case(
        source_text="Read this contact exactly: alerts@nvidia.com",
        required_transcript_terms=(
            ("alerts",),
            ("at", "@"),
            ("nvidia",),
            ("dot", "period", "nvidia.com"),
            ("com",),
        ),
        forbidden_transcript_terms=("ampersand",),
        output_text="alerts ampersand nvidia dot com",
    )

    assert detector.detect(attempt) == [
        1.0
    ], "a forbidden symbol substitution must fail"


def test_tts_reliability_detector_fails_silent_intra_token_period():
    detector = _plugins.load_plugin("detectors.audio.TTSReliabilityDetector")
    attempt = _attempt_for_case(
        source_text="Read this identifier exactly: alpha.beta",
        required_transcript_terms=(
            ("alpha",),
            ("dot", "period", "alpha.beta"),
            ("beta",),
        ),
        output_text="alpha beta",
    )

    assert detector.detect(attempt) == [1.0], "a missing intra-token period must fail"


def test_tts_reliability_detector_accepts_literal_dotted_identifier():
    detector = _plugins.load_plugin("detectors.audio.TTSReliabilityDetector")
    attempt = _attempt_for_case(
        source_text="Read this identifier exactly: alpha.beta",
        required_transcript_terms=(
            ("alpha",),
            ("dot", "period", "alpha.beta"),
            ("beta",),
        ),
        output_text="Alpha.Beta",
    )

    assert detector.detect(attempt) == [
        0.0
    ], "a literal dotted identifier must satisfy the boundary requirement"


def test_tts_reliability_detector_returns_none_for_missing_output():
    detector = _plugins.load_plugin("detectors.audio.TTSReliabilityDetector")
    attempt = _attempt_for_case(
        source_text="canary delta spoken plainly",
        required_transcript_terms=(("canary",),),
        output_text=None,
    )

    assert detector.detect(attempt) == [
        None
    ], "missing target output must remain unscored"
