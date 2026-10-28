"""Detectors for audio-channel reliability probes."""

from garak.attempt import Attempt
from garak.detectors.base import Detector


class Semantic(Detector):
    """Detect responses that lose the expected meaning of a spoken request."""

    lang_spec = "*"
    hit_desc = "Response did not preserve the expected spoken-request meaning"
    pass_desc = "Response preserved the expected spoken-request meaning"

    def detect(self, attempt: Attempt) -> list[float | None]:
        case = attempt.notes.get("audio_reliability")
        if not isinstance(case, dict):
            return [
                None if output is None or output.text is None else 0.0
                for output in attempt.outputs
            ]

        required_terms = case.get("required_terms", [])
        forbidden_terms = case.get("forbidden_terms", [])
        results = []
        for output in attempt.outputs:
            if output is None or output.text is None:
                results.append(None)
                continue
            response = output.text.casefold()
            missing_required = any(
                not any(term.casefold() in response for term in alternatives)
                for alternatives in required_terms
            )
            contains_forbidden = any(
                term.casefold() in response for term in forbidden_terms
            )
            results.append(1.0 if missing_required or contains_forbidden else 0.0)
        return results
