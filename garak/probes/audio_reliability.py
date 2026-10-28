"""Audio-channel reliability probes."""

import json
from pathlib import Path

from garak import _config
from garak.attempt import Attempt
from garak.data import path as data_path
from garak.probes.audio import PETTS


class Semantic(PETTS):
    """Check whether a target understands benign requests rendered as speech.

    Render validated benign cases with PETTS and retain their expected response
    terms for the paired semantic reliability detector.
    """

    active = False
    lang = "*"
    primary_detector = "audio_reliability.Semantic"
    goal = "measure semantic preservation across text-to-speech audio input"
    doc_uri = "https://arxiv.org/abs/2410.23861"
    tags = [
        "avid-effect:performance:P0204",  # measures accuracy on spoken requests
        "quality:Security:PromptStability",  # checks semantic stability after speech rendering
    ]

    DEFAULT_PARAMS = PETTS.DEFAULT_PARAMS | {
        "source_data_path": None,
    }

    def _populate_intents(self) -> None:
        self.intents = set()

    def _populate_stubs(self) -> None:
        self.stubs = []
        self.stub_intents = []

    def _source_path(self) -> Path:
        if self.source_data_path is not None:
            return Path(self.source_data_path)
        return data_path / "audio" / "semantic_reliability.json"

    def _load_cases(self) -> list[dict]:
        with self._source_path().open(encoding="utf-8") as source_file:
            cases = json.load(source_file)
        if not isinstance(cases, list) or not cases:
            raise ValueError("audio reliability data must be a non-empty list")

        required_keys = {"id", "prompt", "required_terms", "forbidden_terms"}
        seen_ids = set()
        for case in cases:
            if not isinstance(case, dict) or not required_keys.issubset(case):
                raise ValueError(
                    "audio reliability cases require id, prompt, and terms"
                )
            if not isinstance(case["id"], str) or not case["id"]:
                raise ValueError("audio reliability case IDs must be non-empty text")
            if case["id"] in seen_ids:
                raise ValueError(f"duplicate audio reliability case ID: {case['id']}")
            seen_ids.add(case["id"])
            if not isinstance(case["prompt"], str) or not case["prompt"].strip():
                raise ValueError("audio reliability prompts must be non-empty text")
            if not all(
                isinstance(group, list)
                and group
                and all(isinstance(term, str) and term for term in group)
                for group in case["required_terms"]
            ):
                raise ValueError("required_terms must contain non-empty term groups")
            if not isinstance(case["forbidden_terms"], list) or not all(
                isinstance(term, str) and term for term in case["forbidden_terms"]
            ):
                raise ValueError("forbidden_terms must contain text values")
        return cases

    def build_prompts(self):
        """Build prompts from validated reliability cases."""

        self._reliability_cases = self._load_cases()
        self.prompts = [case["prompt"] for case in self._reliability_cases]
        self.prompt_intents = [case["id"] for case in self._reliability_cases]

    def _attempt_prestore_hook(self, attempt: Attempt, seq: int) -> Attempt:
        attempt = super()._attempt_prestore_hook(attempt, seq)
        case = self._reliability_cases[seq]
        attempt.notes["audio_reliability"] = {
            "case_id": case["id"],
            "required_terms": case["required_terms"],
            "forbidden_terms": case["forbidden_terms"],
        }
        return attempt
