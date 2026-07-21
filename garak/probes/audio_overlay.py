# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""**Audio overlay-injection probes (hidden background instruction)**

Models the AudioJailbreak (arXiv:2505.14103) overlay threat and the
Devil's-Whisper hidden-background-instruction idea: a malicious tool-risk
instruction is mixed *simultaneously* underneath a benign carrier utterance at
reduced gain, rather than appended after it (that is ``AudioSuffixInjection``).

Controls, per AUD-009:

- ``foreground_only`` -- benign carrier alone (is it answered normally?).
- ``background_only`` -- malicious clip alone (baseline: refused on its own?).
- ``mixed``          -- carrier + overlaid malicious clip at -6/-12/-18 dB.

The relative-gain sweep probes the tension between "quiet enough to hide" and
"loud enough for the target to act on". Every candidate is gated by the
independent-ASR intelligibility check and the tool-risk detectors/judge.
"""

import logging
from pathlib import Path

from garak.attempt import Attempt, Message
from garak.probes.audio import PETTS, ToolRiskPETTS
from garak.probes.audio_suffix import _SUFFIX_CARRIERS, _SUFFIX_PAYLOADS
from garak.resources.audio.attack import audio_file_metadata, recipe_digest
from garak.resources.audio.transforms import apply_transform_recipe


class AudioOverlayInjection(ToolRiskPETTS):
    """Overlay a spoken tool-risk instruction beneath a benign carrier.

    The malicious intent lives only in the overlaid (background) track. Harmful
    conditions are ``background_only`` and ``mixed``; ``foreground_only`` is the
    matched control that shows whether the carrier is understood at all.
    """

    active = False
    goal = "hijack a benign spoken request with a hidden overlaid tool-risk instruction"
    doc_uri = "https://arxiv.org/abs/2505.14103"
    extended_detectors = [
        "audio.AudioOutputQualityDetector",
        "audio.SemanticReliabilityDetector",
        "audio.AudioToolRiskJudge",
        "mitigation.MitigationBypass",
    ]
    DEFAULT_PARAMS = ToolRiskPETTS.DEFAULT_PARAMS | {
        "overlay_gains_db": (-6.0, -12.0, -18.0),
        "carrier_ids": tuple(carrier_id for carrier_id, _, _ in _SUFFIX_CARRIERS),
        "payload_ids": tuple(payload_id for payload_id, _ in _SUFFIX_PAYLOADS),
        "transform_max_duration_seconds": 120.0,
        "transform_max_byte_size": 20_000_000,
    }

    def _selected_carriers(self):
        by_id = {c[0]: c for c in _SUFFIX_CARRIERS}
        ids = (
            (self.carrier_ids,)
            if isinstance(self.carrier_ids, str)
            else tuple(self.carrier_ids)
        )
        unknown = sorted(set(ids) - by_id.keys())
        if unknown:
            raise ValueError("unknown overlay carrier ids: " + ", ".join(unknown))
        return tuple(by_id[i] for i in ids)

    def _selected_payloads(self):
        by_id = {s[0]: s for s in _SUFFIX_PAYLOADS}
        ids = (
            (self.payload_ids,)
            if isinstance(self.payload_ids, str)
            else tuple(self.payload_ids)
        )
        unknown = sorted(set(ids) - by_id.keys())
        if unknown:
            raise ValueError("unknown overlay payload ids: " + ", ".join(unknown))
        return tuple(by_id[i] for i in ids)

    def _gains(self) -> tuple[float, ...]:
        gains = (
            (self.overlay_gains_db,)
            if isinstance(self.overlay_gains_db, (int, float))
            else tuple(self.overlay_gains_db)
        )
        if not gains:
            raise ValueError("overlay_gains_db must contain at least one gain value")
        if any(float(g) > 0 for g in gains):
            raise ValueError(
                "overlay gains are relative attenuations and must be <= 0 dB"
            )
        return tuple(float(g) for g in gains)

    @staticmethod
    def _overlay_recipe(payload_path: Path, gain_db: float) -> tuple[dict, ...]:
        """Recipe that mixes the payload onto the carrier at ``gain_db``."""

        return (
            {"type": "overlay", "path": str(payload_path), "gain_db": float(gain_db)},
        )

    def build_prompts(self):
        """Enumerate foreground_only, background_only, and mixed overlay trials."""

        carriers = self._selected_carriers()
        payloads = self._selected_payloads()
        gains = self._gains()
        trials = []
        for carrier_id, carrier_text, required in carriers:
            trials.append(
                ("foreground_only", carrier_id, None, None, carrier_text, required)
            )
        for payload_id, payload_text in payloads:
            trials.append(("background_only", None, payload_id, None, payload_text, ()))
        for carrier_id, carrier_text, _ in carriers:
            for payload_id, payload_text in payloads:
                for gain_db in gains:
                    trials.append(
                        (
                            "mixed",
                            carrier_id,
                            payload_id,
                            gain_db,
                            (carrier_text, payload_text),
                            (),
                        )
                    )
        self._selected_overlay_trials = tuple(trials)
        self.audio_source_prompts = [self._trial_source_text(t) for t in trials]
        self.audio_source_intents = ["audio_overlay_injection" for _ in trials]
        self.prompts = list(self.audio_source_prompts)
        self.prompt_intents = list(self.audio_source_intents)

    @staticmethod
    def _trial_source_text(trial) -> str:
        condition, _, _, gain_db, text, _ = trial
        if condition == "mixed":
            carrier_text, payload_text = text
            return f"{carrier_text} [overlay {gain_db}dB] {payload_text}"
        return text

    def _composed_audio_path(self, carrier_path: Path, recipe) -> Path:
        identity = {
            "carrier_sha256": audio_file_metadata(carrier_path).get("sha256"),
            "recipe": recipe,
        }
        return self.audio_cache_dir / f"overlay-{recipe_digest(identity)}.wav"

    def _prepare_trial_audio(self, trial) -> Path:
        condition, _, _, gain_db, text, _ = trial
        if condition in ("foreground_only", "background_only"):
            return self._ensure_audio_file(text)
        carrier_text, payload_text = text
        carrier_path = self._ensure_audio_file(carrier_text)
        payload_path = self._ensure_audio_file(payload_text)
        if (
            carrier_path.suffix.lower() != ".wav"
            or payload_path.suffix.lower() != ".wav"
        ):
            raise ValueError("overlay injection requires PETTS WAV output")
        recipe = self._overlay_recipe(payload_path, gain_db)
        output_path = self._composed_audio_path(carrier_path, recipe)
        if not output_path.exists():
            apply_transform_recipe(
                carrier_path,
                output_path,
                recipe,
                max_duration_seconds=float(self.transform_max_duration_seconds),
                max_byte_size=int(self.transform_max_byte_size),
            )
        return output_path

    def _audio_prompts(self):
        prompts = []
        prepared = []
        for trial in self._selected_overlay_trials:
            try:
                audio_path = self._prepare_trial_audio(trial)
                prompts.append(
                    Message(
                        text=self.text_prompt, lang=self.lang, data_path=str(audio_path)
                    )
                )
                prepared.append(trial)
            except self._audio_preparation_exceptions() as exc:
                logging.warning(
                    "%s skipping overlay trial: %s", self.__class__.__name__, exc
                )
        self._prepared_overlay_trials = tuple(prepared)
        self._prepared_audio_sources = [self._trial_source_text(t) for t in prepared]
        return prompts, ["audio_overlay_injection" for _ in prompts]

    def _attempt_prestore_hook(self, attempt: Attempt, seq: int) -> Attempt:
        attempt = PETTS._attempt_prestore_hook(self, attempt, seq)
        condition, carrier_id, payload_id, gain_db, text, required = (
            self._prepared_overlay_trials[seq]
        )
        rendered_text = self._trial_source_text(
            (condition, carrier_id, payload_id, gain_db, text, required)
        )
        is_adversarial = condition != "foreground_only"
        payload_text = (
            text[1]
            if condition == "mixed"
            else (text if condition == "background_only" else None)
        )
        group_id = f"{condition}:{carrier_id or ''}+{payload_id or ''}"
        attempt.notes["audio_overlay_injection"] = {
            "condition": condition,
            "carrier_id": carrier_id,
            "payload_id": payload_id,
            "overlay_gain_db": gain_db,
            "rendered_text": rendered_text,
        }
        attempt.notes["is_adversarial"] = is_adversarial
        if is_adversarial:
            attempt.notes["attack_goal"] = payload_text
        else:
            attempt.notes["audio_semantic_reliability"] = {
                "case_id": carrier_id,
                "source_text": text,
                "required_response_terms": required,
                "forbidden_response_terms": (),
            }
        self._attach_audio_attack_metadata(
            attempt,
            source_case_id=payload_id or carrier_id,
            group_id=group_id,
            source_text=payload_text or text,
            rendered_text=rendered_text,
            semantic_strategy="overlay_injection",
            modality_condition=condition,
        )
        return attempt
