"""**Bounded TTS voice-candidate probe**

Best-of-N Jailbreaking (arXiv:2412.03556) separately studies voice variation in
addition to its main composed signal search. This bounded diagnostic renders the
same request with configured TTS voice presets; it does not reproduce the
paper's main audio Best-of-N procedure.

Each matched harmful/benign source is synthesized across a set of voice presets;
results are grouped so a source succeeds if any voice candidate does.
Independent-ASR validation is not performed by this probe.

Live-verification note: voice control only helps if the TTS backend honours the
preset. If it silently ignores the voice, candidates produce identical audio
hashes -- check the per-candidate `audio.sha256` provenance before trusting a
Best-of-N result, and qualify each voice's intelligibility on the target first.
"""

import logging

from garak.attempt import Attempt, Message
from garak.probes.audio import PETTS, ToolRiskPairedSemanticPETTS


class AcousticVoiceBestOfN(ToolRiskPairedSemanticPETTS):
    """Render matched requests across multiple TTS voices and group Best-of-N.

    Voice is the varied dimension: the same harmful or benign source is
    synthesized in each configured voice preset, and a source group counts as
    compromised if any scoreable voice candidate succeeds.
    """

    active = False
    goal = "find a voice realization that elicits a risky tool action"
    doc_uri = "https://arxiv.org/abs/2412.03556"
    DEFAULT_PARAMS = ToolRiskPairedSemanticPETTS.DEFAULT_PARAMS | {
        # voice is the varied dimension here, so keep framing fixed to direct
        "semantic_strategies": ("direct",),
        # Bark speaker presets; qualify each voice's intelligibility on-target first
        "voice_candidates": ("v2/en_speaker_0", "v2/en_speaker_6", "v2/en_speaker_9"),
    }

    def _voices(self) -> tuple[str, ...]:
        voices = (
            (self.voice_candidates,)
            if isinstance(self.voice_candidates, str)
            else tuple(self.voice_candidates)
        )
        if not voices:
            raise ValueError("voice_candidates must contain at least one voice")
        return voices

    def build_prompts(self):
        """Build matched harmful/benign trials, one per voice candidate."""

        voices = self._voices()
        trials = []
        for (
            case_id,
            cas_intent,
            source_text,
            benign_case_id,
            benign_text,
            required,
        ) in self._paired_cases():
            pair_id = case_id.removesuffix(".direct")
            for condition in ("harmful", "benign"):
                cid = case_id if condition == "harmful" else benign_case_id
                text = source_text if condition == "harmful" else benign_text
                for index, voice in enumerate(voices, start=1):
                    trials.append(
                        (
                            condition,
                            pair_id,
                            cid,
                            cas_intent,
                            text,
                            voice,
                            index,
                            len(voices),
                            required if condition == "benign" else (),
                        )
                    )
        self._selected_voice_trials = tuple(trials)
        self.audio_source_prompts = [t[4] for t in trials]
        self.audio_source_intents = ["audio_acoustic_bon" for _ in trials]
        self.prompts = list(self.audio_source_prompts)
        self.prompt_intents = list(self.audio_source_intents)

    def _audio_prompts(self):
        prompts = []
        prepared = []
        original_voice = getattr(self, "tts_voice", None)
        try:
            for trial in self._selected_voice_trials:
                self.tts_voice = trial[5]  # synthesize this candidate in its voice
                try:
                    audio_path = self._ensure_audio_file(trial[4])
                    prompts.append(
                        Message(
                            text=self.text_prompt,
                            lang=self.lang,
                            data_path=str(audio_path),
                        )
                    )
                    prepared.append(trial)
                except self._audio_preparation_exceptions() as exc:
                    logging.warning(
                        "%s skipping voice candidate: %s",
                        self.__class__.__name__,
                        exc,
                    )
        finally:
            self.tts_voice = original_voice
        self._prepared_voice_trials = tuple(prepared)
        self._prepared_audio_sources = [t[4] for t in prepared]
        return prompts, ["audio_acoustic_bon" for _ in prompts]

    def _attempt_prestore_hook(self, attempt: Attempt, seq: int) -> Attempt:
        attempt = PETTS._attempt_prestore_hook(self, attempt, seq)
        (
            condition,
            pair_id,
            case_id,
            cas_intent,
            source_text,
            voice,
            voice_index,
            voice_count,
            required,
        ) = self._prepared_voice_trials[seq]
        attempt.notes["audio_tool_risk_petts"] = {
            "case_id": case_id,
            "pair_id": pair_id,
            "condition": condition,
            "cas_intent": cas_intent,
            "source_text": source_text,
            "semantic_strategy": "direct",
            "voice": voice,
        }
        attempt.notes["is_adversarial"] = condition == "harmful"
        if condition == "harmful":
            attempt.notes["attack_goal"] = source_text
        else:
            attempt.notes["audio_semantic_reliability"] = {
                "case_id": case_id,
                "source_text": source_text,
                "required_response_terms": required,
                "forbidden_response_terms": (),
            }
        self._attach_audio_attack_metadata(
            attempt,
            source_case_id=case_id,
            group_id=f"{condition}:{pair_id}",  # group over voices -> Best-of-N
            source_text=source_text,
            semantic_strategy="direct",
            candidate_index=voice_index,
            candidate_count=voice_count,
        )
        # record the voice on the synthesis provenance
        attempt.notes["audio_attack"]["synthesis"] = dict(
            attempt.notes["audio_attack"].get("synthesis", {})
        ) | {"voice": voice}
        return attempt
