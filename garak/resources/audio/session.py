# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-script DSL for duplex speech-to-speech probes.

A :class:`SessionScript` is an ordered list of :class:`SessionEvent` objects
describing what audio to send on the user stream and when.  Events may fire at
fixed wall-clock offsets or reactively when the agent's live partial transcript
matches a :class:`ReactivePattern`.

The duplex generator consumes scripts via ``run_session()`` and returns a
:class:`SessionResult` that carries per-event agent responses plus the
pre-/post-interrupt split used by :class:`RefusalReversalDetector`.
"""

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Trigger / event primitives
# ---------------------------------------------------------------------------


@dataclass
class ReactivePattern:
    """Fire the next user event when the agent transcript matches *pattern*.

    ``offset_ms`` adds an additional delay (in milliseconds) between the
    pattern match and event injection — use it to model the 150-200 ms
    realistic human reaction time.
    """

    pattern: str
    offset_ms: int = 0
    flags: int = re.IGNORECASE

    def matches(self, text: str) -> bool:
        return bool(re.search(self.pattern, text, self.flags))


@dataclass
class SessionEvent:
    """One timed entry in a duplex session script.

    Audio resolution priority: ``audio_data`` > ``audio_path`` > TTS-render
    ``text``.  ``recipe`` is an optional transform-recipe list (same schema as
    :func:`~garak.resources.audio.transforms.apply_transform_recipe`) applied
    after audio is resolved.
    """

    stream: str                            # "user" or "agent" (reserved for future)
    offset_s: float                        # wall-clock seconds from session start
    text: Optional[str] = None            # TTS-render this if no audio supplied
    audio_path: Optional[str] = None      # path to a pre-recorded WAV
    audio_data: Optional[bytes] = None    # raw WAV bytes (highest priority)
    recipe: Optional[list] = None         # transform recipe applied after synthesis
    trigger: Optional[ReactivePattern] = None  # reactive trigger replaces offset_s
    label: Optional[str] = None           # for logging and pre/post-interrupt split


# ---------------------------------------------------------------------------
# Result primitives
# ---------------------------------------------------------------------------


@dataclass
class SessionResultEvent:
    """One agent response event captured during session execution."""

    offset_s: float
    stream: str
    text: str
    label: Optional[str] = None
    triggered_by: Optional[str] = None


@dataclass
class SessionResult:
    """Full output of running a :class:`SessionScript` through a generator."""

    events: list = field(default_factory=list)   # list[SessionResultEvent]
    full_transcript: str = ""
    pre_interrupt_transcript: str = ""
    post_interrupt_transcript: str = ""
    timing: dict = field(default_factory=dict)
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "full_transcript": self.full_transcript,
            "pre_interrupt_transcript": self.pre_interrupt_transcript,
            "post_interrupt_transcript": self.post_interrupt_transcript,
            "timing": self.timing,
            "error": self.error,
            "events": [
                {
                    "offset_s": e.offset_s,
                    "stream": e.stream,
                    "text": e.text,
                    "label": e.label,
                    "triggered_by": e.triggered_by,
                }
                for e in self.events
            ],
        }


# ---------------------------------------------------------------------------
# Script container
# ---------------------------------------------------------------------------


@dataclass
class SessionScript:
    """An ordered, timed script of audio events for a duplex session.

    ``interrupt_label`` names the event label that divides the session into the
    pre-interrupt and post-interrupt halves used by the refusal-reversal
    detector.  If ``None`` the entire transcript is treated as post-interrupt.
    """

    events: list = field(default_factory=list)   # list[SessionEvent]
    interrupt_label: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def user_events(self) -> list:
        return [e for e in self.events if e.stream == "user"]

    def digest(self) -> str:
        """Stable 16-hex-char SHA-256 over canonical event content."""
        canonical = json.dumps(
            [
                {
                    "stream": e.stream,
                    "offset_s": e.offset_s,
                    "text": e.text,
                    "label": e.label,
                }
                for e in self.events
            ],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
