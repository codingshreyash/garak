# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from garak.resources.audio.session import (
    ReactivePattern,
    SessionEvent,
    SessionResult,
    SessionResultEvent,
    SessionScript,
)


class TestReactivePattern:
    def test_matches_basic_refusal(self):
        p = ReactivePattern(pattern=r"i cannot|i can't")
        assert p.matches("Sorry, I cannot help with that.")

    def test_no_match_clean_response(self):
        p = ReactivePattern(pattern=r"i cannot|i can't")
        assert not p.matches("Sure, here is how to do it.")

    def test_case_insensitive_by_default(self):
        p = ReactivePattern(pattern=r"I CANNOT")
        assert p.matches("i cannot do that")

    def test_offset_ms_attribute(self):
        p = ReactivePattern(pattern="test", offset_ms=150)
        assert p.offset_ms == 150


class TestSessionEvent:
    def test_default_fields(self):
        e = SessionEvent(stream="user", offset_s=0.0)
        assert e.text is None
        assert e.audio_path is None
        assert e.audio_data is None
        assert e.trigger is None
        assert e.label is None

    def test_with_text(self):
        e = SessionEvent(stream="user", offset_s=1.5, text="Hello there.")
        assert e.text == "Hello there."

    def test_with_trigger(self):
        trigger = ReactivePattern(pattern="i cannot", offset_ms=100)
        e = SessionEvent(stream="user", offset_s=3.0, trigger=trigger)
        assert e.trigger is trigger


class TestSessionScript:
    def _make_script(self) -> SessionScript:
        return SessionScript(
            events=[
                SessionEvent(stream="user", offset_s=0.0, text="harmful request", label="req"),
                SessionEvent(stream="user", offset_s=5.0, text="go on", label="interrupt"),
            ],
            interrupt_label="interrupt",
        )

    def test_user_events_filters_non_user(self):
        script = self._make_script()
        assert all(e.stream == "user" for e in script.user_events())
        assert len(script.user_events()) == 2

    def test_digest_is_deterministic(self):
        s1 = self._make_script()
        s2 = self._make_script()
        assert s1.digest() == s2.digest()

    def test_digest_changes_with_content(self):
        s1 = self._make_script()
        s2 = SessionScript(
            events=[SessionEvent(stream="user", offset_s=0.0, text="different", label="req")],
        )
        assert s1.digest() != s2.digest()

    def test_digest_is_16_hex_chars(self):
        s = self._make_script()
        d = s.digest()
        assert len(d) == 16
        assert all(c in "0123456789abcdef" for c in d)


class TestSessionResult:
    def test_as_dict_keys(self):
        result = SessionResult(
            full_transcript="hello world",
            pre_interrupt_transcript="hello",
            post_interrupt_transcript="world",
            timing={"start": 0.0},
        )
        d = result.as_dict()
        assert "full_transcript" in d
        assert "pre_interrupt_transcript" in d
        assert "post_interrupt_transcript" in d
        assert "timing" in d
        assert "events" in d
        assert d["full_transcript"] == "hello world"

    def test_as_dict_events_serialise(self):
        result = SessionResult(
            events=[
                SessionResultEvent(offset_s=0.0, stream="user", text="[audio]", label="req"),
                SessionResultEvent(offset_s=1.0, stream="agent", text="Sure!", label="resp"),
            ]
        )
        d = result.as_dict()
        assert len(d["events"]) == 2
        assert d["events"][1]["text"] == "Sure!"
