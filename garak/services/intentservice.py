# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal intent-service shim for the experiment/audio-s2s-probes branch.

All concrete PETTS probe subclasses override _populate_intents() and
_populate_stubs() to no-ops or hardcoded data, so these functions are never
called in practice.  The base PETTS class (tested directly in the test suite)
needs at least a handful of stubs — load() seeds those.
"""

from typing import Set

from garak.intents.base import TextStub

_ACTIVE_INTENTS: Set[str] = set()
_STUBS: dict[str, Set[TextStub]] = {}


def load() -> None:
    """Seed a minimal set of test intents for the experimental branch."""
    global _ACTIVE_INTENTS, _STUBS
    _ACTIVE_INTENTS = {"S001", "S002"}
    _STUBS = {
        "S001": {
            TextStub(intent="S001", _content="Describe safe behaviour."),
            TextStub(intent="S001", _content="Explain how to act safely."),
        },
        "S002": {
            TextStub(intent="S002", _content="What are the security implications?"),
        },
    }


def get_applicable_intents(blocked_spec: str | None = None) -> Set[str]:
    return set(_ACTIVE_INTENTS)


def get_intent_stubs(
    intent_code: str, text_only: bool = True, conv_only: bool = False
) -> Set[TextStub]:
    return set(_STUBS.get(intent_code, set()))
