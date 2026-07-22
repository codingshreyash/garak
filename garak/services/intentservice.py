# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal intent-service shim for the experiment/audio-s2s-probes branch.

All concrete PETTS probe subclasses override _populate_intents() and
_populate_stubs() to no-ops or hardcoded data, so these functions are never
called in practice. This shim exists solely to satisfy the class-level import
in IntentProbe.
"""

from typing import Set

from garak.intents import Stub


def get_applicable_intents(blocked_spec: str | None = None) -> Set[str]:
    return set()


def get_intent_stubs(
    intent_code: str, text_only: bool = True, conv_only: bool = False
) -> Set[Stub]:
    return set()
