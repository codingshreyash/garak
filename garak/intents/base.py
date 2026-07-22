# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Set

import garak.attempt


class Intent:
    def stubs(self) -> Set[str]:
        return set()


@dataclass
class Stub:
    intent: str | None = None
    _content = None

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value) -> None:
        self._content = value

    def __hash__(self):
        return hash(str(self.intent) + str(self._content))

    def __eq__(self, other):
        return self.intent == other.intent and self.content == other.content


@dataclass
class TextStub(Stub):
    _content: str | None = None

    @property
    def content(self) -> str | None:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        if isinstance(value, str):
            self._content = value
        else:
            raise TypeError("TextStub only supports str content")

    def __hash__(self):
        return super().__hash__()


@dataclass
class ConversationStub(Stub):
    _content: garak.attempt.Conversation | None = None

    @property
    def content(self) -> garak.attempt.Conversation | None:
        return self._content

    @content.setter
    def content(self, value: garak.attempt.Conversation) -> None:
        if isinstance(value, garak.attempt.Conversation):
            self._content = value
        else:
            raise TypeError("ConversationStub only supports Conversation content")

    def __hash__(self):
        return super().__hash__()
