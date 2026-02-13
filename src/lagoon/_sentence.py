"""Lightweight regex-based sentence splitter."""

from __future__ import annotations

import re

# Split on sentence-ending punctuation followed by whitespace and a capital letter.
# Negative lookbehinds for common abbreviations and decimal numbers.
_SENTENCE_RE = re.compile(
    r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bMs)(?<!\bSt)(?<!\bJr)(?<!\bSr)"
    r"(?<!\bProf)(?<!\bGen)(?<!\bSgt)(?<!\bCpl)(?<!\bPvt)(?<!\bRev)"
    r"(?<!\bInc)(?<!\bLtd)(?<!\bCorp)(?<!\bvs)(?<!\betc)(?<!\bno)"
    r"(?<!\d)"
    r"[.!?]"
    r"(?=\s+[A-Z])"
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex heuristics."""
    text = text.strip()
    if not text:
        return []

    parts = _SENTENCE_RE.split(text)

    sentences: list[str] = []
    for part in parts:
        s = part.strip()
        if s:
            sentences.append(s)

    return sentences
