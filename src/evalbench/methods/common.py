"""Prompt fragments shared by the methods."""
from __future__ import annotations

SYSTEM_MEMORY = ("You are being tested on what you remember from your training data. "
                 "Answer from memory only; do not use tools or browse.")


def about(company: str | None) -> str:
    return f" about {company}" if company else ""


def kind(source_group: str) -> str:
    return "press release" if source_group == "press_release" else "news article"
