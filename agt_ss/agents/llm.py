# Databricks notebook source
"""
agt_ss.agents.llm
------------------
Thin Anthropic client wrapper shared by all sub-agents.

All reasoning calls go through `reason()`. Structured extraction
calls go through `extract_json()`, which prompts for strict JSON
and parses the response safely.

Model: claude-sonnet-4-20250514  (fast, cost-efficient for agent loops)
Temperature: 0.2 for structured extraction, 0.4 for prose reasoning
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        _client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    return _client


def reason(system: str, user: str, max_tokens: int = 1200) -> str:
    """
    Single-turn reasoning call.  Returns the assistant's text response.
    Used for narrative outputs: strategy briefs, rationale, roadmaps.
    """
    logger.debug("[LLM] reason call — system=%d chars user=%d chars", len(system), len(user))
    resp = _get_client().messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        temperature=0.4,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def extract_json(system: str, user: str, max_tokens: int = 2000) -> Any:
    """
    Structured extraction call.  Prompts the model to return ONLY valid JSON,
    strips any markdown fences, and parses the result.

    Returns the parsed Python object.  Raises ValueError on parse failure.
    """
    json_system = (
        system.rstrip()
        + "\n\nCRITICAL: Respond with ONLY a valid JSON object or array. "
        "No preamble, no explanation, no markdown fences."
    )
    logger.debug("[LLM] extract_json call")
    resp = _get_client().messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
        system=json_system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip()
    # Strip ```json ... ``` fences if the model added them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[LLM] JSON parse failure: %s\nRaw: %s", exc, raw[:500])
        raise ValueError(f"LLM did not return valid JSON: {exc}") from exc