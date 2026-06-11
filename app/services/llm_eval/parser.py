"""

Extracts and validates the structured JSON evaluation block from raw
Qwen output.

Qwen may wrap its response in markdown code fences, add trailing
explanation text, or return floats/strings instead of integers.
This module handles all of those cases robustly.

Returns None on any unrecoverable parse failure — the task worker
will use this signal to trigger a retry.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Required keys in the LLM JSON output
_REQUIRED_SCORE_KEYS = {
    "correctness",
    "logic",
    "optimization",
    "edge_case_handling",
    "readability",
}

_REQUIRED_LIST_KEYS = {
    "strengths",
    "weaknesses",
    "recommendations",
}

_ALL_REQUIRED_KEYS = _REQUIRED_SCORE_KEYS | _REQUIRED_LIST_KEYS


def parse(raw_response: str) -> dict | None:
    """
    Extract and validate the JSON evaluation object from a raw LLM response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Trailing/leading explanation text
    - Float scores (8.5 -> 8)
    - String scores ("90" -> 90)
    - Out-of-range scores (clamped to 0-100 with a warning)

    Args:
        raw_response: The raw string output from Ollama.

    Returns:
        A validated dict with all required keys, or None if parsing fails.
    """
    if not raw_response or not raw_response.strip():
        logger.warning("Parser received empty raw response.")
        return None

    # Step 1: Extract JSON block — try markdown fences first, then bare braces
    json_str = _extract_json_block(raw_response)
    if json_str is None:
        logger.warning("Parser could not find a JSON block in the response.")
        return None

    # Step 2: Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Parser JSON decode error: %s | snippet: %.200s", exc, json_str)
        return None

    if not isinstance(data, dict):
        logger.warning("Parser expected a JSON object, got: %s", type(data).__name__)
        return None

    # Step 3: Check all required keys are present
    missing = _ALL_REQUIRED_KEYS - data.keys()
    if missing:
        logger.warning("Parser: missing required keys: %s", missing)
        return None

    # Step 4: Coerce and validate score fields
    for key in _REQUIRED_SCORE_KEYS:
        raw_val = data[key]
        try:
            coerced = int(float(raw_val))   # handles "90", 8.5, 9
        except (TypeError, ValueError):
            logger.warning("Parser: score '%s' has invalid value: %r", key, raw_val)
            return None

        if not (0 <= coerced <= 100):
            logger.warning(
                "Parser: score '%s' out of range: %d — clamping to [0, 100]", key, coerced
            )
            coerced = max(0, min(100, coerced))

        data[key] = coerced

    # Step 5: Validate list fields
    for key in _REQUIRED_LIST_KEYS:
        val = data[key]
        if not isinstance(val, list):
            # Tolerate a single string — wrap it
            if isinstance(val, str):
                data[key] = [val]
            else:
                logger.warning("Parser: '%s' should be a list, got: %s", key, type(val).__name__)
                return None
        # Ensure all items are strings
        data[key] = [str(item) for item in data[key]]

    return data


# Internal helpers

def _extract_json_block(text: str) -> str | None:
    """
    Try to extract a JSON object string from raw text.

    Priority:
    1. Content inside ```json ... ``` or ``` ... ``` fences
    2. First { ... } block found in the text
    """
    # Try markdown fence first
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find the outermost { ... } block
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None
