from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any

DEFAULT_SETTINGS: Dict[str, Any] = {
    "model": "gpt-5-mini",
    "instructions": "You are Botty McBotface, a bot powered by OpenAI's API. You are a friendly, helpful bot that is always willing to chat and help out. You are not perfect, but you are trying your best.",  # default personality
    "reasoning_level": "minimal",
    "enable_web_search": False,
    "maximum_turns": 10,
    "maximum_user_messages": 25,
    "auto_respond_channels": [],
    "dm_whitelist": [],
}


def load_settings() -> Dict[str, Any]:
    """Load settings from a *settings.json* file at project root.

    If the file is missing or malformed, safe defaults are returned.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root_dir, "settings.json")
    if not os.path.exists(path):
        return DEFAULT_SETTINGS.copy()

    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Failed to read settings.json: %s", exc)
        return DEFAULT_SETTINGS.copy()

    # Merge with defaults so unspecified keys keep default values.
    merged = {**DEFAULT_SETTINGS, **data}
    return merged
