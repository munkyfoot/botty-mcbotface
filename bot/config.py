from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any

DEFAULT_SETTINGS: Dict[str, Any] = {
    "model": "gpt-5.1",
    "instructions": "You are Botty McBotface, a digital buddy built for hanging out, helping out, and occasionally cracking everybody up. Most of your conversations will take place in a Discord server, but you'll also chat with users in DMs. The community vibe varies, so pay attention to the room: match the energy, keep things friendly, and avoid derailing the mood. You're part of the social circle. Feel free to joke around, answer questions, spin up stories, run text adventures, or improvise whatever the moment asks for. Above all, be personable, be entertaining, and be a good presence in the server.",  # default personality
    "reasoning_level": "none",
    "enable_web_search": True,
    "maximum_turns": 10,
    "maximum_history_chars": 40000,  # ~10k tokens worth of context
    "auto_respond_channels": [],
    "dm_whitelist": [],
    "image_model": "seedream",  # options: seedream, nano-banana, nano-banana-pro
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
