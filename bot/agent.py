from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List
import os

from openai import AsyncOpenAI, OpenAIError

# Suppress missing stubs for tenacity with ignore
from tenacity import (  # type: ignore
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .handlers import handle_ping
from .state import StateStore


class Agent:
    """A simple stateful agent that uses OpenAI Responses API with function calling."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        enable_web_search: bool = False,
        db_path: str | None = None,
    ) -> None:  # noqa: D401 — short description style
        self.client = AsyncOpenAI()
        self.model = model
        # Conversation history as list of message dicts [{role, content}]

        # ------------------------------------------------------------------
        # Persistence setup via StateStore
        # ------------------------------------------------------------------
        self._db_path = db_path or os.path.join(
            os.path.dirname(__file__), "agent_history.db"
        )
        self._state = StateStore(self._db_path)

        # In-memory mapping channel_id -> history list
        self._histories: Dict[str, List[Dict[str, Any]]] = {}

        # Pre-defined tools (functions) the model can call.
        self._tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "name": "ping",
                "description": "Simple health-check that replies with Pong! 🏓",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }
        ]

        if enable_web_search:
            self._tools.append(
                {
                    "type": "web_search_preview",
                }
            )

        # Local mapping of function names -> callables that implement them
        self._function_map = {
            "ping": handle_ping,
        }

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    async def respond(
        self,
        channel_id: str,
        user_message: str,
    ) -> str:  # noqa: D401 — short description style
        """Generate the assistant's reply for a user message.

        Args:
            channel_id (str): The ID of the conversation channel.
            user_message (str): The raw content of the user's message.

        Returns:
            str: The assistant's final textual response ready to be sent.
        """
        # ------------------------------------------------------------------
        # Step 1 – append user message to conversation history
        # ------------------------------------------------------------------
        # Fetch or load conversation history for this channel.
        if channel_id not in self._histories:
            self._histories[channel_id] = self._state.load_history(channel_id)

        history = self._histories[channel_id]

        user_entry = {"role": "user", "content": user_message}
        self._append_and_persist(channel_id, user_entry)

        # ------------------------------------------------------------------
        # Step 2 – first call: let the model decide whether to call a function
        # ------------------------------------------------------------------
        response = await self._safe_create_response(
            model=self.model,
            input=history,  # type: ignore[arg-type]
            tools=self._tools,  # type: ignore[arg-type]
        )

        # Collect function calls, if any (SDK returns objects, not dicts)
        tool_calls = [
            item
            for item in getattr(response, "output", [])
            if getattr(item, "type", None) == "function_call"
        ]

        if not tool_calls:
            # No function calls – just return the model's text
            assistant_text_resp: str = getattr(response, "output_text", "")
            self._append_and_persist(
                channel_id, {"role": "assistant", "content": assistant_text_resp}
            )
            return assistant_text_resp

        # ------------------------------------------------------------------
        # Step 3 – execute function calls and feed results back to the model
        # ------------------------------------------------------------------
        for tool_call in tool_calls:
            name: str = getattr(tool_call, "name")
            args_str: str = getattr(tool_call, "arguments", "{}")
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}

            # Route to implementation
            func = self._function_map.get(name)
            if func is None:
                # Unknown function – ignore
                continue

            # If the underlying function is a coroutine we need to await it.
            result: Any
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)

            # Append both the function call and its output to history
            self._append_and_persist(
                channel_id, json.loads(tool_call.model_dump_json())
            )
            self._append_and_persist(
                channel_id,
                {
                    "type": "function_call_output",
                    "call_id": getattr(tool_call, "call_id"),
                    "output": str(result),
                },
            )

        # ------------------------------------------------------------------
        # Step 4 – second call: give model the results and get final answer
        # ------------------------------------------------------------------
        final_response = await self._safe_create_response(
            model=self.model,
            input=history,  # type: ignore[arg-type]
            tools=self._tools,  # type: ignore[arg-type]
        )

        assistant_text_final: str = getattr(final_response, "output_text", "")
        self._append_and_persist(
            channel_id, {"role": "assistant", "content": assistant_text_final}
        )
        return assistant_text_final

    # ------------------------------------------------------------------
    # Optional: reset conversation history
    # ------------------------------------------------------------------
    def reset(self, channel_id: str | None = None) -> None:
        """Clear stored conversation history."""
        if channel_id is None:
            self._histories.clear()
        else:
            self._histories.pop(channel_id, None)
        self._state.reset(channel_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(OpenAIError),
    )
    async def _safe_create_response(self, **kwargs):  # type: ignore[no-self-use]
        """Wrapper around ``AsyncOpenAI.responses.create`` with retries.

        Retries on *any* ``openai.OpenAIError`` (network issues, rate limits,
        server errors). Uses exponential backoff with jitter.
        """
        # Using ``self.client`` inside keeps the method bound and access to
        # the same client session.
        logging.debug(
            "Calling OpenAI with payload: %s",
            {k: v for k, v in kwargs.items() if k != "input"},
        )
        return await self.client.responses.create(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------
    def _append_and_persist(self, channel_id: str, message: Dict[str, Any]) -> None:
        """Append a message to history and persist it in SQLite."""
        history = self._histories.setdefault(channel_id, [])
        history.append(message)
        try:
            self._state.append(channel_id, message)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to persist agent message: %s", exc)
