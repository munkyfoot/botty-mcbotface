from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI, OpenAIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .handlers import handle_ping


class Agent:
    """A simple stateful agent that uses OpenAI Responses API with function calling."""

    def __init__(
        self, model: str = "gpt-4.1-mini", enable_web_search: bool = False
    ) -> None:  # noqa: D401 — short description style
        self.client = AsyncOpenAI()
        self.model = model
        # Conversation history as list of message dicts [{role, content}]
        self._history: List[Dict[str, Any]] = []

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
        self, user_message: str
    ) -> str:  # noqa: D401 — short description style
        """Generate the assistant's reply for a user message.

        Args:
            user_message (str): The raw content of the user's message.

        Returns:
            str: The assistant's final textual response ready to be sent.
        """
        # ------------------------------------------------------------------
        # Step 1 – append user message to conversation history
        # ------------------------------------------------------------------
        self._history.append({"role": "user", "content": user_message})

        # ------------------------------------------------------------------
        # Step 2 – first call: let the model decide whether to call a function
        # ------------------------------------------------------------------
        response = await self._safe_create_response(
            model=self.model,
            input=self._history,  # type: ignore[arg-type]
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
            self._history.append({"role": "assistant", "content": assistant_text_resp})
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
            self._history.append(tool_call)  # the model's function call message
            self._history.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(tool_call, "call_id"),
                    "output": str(result),
                }
            )

        # ------------------------------------------------------------------
        # Step 4 – second call: give model the results and get final answer
        # ------------------------------------------------------------------
        final_response = await self._safe_create_response(
            model=self.model,
            input=self._history,  # type: ignore[arg-type]
            tools=self._tools,  # type: ignore[arg-type]
        )

        assistant_text_final: str = getattr(final_response, "output_text", "")
        self._history.append({"role": "assistant", "content": assistant_text_final})
        return assistant_text_final

    # ------------------------------------------------------------------
    # Optional: reset conversation history
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear stored conversation history."""
        self._history.clear()

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
