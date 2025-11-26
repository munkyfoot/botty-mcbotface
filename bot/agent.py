from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Tuple, AsyncGenerator, TYPE_CHECKING
import os
import base64
import io

from openai import AsyncOpenAI, OpenAIError

# Suppress missing stubs for tenacity with ignore
from tenacity import (  # type: ignore
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .handlers import (
    handle_ping,
    handle_roll,
    handle_generate_image,
    handle_generate_meme,
    handle_edit_image,
)
from .state import StateStore
from .utils import prepare_image
from .image_models import get_model_keys, get_models_description_for_tools, get_active_model_key

if TYPE_CHECKING:
    from .storage import StorageProvider

_DEFAULT_REASONING_LEVEL = "medium"
_REASONING_LEVEL_CANONICAL: dict[str, str] = {
    "none": "none",
    "off": "none",
    "disabled": "none",
    "minimal": "minimal",
    "min": "minimal",
    "low": "low",
    "light": "low",
    "quick": "low",
    "fast": "low",
    "medium": "medium",
    "balanced": "medium",
    "standard": "medium",
    "default": "medium",
    "normal": "medium",
    "high": "high",
    "deep": "high",
    "intense": "high",
    "intensive": "high",
    "max": "high",
}
_REASONING_EFFORT: dict[str, str | None] = {
    "none": "none",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class Agent:
    """A simple stateful agent that uses OpenAI Responses API with function calling."""

    def __init__(
        self,
        model: str,
        instructions: str,
        enable_web_search: bool,
        maximum_turns: int,
        maximum_history_chars: int | None = None,
        reasoning_level: str | None = None,
        storage: "StorageProvider | None" = None,
    ) -> None:

        self._model = model
        self._instructions = instructions
        self._enable_web_search = enable_web_search
        self._maximum_turns = maximum_turns
        self._maximum_history_chars = maximum_history_chars
        self._storage = storage
        self._client = AsyncOpenAI()
        self._reasoning_warning_logged = False
        self._reasoning_level = self._normalize_reasoning_level(reasoning_level)
        self._maybe_log_reasoning_ignored()
        # Conversation history as list of message dicts [{role, content}]

        # ------------------------------------------------------------------
        # Persistence setup via StateStore
        # ------------------------------------------------------------------
        self._state = StateStore(maximum_history_chars=self._maximum_history_chars)

        # In-memory mapping channel_id -> history list
        self._histories: Dict[str, List[Dict[str, Any]]] = {}

        # In-memory mapping channel_id -> responding state
        self._responding: Dict[str, bool] = {}
        self._queued: Dict[str, bool] = {}

        # Per-channel locks used to make flag updates atomic and avoid races
        self._locks: Dict[str, asyncio.Lock] = {}

        # Tools are built dynamically per-request to include channel_id
        # See _build_tools() method

    # ---------------------------------------------------------------------
    # Tool definition builders
    # ---------------------------------------------------------------------
    def _build_tools(self, channel_id: str) -> List[Dict[str, Any]]:
        """Build tool definitions with channel context.

        Output-producing tools include a channel_id parameter that defaults
        to the current channel, allowing cross-channel operations.

        Args:
            channel_id: The current channel ID (used as default for channel_id params).

        Returns:
            List of tool definitions for the OpenAI API.
        """
        # Common channel_id property for output-producing tools
        channel_id_prop = {
            "type": "string",
            "description": f"Target channel ID. Use the current channel ({channel_id}) unless sending to a different channel. Check server context for available channels.",
            "default": channel_id,
        }

        tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "name": "quick_message",
                "description": "Send a quick message and prepare to take an action next turn. Use this before and between tool calls to keep users informed. Can send to the current channel or a different channel in the same server.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send.",
                        },
                        "channel_id": channel_id_prop,
                    },
                    "required": ["message", "channel_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "create_poll",
                "description": "Create a poll. Can be sent to the current channel or a different channel.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask in the poll. Max 300 characters.",
                        },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": "The options for the poll. Max 10 options, each max 55 characters. Prefer 2-5 options for best engagement.",
                        },
                        "duration": {
                            "type": "integer",
                            "description": "Duration of the poll in hours.",
                            "default": 24,
                        },
                        "multiple": {
                            "type": "boolean",
                            "description": "Whether to allow multiple selections.",
                            "default": False,
                        },
                        "channel_id": channel_id_prop,
                    },
                    "required": ["question", "options", "duration", "multiple", "channel_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "ping",
                "description": "Simple health-check that replies with Pong! 🏓",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": channel_id_prop,
                    },
                    "required": ["channel_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "roll_dice",
                "description": "Roll dice with parameters (value, count, modifier, drops).",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dice_value": {
                            "type": "integer",
                            "description": "Number of sides on the die (e.g., 6 for d6, 20 for d20)",
                        },
                        "dice_count": {
                            "type": "integer",
                            "description": "How many dice to roll",
                            "default": 1,
                        },
                        "dice_modifier": {
                            "type": "integer",
                            "description": "Flat modifier to add after roll",
                            "default": 0,
                        },
                        "drop_n_lowest": {
                            "type": "integer",
                            "description": "Number of lowest dice to drop",
                            "default": 0,
                        },
                        "drop_n_highest": {
                            "type": "integer",
                            "description": "Number of highest dice to drop",
                            "default": 0,
                        },
                        "channel_id": channel_id_prop,
                    },
                    "required": [
                        "dice_value",
                        "dice_count",
                        "dice_modifier",
                        "drop_n_lowest",
                        "drop_n_highest",
                        "channel_id",
                    ],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "generate_image",
                "description": f"Generate an image based on a prompt. Returns a URL of the generated image.\n\nAvailable models:\n{get_models_description_for_tools()}",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt describing the requested image content",
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Desired aspect ratio for the image (e.g., 1:1, 16:9)",
                            "enum": [
                                "1:1",
                                "4:3",
                                "3:4",
                                "16:9",
                                "9:16",
                                "3:2",
                                "2:3",
                                "21:9",
                            ],
                        },
                        "model": {
                            "type": "string",
                            "description": "The image model to use. Each has different strengths.",
                            "enum": get_model_keys(),
                            "default": get_active_model_key(),
                        },
                        "channel_id": channel_id_prop,
                    },
                    "required": [
                        "prompt",
                        "aspect_ratio",
                        "model",
                        "channel_id",
                    ],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "generate_meme",
                "description": f"Generate a meme based on a prompt.\n\nAvailable models:\n{get_models_description_for_tools()}",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_prompt": {
                            "type": "string",
                            "description": "The prompt to generate an image from.",
                        },
                        "text": {
                            "type": "string",
                            "description": "The text to add to the image.",
                        },
                        "model": {
                            "type": "string",
                            "description": "The image model to use. Each has different strengths.",
                            "enum": get_model_keys(),
                            "default": get_active_model_key(),
                        },
                        "channel_id": channel_id_prop,
                    },
                    "required": ["image_prompt", "text", "model", "channel_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "edit_image",
                "description": f"Edit or combine images based on a prompt. Can accept one or multiple image URLs to blend, edit, or transform together.\n\nAvailable models:\n{get_models_description_for_tools()}",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt describing the edits or how to combine the images.",
                        },
                        "images": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of image URLs to edit or combine. IMPORTANT: Must be URLs, not base64 strings. Use urls from images in the conversation history.",
                        },
                        "model": {
                            "type": "string",
                            "description": "The image model to use. Models support different max input images (see descriptions).",
                            "enum": get_model_keys(),
                            "default": get_active_model_key(),
                        },
                        "channel_id": channel_id_prop,
                    },
                    "required": ["prompt", "images", "model", "channel_id"],
                    "additionalProperties": False,
                },
            },
            # Memory tools don't have channel_id - they always operate on current channel
            {
                "type": "function",
                "name": "save_memory",
                "description": "Save something interesting or meaningful to your long-term memory. Feel free to remember things you find funny, moments that stand out, running jokes, user preferences, interesting facts about the people you chat with, or anything that helps you understand and relate to this channel better. Think of this as your personal diary - save what feels worth remembering to build your own personality and connection with the group.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "What you want to remember. Write it naturally - could be a joke, a fact, an observation, a preference, or anything memorable.",
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "list_memories",
                "description": "List all long-term memories for the current channel. Use this to review what you've remembered.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "update_memory",
                "description": "Update an existing long-term memory with new content. Use this to correct or expand upon a previously saved memory.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to update (can be the short 8-character prefix shown in list_memories).",
                        },
                        "content": {
                            "type": "string",
                            "description": "The new content for the memory.",
                        },
                    },
                    "required": ["memory_id", "content"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "delete_memory",
                "description": "Delete a long-term memory that is no longer relevant or accurate.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to delete (can be the short 8-character prefix shown in list_memories).",
                        },
                    },
                    "required": ["memory_id"],
                    "additionalProperties": False,
                },
            },
        ]

        if self._enable_web_search:
            tools.append(
                {
                    "type": "web_search_preview",
                }
            )

        return tools

    # ---------------------------------------------------------------------
    # Function handler builders
    # ---------------------------------------------------------------------
    def _build_function_handlers(self, source_channel_id: str) -> Dict[str, Any]:
        """Build function handlers with channel context.

        Returns a dict mapping function names to handler functions.
        Each handler returns a dict with:
        - type: "text", "poll", "image", "memory", or None (no output to user)
        - content: The data to send (string, dict, bytes, etc.)
        - channel_id: Target channel for the output

        Args:
            source_channel_id: The channel where the user message originated.
        """
        def _result(output_type: str, content: Any, channel_id: str = source_channel_id) -> Dict[str, Any]:
            """Helper to create consistent result dicts."""
            return {"type": output_type, "content": content, "channel_id": channel_id}

        return {
            "quick_message": lambda message, channel_id=source_channel_id: _result("text", message, channel_id),
            "create_poll": lambda question, options, duration=24, multiple=False, channel_id=source_channel_id: _result(
                "poll",
                {
                    "question": question,
                    "options": options,
                    "duration": duration,
                    "multiple": multiple,
                },
                channel_id,
            ),
            "ping": lambda channel_id=source_channel_id: _result("text", handle_ping(), channel_id),
            "roll_dice": lambda dice_value, dice_count=1, dice_modifier=0, drop_n_lowest=0, drop_n_highest=0, channel_id=source_channel_id: _result(
                "text",
                handle_roll(
                    dice_value=dice_value,
                    dice_count=dice_count,
                    dice_modifier=dice_modifier,
                    drop_n_lowest=drop_n_lowest,
                    drop_n_highest=drop_n_highest,
                ),
                channel_id,
            ),
            "generate_image": lambda prompt, aspect_ratio, model, channel_id=source_channel_id: _result(
                "image",
                handle_generate_image(prompt=prompt, aspect_ratio=aspect_ratio, model_key=model),
                channel_id,
            ),
            "generate_meme": lambda image_prompt, text, model, channel_id=source_channel_id: _result(
                "image",
                handle_generate_meme(image_prompt=image_prompt, text=text, model_key=model),
                channel_id,
            ),
            "edit_image": lambda prompt, images, model, channel_id=source_channel_id: _result(
                "image",
                handle_edit_image(prompt=prompt, images=images, model_key=model),
                channel_id,
            ),
            # Memory tools always use source channel - no cross-channel memory ops
            "save_memory": lambda content: _result(
                "memory:save",
                self._save_memory(source_channel_id, content),
                source_channel_id,
            ),
            "list_memories": lambda: _result("memory:list", self._list_memories(source_channel_id), source_channel_id),
            "update_memory": lambda memory_id, content: _result(
                "memory:update",
                self._update_memory(source_channel_id, memory_id, content),
                source_channel_id,
            ),
            "delete_memory": lambda memory_id: _result(
                "memory:delete",
                self._delete_memory(source_channel_id, memory_id),
                source_channel_id,
            ),
        }

    # ---------------------------------------------------------------------
    # Memory management helpers
    # ---------------------------------------------------------------------
    def _save_memory(self, channel_id: str, content: str) -> str:
        """Save a new long-term memory."""
        memory = self._state.add_memory(channel_id, content)
        return f"Memory saved with ID: {memory['id'][:8]}"

    def _list_memories(self, channel_id: str) -> str:
        """List all memories for a channel."""
        memories = self._state.load_memories(channel_id)
        if not memories:
            return "No memories saved for this channel."

        lines = ["Long-term memories:"]
        for memory in memories:
            lines.append(f"- [{memory['id'][:8]}] {memory['content']}")
        return "\n".join(lines)

    def _update_memory(self, channel_id: str, memory_id: str, content: str) -> str:
        """Update an existing memory."""
        # Try to find memory by prefix match if short ID provided
        memories = self._state.load_memories(channel_id)
        full_id = None
        for memory in memories:
            if memory["id"].startswith(memory_id):
                full_id = memory["id"]
                break

        if not full_id:
            return f"Memory with ID '{memory_id}' not found."

        result = self._state.update_memory(full_id, content)
        if result:
            return f"Memory [{memory_id[:8]}] updated successfully."
        return f"Failed to update memory with ID '{memory_id}'."

    def _delete_memory(self, channel_id: str, memory_id: str) -> str:
        """Delete a memory."""
        # Try to find memory by prefix match if short ID provided
        memories = self._state.load_memories(channel_id)
        full_id = None
        for memory in memories:
            if memory["id"].startswith(memory_id):
                full_id = memory["id"]
                break

        if not full_id:
            return f"Memory with ID '{memory_id}' not found."

        if self._state.delete_memory(full_id):
            return f"Memory [{memory_id[:8]}] deleted successfully."
        return f"Failed to delete memory with ID '{memory_id}'."

    def _get_memories_context(self, channel_id: str) -> str:
        """Get formatted memories for inclusion in instructions."""
        memories_text = self._state.get_memories_text(channel_id)
        if not memories_text:
            return ""
        return f"\n\n## Long-Term Memories\nThese are important facts and information you've saved about this channel. Use them to provide personalized and contextually relevant responses:\n{memories_text}"

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    async def respond(
        self,
        channel_id: str,
        user_message: str,
        user_name: str,
        image_urls: List[str] = [],
        server_context: str | None = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:  # noqa: D401 — short description style
        """Generate the assistant's reply for a user message.

        Args:
            channel_id (str): The ID of the conversation channel.
            user_message (str): The raw content of the user's message.
            user_name (str): The name of the user.
            image_urls (List[str]): The URLs of the images attached to the user's message.
        Returns:
            str: The assistant's final textual response ready to be sent.
        """
        # ------------------------------------------------------------------
        # Step 1 – append user message to conversation history
        # ------------------------------------------------------------------
        # Fetch or load conversation history for this channel.
        if channel_id not in self._histories:
            # Load from DB – may be empty list
            hist = self._state.load_history(channel_id)
            self._histories[channel_id] = hist

        user_entry: Dict[str, Any] = {"role": "user"}
        if image_urls and len(image_urls) > 0:
            image_context_message = (
                "Here are the urls for the attached image(s):\n" + "\n".join(image_urls)
            )
            user_entry["content"] = [
                {
                    "type": "input_text",
                    "text": f"<{user_name}> {user_message}\n{image_context_message}",
                },
                *[
                    {"type": "input_image", "image_url": image_url}
                    for image_url in image_urls
                ],
            ]
        else:
            user_entry["content"] = f"<{user_name}> {user_message}"
        self._append_and_persist(channel_id, user_entry)

        if channel_id not in self._responding or channel_id not in self._queued:
            self._responding[channel_id] = False
            self._queued[channel_id] = False

        # ------------------------------------------------------------------
        # Atomic check/set for responding & queued flags (race-free)
        # ------------------------------------------------------------------
        lock = self._locks.setdefault(channel_id, asyncio.Lock())
        async with lock:
            if self._responding[channel_id]:
                # Another coroutine is already generating a reply – only mark
                # that we have more messages queued and exit.
                self._queued[channel_id] = True
                return
            self._responding[channel_id] = True

        # ------------------------------------------------------------------
        # Step 2 – first call: let the model decide whether to call a function
        # ------------------------------------------------------------------
        # Use try/finally so the flags are cleared even on errors or
        # if the async-generator is closed prematurely by the caller.
        try:
            turns = 0
            while turns <= self._maximum_turns:
                # Clear queued flag if it's set to prevent unneeded turns.
                if self._queued[channel_id]:
                    self._queued[channel_id] = False

                turns += 1

                last_turn = turns == self._maximum_turns
                if last_turn:
                    self._append_and_persist(
                        channel_id,
                        {
                            "role": "system",
                            "content": "You have reached the turn limit. If you have not completed all necessary actions, you can request the user to continue the conversation. Otherwise, respond normally.",
                        },
                    )
                raw_history = self._state.load_history(channel_id)
                history = self._prepare_history_for_model(raw_history)

                # Build instructions with long-term memories
                memories_context = self._get_memories_context(channel_id)
                full_instructions = self._instructions + memories_context
                if server_context:
                    full_instructions = "\n\n".join([full_instructions, server_context])

                # Build tools dynamically with current channel_id for defaults
                tools = self._build_tools(channel_id) if not last_turn else []

                response_kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "input": history,  # type: ignore[arg-type]
                    "tools": tools,  # type: ignore[arg-type]
                    "parallel_tool_calls": False,
                    "instructions": full_instructions,
                    "truncation": "auto",
                }

                reasoning_payload = self._reasoning_request_payload()
                if reasoning_payload:
                    response_kwargs.update(reasoning_payload)

                response = await self._safe_create_response(**response_kwargs)

                output_items = list(getattr(response, "output", []))
                tool_calls = []
                for item in output_items:
                    item_type = getattr(item, "type", None)
                    if item_type not in {"reasoning", "function_call", "message"}:
                        continue

                    try:
                        serialized = json.loads(item.model_dump_json())
                    except (TypeError, ValueError):
                        continue

                    self._append_and_persist(channel_id, serialized)

                    if item_type == "function_call":
                        tool_calls.append(item)

                if not tool_calls:
                    # No function calls – just return the model's text response
                    assistant_text_resp: str = getattr(response, "output_text", "")
                    yield {"type": "text", "content": assistant_text_resp, "channel_id": channel_id}

                    # We have produced a textual response, so end this respond() cycle, unless there are more messages queued.
                    if self._queued[channel_id]:
                        self._queued[channel_id] = False
                        continue
                    else:
                        break

                # ------------------------------------------------------------------
                # Step 3 – execute function calls and feed results back to the model
                # ------------------------------------------------------------------
                function_handlers = self._build_function_handlers(channel_id)

                for tool_call in tool_calls:
                    name: str = getattr(tool_call, "name")
                    args_str: str = getattr(tool_call, "arguments", "{}")
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}

                    # Get handler for this function
                    handler = function_handlers.get(name)
                    if handler is None:
                        # Unknown function – skip
                        continue

                    # Execute the handler
                    output_type: str | None = None
                    result: str | bytes | dict | None = None
                    success = False

                    try:
                        # Check if handler or underlying function is async
                        if asyncio.iscoroutinefunction(handler):
                            handler_result = await handler(**args)
                        else:
                            handler_result = handler(**args)
                            # Handle case where handler returns a coroutine (e.g., wrapped async functions)
                            if asyncio.iscoroutine(handler_result):
                                handler_result = await handler_result

                        # Handler returns a dict with type, content, channel_id
                        if isinstance(handler_result, dict):
                            output_type = handler_result.get("type")
                            result = handler_result.get("content")
                            target_channel_id = handler_result.get("channel_id", channel_id)
                            # If result content is a coroutine, await it
                            if asyncio.iscoroutine(result):
                                result = await result
                        else:
                            # Fallback for unexpected return format
                            output_type = None
                            result = handler_result
                            target_channel_id = channel_id
                        success = True
                    except Exception as e:
                        logging.error(f"Error calling function {name}: {e}")
                        result = f"Error calling function {name}: {e}"
                        target_channel_id = channel_id

                    # Append function call output to history
                    self._append_and_persist(
                        channel_id,
                        {
                            "type": "function_call_output",
                            "call_id": getattr(tool_call, "call_id"),
                            "output": (
                                result
                                if isinstance(result, str)
                                else str(result)[:1024]
                            ),
                        },
                    )

                    if not success:
                        continue

                    # Determine if this is a cross-channel operation
                    is_cross_channel = target_channel_id != channel_id

                    # Handle output based on type
                    if output_type and output_type.startswith("memory:"):
                        _, memory_action = output_type.split(":", 1)
                        if memory_action == "save":
                            self._append_and_persist(
                                channel_id,
                                {
                                    "role": "developer",
                                    "content": "The memory has been saved successfully. The user has been notified. You may continue the conversation naturally without repeating confirmation of the save.",
                                },
                            )
                            yield {"type": "text", "content": "*A new memory was created.*", "channel_id": channel_id}
                        elif memory_action == "update":
                            self._append_and_persist(
                                channel_id,
                                {
                                    "role": "developer",
                                    "content": "The memory has been updated successfully. The user has been notified. You may continue the conversation naturally without repeating confirmation of the update.",
                                },
                            )
                            yield {"type": "text", "content": "*A memory has been updated.*", "channel_id": channel_id}
                        elif memory_action == "delete":
                            self._append_and_persist(
                                channel_id,
                                {
                                    "role": "developer",
                                    "content": "The memory has been deleted successfully. The user has been notified. You may continue the conversation naturally without repeating confirmation of the deletion.",
                                },
                            )
                            yield {"type": "text", "content": "*A memory has been deleted.*", "channel_id": channel_id}
                        # list_memories result is returned to the model, not yielded to user

                    elif output_type == "text":
                        if is_cross_channel:
                            self._append_and_persist(
                                channel_id,
                                {
                                    "role": "developer",
                                    "content": f"The message has been sent to channel {target_channel_id}. You may inform the user that the message was sent successfully.",
                                },
                            )
                        yield {"type": "text", "content": result, "channel_id": target_channel_id}

                    elif output_type == "poll":
                        channel_note = f" in channel {target_channel_id}" if is_cross_channel else ""
                        self._append_and_persist(
                            channel_id,
                            {
                                "role": "developer",
                                "content": f"The poll has already been created and sent{channel_note}. You do not need to reshare the options. Instead, you can inform the user that the poll is live and encourage them to participate.",
                            },
                        )
                        yield {"type": "poll", "content": result, "channel_id": target_channel_id}

                    elif output_type == "image":
                        image_data = result
                        if image_data and isinstance(image_data, bytes):
                            if self._storage:
                                image_url, _ = prepare_image(image_data, self._storage)
                                image_context_message = (
                                    f"Here is the image url: {image_url} -"
                                )
                            else:
                                # No storage configured - this shouldn't happen in normal operation
                                # but handle gracefully by skipping URL generation
                                image_url = None
                                image_context_message = (
                                    "Image generated but no storage configured for URL."
                                )

                            channel_note = f" to channel {target_channel_id}" if is_cross_channel else " to the user"
                            self._append_and_persist(
                                channel_id,
                                {
                                    "role": "developer",
                                    "content": f"{image_context_message} The generated image has already been sent{channel_note}. You do not need to reshare the image data. Instead, you can describe the image, react to it, or simply inform the user that the image has been sent.",
                                },
                            )
                            self._append_and_persist(
                                channel_id,
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": f"[System: This is the image you just generated, provided for your reference. This is not a user message.] {image_context_message}",
                                        },
                                        {
                                            "type": "input_image",
                                            "image_url": image_url,
                                        },
                                    ],
                                },
                            )
                            yield {"type": "image_data", "content": image_data, "channel_id": target_channel_id}
                        else:
                            yield {"type": "text", "content": "Failed to generate image.", "channel_id": channel_id}

                    # output_type == None means no output to user
                    # The result is still recorded in history for the model to see
        finally:
            # Ensure flags are cleared so the channel never gets stuck
            self._responding[channel_id] = False
            self._queued[channel_id] = False

    # ------------------------------------------------------------------
    # Optional: reset conversation history
    # ------------------------------------------------------------------
    def reset(self, channel_id: str, clear_memories: bool = False) -> None:
        """Clear stored conversation history.

        Args:
            channel_id: The channel to reset.
            clear_memories: If True, also clear long-term memories for this channel.
        """
        self._histories.pop(channel_id, None)
        self._state.reset(channel_id)
        if clear_memories:
            self._state.clear_memories(channel_id)
        # Also clear any runtime flags/locks for this channel
        self._responding.pop(channel_id, None)
        self._queued.pop(channel_id, None)
        self._locks.pop(channel_id, None)

    def set_reasoning_level(self, level: str | None) -> None:
        """Update the reasoning level for future GPT-5 calls."""
        self._reasoning_level = self._normalize_reasoning_level(level)
        self._reasoning_warning_logged = False
        self._maybe_log_reasoning_ignored()

    def get_reasoning_level(self) -> str:
        """Return the canonical reasoning level currently configured."""
        return self._reasoning_level

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
        # Using ``self._client``
        logging.debug(
            "Calling OpenAI with payload: %s",
            {k: v for k, v in kwargs.items() if k != "input"},
        )
        return await self._client.responses.create(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------
    def _append_and_persist(self, channel_id: str, message: Dict[str, Any]) -> None:
        """Append a message to in-memory history and persist via StateStore."""
        try:
            history = self._state.append(channel_id, message)
            self._histories[channel_id] = history
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to persist agent message: %s", exc)

    def _prepare_history_for_model(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for item in history:
            sanitized_item = self._sanitize_response_item(item)
            sanitized.append(sanitized_item)
        return sanitized

    def _sanitize_response_item(self, item: Any) -> Any:
        if isinstance(item, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in item.items():
                if key == "status":
                    continue
                cleaned[key] = self._sanitize_response_item(value)
            return cleaned
        if isinstance(item, list):
            return [self._sanitize_response_item(elem) for elem in item]
        return item

    def _reasoning_request_payload(self) -> Dict[str, Any]:
        if not self._model_supports_reasoning(self._model):
            return {}

        if self._reasoning_level == "none":
            return {}

        effort = _REASONING_EFFORT.get(self._reasoning_level)
        if not effort:
            return {}

        return {"reasoning": {"effort": effort}}

    def _normalize_reasoning_level(self, level: str | None) -> str:
        if level is None:
            canonical = _DEFAULT_REASONING_LEVEL
        else:
            canonical = _REASONING_LEVEL_CANONICAL.get(level.strip().lower())
            if canonical is None:
                valid = sorted(set(_REASONING_LEVEL_CANONICAL.values()))
                raise ValueError(
                    "Unsupported reasoning level %r. Valid options: %s"
                    % (level, ", ".join(valid))
                )

        return canonical

    def _maybe_log_reasoning_ignored(self) -> None:
        if self._reasoning_warning_logged:
            return

        if self._reasoning_level != "none" and not self._model_supports_reasoning(
            self._model
        ):
            logging.info(
                "Reasoning level '%s' configured but model '%s' is not a GPT-5 family model; reasoning payload will be ignored.",
                self._reasoning_level,
                self._model,
            )
            self._reasoning_warning_logged = True

    @staticmethod
    def _model_supports_reasoning(model: str) -> bool:
        base_name = model.split("@", 1)[0]
        return base_name.startswith("gpt-5")
