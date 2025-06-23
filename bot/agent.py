from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Tuple, AsyncGenerator
import os
import base64
import io
import aiohttp
from PIL import Image

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
from .s3 import S3


class Agent:
    """A simple stateful agent that uses OpenAI Responses API with function calling."""

    def __init__(
        self,
        model: str,
        instructions: str,
        enable_web_search: bool,
        maximum_turns: int,
        s3: S3 | None = None,
    ) -> None:

        self._model = model
        self._instructions = instructions
        self._enable_web_search = enable_web_search
        self._maximum_turns = maximum_turns
        self._s3 = s3
        self._client = AsyncOpenAI()
        # Conversation history as list of message dicts [{role, content}]

        # ------------------------------------------------------------------
        # Persistence setup via StateStore
        # ------------------------------------------------------------------
        self._state = StateStore()

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
                    },
                    "required": [
                        "dice_value",
                        "dice_count",
                        "dice_modifier",
                        "drop_n_lowest",
                        "drop_n_highest",
                    ],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "generate_image",
                "description": "Generate an image based on a prompt. Returns a URL of the generated image.",
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
                                "16:9",
                                "9:16",
                                "4:3",
                                "3:4",
                                "3:2",
                                "2:3",
                                "4:5",
                                "5:4",
                                "21:9",
                                "9:21",
                                "2:1",
                                "1:2",
                            ],
                            "default": "1:1",
                        },
                    },
                    "required": [
                        "prompt",
                        "aspect_ratio",
                    ],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "generate_meme",
                "description": "Generate a meme based on a prompt.",
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
                    },
                    "required": ["image_prompt", "text"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "edit_image",
                "description": "Edit an image based on a prompt.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to edit the image with. Describe the changes you want to make to the image.",
                        },
                        "image": {
                            "type": "string",
                            "description": "The url of the image to edit. IMPORTANT: The image must be a URL, not a base64 encoded string. You should only use urls for images available in the conversation history.",
                        },
                    },
                    "required": ["prompt", "image"],
                    "additionalProperties": False,
                },
            },
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
            "roll_dice": handle_roll,
            "generate_image": handle_generate_image,
            "generate_meme": handle_generate_meme,
            "edit_image": handle_edit_image,
        }

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    async def respond(
        self,
        channel_id: str,
        user_message: str,
        image_urls: List[str] = [],
    ) -> AsyncGenerator[Tuple[str, Any], None]:  # noqa: D401 — short description style
        """Generate the assistant's reply for a user message.

        Args:
            channel_id (str): The ID of the conversation channel.
            user_message (str): The raw content of the user's message.
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

        history = self._histories[channel_id]

        user_entry: Dict[str, Any] = {"role": "user"}
        if image_urls and len(image_urls) > 0:
            image_context_message = (
                "Here are the urls for the attached image(s):\n" + "\n".join(image_urls)
            )
            user_entry["content"] = [
                {
                    "type": "input_text",
                    "text": f"{user_message}\n{image_context_message}",
                },
                *[
                    {"type": "input_image", "image_url": image_url}
                    for image_url in image_urls
                ],
            ]
        else:
            user_entry["content"] = user_message
        self._append_and_persist(channel_id, user_entry)

        # ------------------------------------------------------------------
        # Step 2 – first call: let the model decide whether to call a function
        # ------------------------------------------------------------------
        turns = 0
        while turns <= self._maximum_turns:
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
            response = await self._safe_create_response(
                model=self._model,
                input=history,  # type: ignore[arg-type]
                tools=self._tools if not last_turn else [],  # type: ignore[arg-type]
                parallel_tool_calls=False,
                instructions=self._instructions,
                truncation="auto",
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
                yield "text", assistant_text_resp
                # We have produced a textual response, so end this respond() cycle.
                break

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
                result: str | bytes | None
                success = False
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = func(**args)
                    success = True
                except Exception as e:
                    logging.error(f"Error calling function {name}: {e}")
                    result = f"Error calling function {name}: {e}"

                # Append both the function call and its output to history
                self._append_and_persist(
                    channel_id, json.loads(tool_call.model_dump_json())
                )
                self._append_and_persist(
                    channel_id,
                    {
                        "type": "function_call_output",
                        "call_id": getattr(tool_call, "call_id"),
                        "output": (
                            result if isinstance(result, str) else str(result)[:1024]
                        ),
                    },
                )

                if not success:
                    continue

                if name == "ping":
                    yield "text", result

                if name == "roll_dice":
                    yield "text", result

                if name in ["generate_image", "generate_meme", "edit_image"]:
                    image_data = result

                    if image_data:
                        if isinstance(image_data, bytes):
                            if self._s3:
                                key = f"images/{channel_id}/{uuid.uuid4()}.jpg"
                                image_url = self._s3.public_upload(
                                    io.BytesIO(image_data), key
                                )
                                image_context_message = (
                                    f"Here is the image url: {image_url}"
                                )
                            else:
                                image = Image.open(io.BytesIO(image_data))
                                # Resize image to max 512px width/height, maintaining aspect ratio
                                max_size = 512
                                image.thumbnail(
                                    (max_size, max_size), Image.Resampling.LANCZOS
                                )
                                # Compress image to JPEG with quality=70
                                buffer = io.BytesIO()
                                image = image.convert("RGB")  # Ensure JPEG compatible
                                image.save(
                                    buffer, format="JPEG", quality=70, optimize=True
                                )
                                compressed_data = buffer.getvalue()
                                base64_image = base64.b64encode(
                                    compressed_data
                                ).decode()
                                image_url = f"data:image/jpeg;base64,{base64_image}"
                                image_context_message = (
                                    "This is a base64 encoded image."
                                )
                            self._append_and_persist(
                                channel_id,
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": f"Here is the image you generated. {image_context_message}",
                                        },
                                        {
                                            "type": "input_image",
                                            "image_url": image_url,
                                        },
                                    ],
                                },
                            )
                            yield "image_data", image_data  # Send uncompressed image
                        else:
                            yield "text", "Failed to generate image."
                    else:
                        yield "text", "Failed to generate image."

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
        history = self._histories.setdefault(channel_id, [])
        history.append(message)
        try:
            self._state.append(channel_id, message)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to persist agent message: %s", exc)
