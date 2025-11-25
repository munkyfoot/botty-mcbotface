"""Business logic handlers for bot functionality."""

import random
import replicate
import asyncio
import io
import aiohttp
from typing import Literal

from bot.image_models import (
    AspectRatio,
    build_generation_params,
    build_editing_params,
)


async def handle_ping() -> str:
    """Handle ping functionality and return the response message.

    Returns:
        str: The ping response message.
    """
    return "Pong! 🏓"


async def handle_roll(
    dice_value: int,
    dice_count: int = 1,
    dice_modifier: int = 0,
    drop_n_lowest: int = 0,
    drop_n_highest: int = 0,
) -> str:
    """Roll dice based on explicit parameters.

    Args:
        dice_value: Number of sides on each die (e.g. 6 for d6).
        dice_count: How many dice to roll.
        dice_modifier: Flat modifier to add to the sum.
        drop_n_lowest: Discard this many lowest rolls before summing.
        drop_n_highest: Discard this many highest rolls before summing.

    Returns:
        str: Detailed roll explanation and final total.
    """
    if dice_count <= 0 or dice_value <= 0:
        return "Dice count and value must be positive integers."
    if dice_count > 100:
        return "Too many dice. Please roll at most 100 dice at a time."

    rolls = [random.randint(1, dice_value) for _ in range(dice_count)]

    # Apply dropping
    sorted_rolls = sorted(rolls)
    if drop_n_lowest > 0:
        sorted_rolls = sorted_rolls[drop_n_lowest:]
    if drop_n_highest > 0:
        sorted_rolls = sorted_rolls[: max(len(sorted_rolls) - drop_n_highest, 0)]

    kept_rolls = sorted_rolls
    total = sum(kept_rolls) + dice_modifier

    # Prepare dice count and pluralization
    dice_count_str = "a" if dice_count == 1 else str(dice_count)
    dice_plural = "s" if dice_count > 1 else ""

    # Format modifier
    modifier_str = ""
    if dice_modifier > 0:
        modifier_str = f" with a +{dice_modifier} modifier"
    elif dice_modifier < 0:
        modifier_str = f" with a -{abs(dice_modifier)} modifier"

    # Format drops
    drop_str = ""
    drop_parts = []
    if drop_n_lowest > 0:
        drop_count = f"{drop_n_lowest} " if drop_n_lowest > 1 else ""
        drop_parts.append(f"{drop_count}lowest")
    if drop_n_highest > 0:
        drop_count = f"{drop_n_highest} " if drop_n_highest > 1 else ""
        drop_parts.append(f"{drop_count}highest")
    if drop_parts:
        drop_str = ", dropped the " + " and ".join(drop_parts) + ","

    # Figure out which dice were dropped
    # Sort original rolls to match dropping logic
    sorted_rolls = sorted(rolls)
    dropped_dice = []
    kept_rolls = sorted_rolls
    if drop_n_lowest > 0:
        dropped_dice.extend(sorted_rolls[:drop_n_lowest])
        kept_rolls = kept_rolls[drop_n_lowest:]
    if drop_n_highest > 0:
        dropped_dice.extend(kept_rolls[-drop_n_highest:])
        kept_rolls = kept_rolls[: max(len(kept_rolls) - drop_n_highest, 0)]

    # For display, reconstruct the kept and dropped dice in order of original rolls
    # Mark dropped dice with strikethrough
    display_rolls = []
    temp_kept = kept_rolls.copy()
    temp_dropped = dropped_dice.copy()
    for r in rolls:
        if r in temp_kept:
            display_rolls.append(str(r))
            temp_kept.remove(r)
        elif r in temp_dropped:
            display_rolls.append(f"~~{r}~~")
            temp_dropped.remove(r)
        else:
            display_rolls.append(str(r))  # fallback

    # Format details string
    details_str = ""
    if dice_count > 1 or dice_modifier != 0:
        details_str = f" `[{', '.join(display_rolls)}]"
        if dice_modifier != 0:
            mod_sign = "+" if dice_modifier > 0 else "-"
            details_str += f" {mod_sign} {abs(dice_modifier)}"
        details_str += "`"

    return (
        f"*Rolled {dice_count_str} d{dice_value}{dice_plural}"
        f"{modifier_str}{drop_str} and got* ***{total}***.{details_str}"
    )


async def handle_generate_image(
    prompt: str,
    aspect_ratio: AspectRatio = "1:1",
    model_key: str | None = None,
) -> bytes | None:
    """Generate an image based on a prompt.

    Args:
        prompt: The prompt to generate an image from.
        aspect_ratio: The aspect ratio of the generated image.
        model_key: Optional model key to use (uses active model if not specified).

    Returns:
        bytes: The generated image, or None if generation failed.
    """
    model_id, params = build_generation_params(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        model_key=model_key,
    )

    output = await asyncio.to_thread(
        replicate.run,
        model_id,
        input=params,
    )

    # Some models return a list of URLs, others a single URL string.
    if isinstance(output, list):
        image_url = str(output[0]) if output else ""
    else:
        image_url = str(output)

    if not image_url:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                else:
                    return None
    except Exception as e:  # noqa: BLE001
        return None

    return data


async def handle_generate_meme(
    image_prompt: str,
    text: str,
) -> bytes | None:
    """Generate a meme based on a prompt.

    Args:
        image_prompt: The prompt to generate an image from.
        text: The text to add to the image.
    """
    meme_prompt = f"A meme of {image_prompt} with the text: {text}"
    image_data = await handle_generate_image(meme_prompt)
    return image_data


async def handle_edit_image(
    prompt: str,
    images: bytes | str | list[bytes | str],
    model_key: str | None = None,
) -> bytes | None:
    """Edit image(s) based on a prompt.

    Args:
        prompt: The prompt to edit the image(s) with.
        images: The image(s) to edit - single item or list of (bytes or URL).
        model_key: Optional model key to use (uses active model if not specified).

    Returns:
        bytes: The edited image, or None if editing failed.
    """
    # Normalize to list
    if isinstance(images, list):
        image_list = images
    else:
        image_list = [images]

    # Convert bytes to BytesIO
    processed_images = []
    for img in image_list:
        if isinstance(img, (bytes, bytearray)):
            processed_images.append(io.BytesIO(img))
        else:
            processed_images.append(img)

    model_id, params = build_editing_params(
        prompt=prompt,
        image_input=processed_images,
        model_key=model_key,
    )

    output = await asyncio.to_thread(
        replicate.run,
        model_id,
        input=params,
    )

    # Some models return a list of URLs, others a single URL string.
    if isinstance(output, list):
        image_url = str(output[0]) if output else ""
    else:
        image_url = str(output)

    if not image_url:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                else:
                    return None
    except Exception as e:  # noqa: BLE001
        return None

    return data
