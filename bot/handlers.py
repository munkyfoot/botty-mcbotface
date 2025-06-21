"""Business logic handlers for bot functionality."""

from typing import List


async def handle_ping() -> str:
    """Handle ping functionality and return the response message.

    Returns:
        str: The ping response message.
    """
    return "Pong! 🏓"


# ---------------------------------------------------------------------------
# Dice rolling handler
# ---------------------------------------------------------------------------
import random


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
