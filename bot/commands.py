import discord
import io
import re
from typing import Literal
from datetime import datetime, timezone

from .handlers import (
    handle_ping,
    handle_roll,
    handle_generate_image,
    handle_generate_meme,
    handle_edit_image,
)


def setup_commands(tree: discord.app_commands.CommandTree) -> None:
    """Set up all slash commands for the bot."""

    @tree.command(name="ping", description="Replies with Pong!")
    async def _ping(
        interaction: discord.Interaction,
    ):  # noqa: D401, N802 — internal callback name
        """Simple health-check slash command."""
        response = await handle_ping()
        await interaction.response.send_message(response, ephemeral=True)

    @tree.command(name="roll", description="Roll dice with advanced options")
    async def _roll(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        dice_value: int,
        dice_count: int = 1,
        dice_modifier: int = 0,
        drop_n_lowest: int = 0,
        drop_n_highest: int = 0,
        private: bool = False,
    ) -> None:
        """Roll dice according to individual parameters."""
        response = await handle_roll(
            dice_value=dice_value,
            dice_count=dice_count,
            dice_modifier=dice_modifier,
            drop_n_lowest=drop_n_lowest,
            drop_n_highest=drop_n_highest,
        )
        await interaction.response.send_message(response, ephemeral=private)

    @tree.command(name="image", description="Generate an image based on a prompt")
    async def _image(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        prompt: str,
        aspect_ratio: Literal[
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
        ] = "1:1",
        private: bool = False,
    ) -> None:
        """Generate an image based on a prompt."""
        # Defer the response to avoid "Unknown interaction" error
        await interaction.response.defer(thinking=True, ephemeral=private)
        # Call the image generation handler (should return a URL or similar)
        image_data = await handle_generate_image(prompt, aspect_ratio)

        if not image_data:
            await interaction.edit_original_response(
                content="Failed to generate image."
            )
            return

        if not isinstance(image_data, bytes):
            await interaction.edit_original_response(
                content="Failed to generate image."
            )
            return

        # Sanitize filename and create attachment
        filesafe_name = re.sub(r"[^a-zA-Z0-9]", "_", prompt) or "image"
        filename = f"{filesafe_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg"
        file = discord.File(io.BytesIO(image_data), filename=filename)

        await interaction.edit_original_response(
            attachments=[file],
        )

    @tree.command(name="meme", description="Generate a meme based on a prompt")
    async def _meme(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        image_prompt: str,
        text: str,
        private: bool = False,
    ) -> None:
        """Generate a meme based on a prompt."""
        # Defer the response to avoid "Unknown interaction" error
        await interaction.response.defer(thinking=True, ephemeral=private)
        # Call the image generation handler (should return a URL or similar)
        image_data = await handle_generate_meme(image_prompt, text)

        if not image_data:
            await interaction.edit_original_response(content="Failed to generate meme.")
            return

        if not isinstance(image_data, bytes):
            await interaction.edit_original_response(content="Failed to generate meme.")
            return

        # Sanitize filename and create attachment
        filesafe_name = re.sub(r"[^a-zA-Z0-9]", "_", image_prompt) or "image"
        filename = f"{filesafe_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg"
        file = discord.File(io.BytesIO(image_data), filename=filename)

        await interaction.edit_original_response(
            attachments=[file],
        )

    @tree.command(name="edit", description="Edit an image based on a prompt")
    async def _edit(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        prompt: str,
        image: discord.Attachment,
        private: bool = False,
    ) -> None:
        """Edit an image based on a prompt."""
        # Defer the response to avoid "Unknown interaction" error
        await interaction.response.defer(thinking=True, ephemeral=private)

        # Read the uploaded attachment into memory so we can send raw bytes to the
        # image-editing handler. Replicate expects a file-like object or bytes,
        # not a Discord Attachment instance.
        try:
            attachment_bytes = await image.read()
        except Exception:  # noqa: BLE001 — fallback for any I/O issues
            await interaction.edit_original_response(
                content="Failed to read the attached image."
            )
            return

        # Call the image-editing handler (should return the edited image bytes)
        image_data = await handle_edit_image(prompt, attachment_bytes)

        if not image_data:
            await interaction.edit_original_response(content="Failed to edit image.")
            return

        if not isinstance(image_data, bytes):
            await interaction.edit_original_response(content="Failed to edit image.")
            return

        # Sanitize filename and create attachment
        filesafe_name = re.sub(r"[^a-zA-Z0-9]", "_", prompt) or "image"
        filename = f"{filesafe_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg"
        file = discord.File(io.BytesIO(image_data), filename=filename)

        await interaction.edit_original_response(
            attachments=[file],
        )
