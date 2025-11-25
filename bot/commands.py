import base64
import discord
import io
import re
import uuid
from typing import Literal, Any
from datetime import datetime, timezone

from .agent import Agent
from .handlers import (
    handle_ping,
    handle_roll,
    handle_generate_image,
    handle_generate_meme,
    handle_edit_image,
)
from .s3 import S3
from .utils import prepare_image


def setup_commands(
    tree: discord.app_commands.CommandTree, agent: Agent, s3: S3 | None = None
) -> None:
    """Set up all slash commands for the bot."""

    async def _get_image_user_message(
        base_message: str, image_data: bytes, filename: str, user_name: str
    ) -> dict[str, Any]:

        image_url, _ = prepare_image(image_data, s3, filename)
        image_context_message = (
            f"Here is the image url: {image_url}"
            if s3
            else "This is a base64 encoded image."
        )

        return {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"<{user_name}> {base_message} {image_context_message}",
                },
                {
                    "type": "input_image",
                    "image_url": image_url,
                },
            ],
        }

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
        if not private:
            agent._append_and_persist(
                str(interaction.channel_id),
                {
                    "role": "system",
                    "content": f"{interaction.user.name} rolled {dice_value} {dice_count} times with a modifier of {dice_modifier} and dropped {drop_n_lowest} lowest and {drop_n_highest} highest.",
                },
            )
            agent._append_and_persist(
                str(interaction.channel_id),
                {
                    "role": "user",
                    "content": f"<{interaction.user.name}> {response}",
                },
            )

    @tree.command(name="image", description="Generate an image based on a prompt")
    async def _image(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        prompt: str,
        aspect_ratio: Literal[
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
            "3:2",
            "2:3",
            "21:9",
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
        key = f"images/{interaction.channel_id}/{uuid.uuid4()}.jpg"
        file = discord.File(io.BytesIO(image_data), filename=key)

        await interaction.edit_original_response(
            attachments=[file],
        )
        if not private:
            agent._append_and_persist(
                str(interaction.channel_id),
                {
                    "role": "system",
                    "content": f"{interaction.user.name} generated an image with the prompt: {prompt}.",
                },
            )
            user_message = await _get_image_user_message(
                "Here is the generated image.", image_data, key, interaction.user.name
            )
            agent._append_and_persist(
                str(interaction.channel_id),
                user_message,
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
        key = f"images/{interaction.channel_id}/{uuid.uuid4()}.jpg"
        file = discord.File(io.BytesIO(image_data), filename=key)

        await interaction.edit_original_response(
            attachments=[file],
        )
        if not private:
            agent._append_and_persist(
                str(interaction.channel_id),
                {
                    "role": "system",
                    "content": f"{interaction.user.name} generated a meme with the prompt: {image_prompt} and text: {text}.",
                },
            )
            user_message = await _get_image_user_message(
                "Here is the generated meme.", image_data, key, interaction.user.name
            )
            agent._append_and_persist(
                str(interaction.channel_id),
                user_message,
            )

    @tree.command(name="edit", description="Edit/combine images based on a prompt")
    @discord.app_commands.describe(
        prompt="Describe the edits or how to combine the images",
        image1="First image (required)",
        image2="Second image (optional)",
        image3="Third image (optional)",
        private="Whether to hide the response",
    )
    async def _edit(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        prompt: str,
        image1: discord.Attachment,
        image2: discord.Attachment | None = None,
        image3: discord.Attachment | None = None,
        private: bool = False,
    ) -> None:
        """Edit or combine images based on a prompt."""
        # Defer the response to avoid "Unknown interaction" error
        await interaction.response.defer(thinking=True, ephemeral=private)

        # Collect all provided attachments
        attachments = [image1]
        if image2:
            attachments.append(image2)
        if image3:
            attachments.append(image3)

        # Read the uploaded attachments into memory
        attachment_bytes_list: list[bytes] = []
        original_keys: list[str] = []
        try:
            for attachment in attachments:
                data = await attachment.read()
                attachment_bytes_list.append(data)
                original_keys.append(
                    f"images/{interaction.channel_id}/{attachment.id}.jpg"
                )
        except Exception:  # noqa: BLE001 — fallback for any I/O issues
            await interaction.edit_original_response(
                content="Failed to read the attached image(s)."
            )
            return

        # Call the image-editing handler (should return the edited image bytes)
        image_data = await handle_edit_image(prompt, attachment_bytes_list)

        if not image_data:
            await interaction.edit_original_response(content="Failed to edit image(s).")
            return

        if not isinstance(image_data, bytes):
            await interaction.edit_original_response(content="Failed to edit image(s).")
            return

        # Sanitize filename and create attachment
        edited_key = f"images/{interaction.channel_id}/{uuid.uuid4()}.jpg"
        file = discord.File(io.BytesIO(image_data), filename=edited_key)

        await interaction.edit_original_response(
            attachments=[file],
        )
        if not private:
            image_count = len(attachment_bytes_list)
            image_word = "image" if image_count == 1 else f"{image_count} images"
            agent._append_and_persist(
                str(interaction.channel_id),
                {
                    "role": "system",
                    "content": f"{interaction.user.name} edited {image_word} with the prompt: {prompt}.",
                },
            )
            # Log original images
            for i, (img_bytes, key) in enumerate(
                zip(attachment_bytes_list, original_keys)
            ):
                label = (
                    "Here is the original image."
                    if image_count == 1
                    else f"Here is original image {i + 1}."
                )
                user_message_original = await _get_image_user_message(
                    label,
                    img_bytes,
                    key,
                    interaction.user.name,
                )
                agent._append_and_persist(
                    str(interaction.channel_id),
                    user_message_original,
                )
            # Log edited result
            user_message_edited = await _get_image_user_message(
                "Here is the edited image.",
                image_data,
                edited_key,
                interaction.user.name,
            )
            agent._append_and_persist(
                str(interaction.channel_id),
                user_message_edited,
            )

    @tree.command(name="clear", description="Clear the conversation history")
    async def _clear(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
    ) -> None:
        """Clear the conversation history. Only allowed in DMs or by server admins."""
        await interaction.response.defer(thinking=True, ephemeral=True)

        # Check if DM or server admin
        if interaction.guild is None:
            # DM context, allow
            allowed = True
        else:
            # Guild context, check admin
            member = interaction.user
            # member is discord.Member in guild, discord.User in DM
            allowed = False
            if isinstance(member, discord.Member):
                allowed = member.guild_permissions.administrator

        if not allowed:
            await interaction.edit_original_response(
                content="You must be a server admin to clear the conversation history in this channel."
            )
            return

        try:
            agent.reset(str(interaction.channel_id))
            await interaction.edit_original_response(content="Memory wiped.")
        except Exception as e:
            await interaction.edit_original_response(
                content=f"Failed to wipe memory: {e}"
            )
