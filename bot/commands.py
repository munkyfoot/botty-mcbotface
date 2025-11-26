import discord
import io
import uuid
from typing import Literal, Any, TYPE_CHECKING
from datetime import datetime, timezone

from .agent import Agent
from .handlers import (
    handle_ping,
    handle_roll,
    handle_generate_image,
    handle_generate_meme,
    handle_edit_image,
)
from .utils import prepare_image
from .image_models import get_models_info, get_model_keys, get_active_model_key

if TYPE_CHECKING:
    from .storage import StorageProvider

# Type alias for model choices - will be used to create Literal type dynamically
ModelChoice = Literal["seedream", "nano-banana", "nano-banana-pro"]


def setup_commands(
    tree: discord.app_commands.CommandTree, agent: Agent, storage: "StorageProvider"
) -> None:
    """Set up all slash commands for the bot."""

    def _get_scope_id(interaction: discord.Interaction) -> str:
        """Get the scope ID for history (server ID for guilds, channel ID for DMs)."""
        if interaction.guild:
            return str(interaction.guild.id)
        return str(interaction.channel_id)

    def _get_channel_prefix(interaction: discord.Interaction) -> str:
        """Get the channel prefix for messages (e.g., '[#general] ')."""
        if interaction.guild and hasattr(interaction.channel, 'name'):
            return f"[#{interaction.channel.name}] "
        return ""

    async def _get_image_user_message(
        base_message: str, image_data: bytes, user_name: str, channel_prefix: str = ""
    ) -> dict[str, Any]:

        image_url, _ = prepare_image(image_data, storage)
        image_context_message = f"Here is the image url: {image_url}"

        return {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"{channel_prefix}<{user_name}> {base_message} {image_context_message}",
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
            scope_id = _get_scope_id(interaction)
            channel_prefix = _get_channel_prefix(interaction)
            agent._append_and_persist(
                scope_id,
                {
                    "role": "system",
                    "content": f"{channel_prefix}{interaction.user.name} rolled {dice_value} {dice_count} times with a modifier of {dice_modifier} and dropped {drop_n_lowest} lowest and {drop_n_highest} highest.",
                },
            )
            agent._append_and_persist(
                scope_id,
                {
                    "role": "user",
                    "content": f"{channel_prefix}<{interaction.user.name}> {response}",
                },
            )

    @tree.command(name="image", description="Generate an image based on a prompt")
    @discord.app_commands.describe(
        prompt="The prompt describing the image to generate",
        aspect_ratio="Aspect ratio for the image",
        model="Image model to use (leave blank for default)",
        private="Whether to hide the response",
    )
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
        model: ModelChoice | None = None,
        private: bool = False,
    ) -> None:
        """Generate an image based on a prompt."""
        # Defer the response to avoid "Unknown interaction" error
        await interaction.response.defer(thinking=True, ephemeral=private)
        # Call the image generation handler (should return a URL or similar)
        image_data = await handle_generate_image(prompt, aspect_ratio, model_key=model)

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
            scope_id = _get_scope_id(interaction)
            channel_prefix = _get_channel_prefix(interaction)
            model_used = model or get_active_model_key()
            agent._append_and_persist(
                scope_id,
                {
                    "role": "system",
                    "content": f"{channel_prefix}{interaction.user.name} generated an image with the prompt: {prompt} (model: {model_used}).",
                },
            )
            user_message = await _get_image_user_message(
                "Here is the generated image.", image_data, interaction.user.name, channel_prefix
            )
            agent._append_and_persist(
                scope_id,
                user_message,
            )

    @tree.command(name="meme", description="Generate a meme based on a prompt")
    @discord.app_commands.describe(
        image_prompt="The prompt to generate an image from",
        text="The text to add to the image",
        model="Image model to use (leave blank for default)",
        private="Whether to hide the response",
    )
    async def _meme(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        image_prompt: str,
        text: str,
        model: ModelChoice | None = None,
        private: bool = False,
    ) -> None:
        """Generate a meme based on a prompt."""
        # Defer the response to avoid "Unknown interaction" error
        await interaction.response.defer(thinking=True, ephemeral=private)
        # Call the image generation handler (should return a URL or similar)
        image_data = await handle_generate_meme(image_prompt, text, model_key=model)

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
            scope_id = _get_scope_id(interaction)
            channel_prefix = _get_channel_prefix(interaction)
            model_used = model or get_active_model_key()
            agent._append_and_persist(
                scope_id,
                {
                    "role": "system",
                    "content": f"{channel_prefix}{interaction.user.name} generated a meme with the prompt: {image_prompt} and text: {text} (model: {model_used}).",
                },
            )
            user_message = await _get_image_user_message(
                "Here is the generated meme.", image_data, interaction.user.name, channel_prefix
            )
            agent._append_and_persist(
                scope_id,
                user_message,
            )

    @tree.command(name="edit", description="Edit/combine images based on a prompt")
    @discord.app_commands.describe(
        prompt="Describe the edits or how to combine the images",
        image1="First image (required)",
        image2="Second image (optional)",
        image3="Third image (optional)",
        model="Image model to use (leave blank for default)",
        private="Whether to hide the response",
    )
    async def _edit(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
        prompt: str,
        image1: discord.Attachment,
        image2: discord.Attachment | None = None,
        image3: discord.Attachment | None = None,
        model: ModelChoice | None = None,
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
        try:
            for attachment in attachments:
                data = await attachment.read()
                attachment_bytes_list.append(data)
        except Exception:  # noqa: BLE001 — fallback for any I/O issues
            await interaction.edit_original_response(
                content="Failed to read the attached image(s)."
            )
            return

        # Call the image-editing handler (should return the edited image bytes)
        image_data = await handle_edit_image(prompt, attachment_bytes_list, model_key=model)

        if not image_data:
            await interaction.edit_original_response(content="Failed to edit image(s).")
            return

        if not isinstance(image_data, bytes):
            await interaction.edit_original_response(content="Failed to edit image(s).")
            return

        # Sanitize filename and create attachment
        filename = f"edited_{uuid.uuid4()}.jpg"
        file = discord.File(io.BytesIO(image_data), filename=filename)

        await interaction.edit_original_response(
            attachments=[file],
        )
        if not private:
            scope_id = _get_scope_id(interaction)
            channel_prefix = _get_channel_prefix(interaction)
            image_count = len(attachment_bytes_list)
            image_word = "image" if image_count == 1 else f"{image_count} images"
            model_used = model or get_active_model_key()
            agent._append_and_persist(
                scope_id,
                {
                    "role": "system",
                    "content": f"{channel_prefix}{interaction.user.name} edited {image_word} with the prompt: {prompt} (model: {model_used}).",
                },
            )
            # Log original images
            for i, img_bytes in enumerate(attachment_bytes_list):
                label = (
                    "Here is the original image."
                    if image_count == 1
                    else f"Here is original image {i + 1}."
                )
                user_message_original = await _get_image_user_message(
                    label,
                    img_bytes,
                    interaction.user.name,
                    channel_prefix,
                )
                agent._append_and_persist(
                    scope_id,
                    user_message_original,
                )
            # Log edited result
            user_message_edited = await _get_image_user_message(
                "Here is the edited image.",
                image_data,
                interaction.user.name,
                channel_prefix,
            )
            agent._append_and_persist(
                scope_id,
                user_message_edited,
            )

    @tree.command(name="models", description="List available image models and their descriptions")
    async def _models(  # noqa: D401, N802 — internal callback name
        interaction: discord.Interaction,
    ) -> None:
        """List available image generation models."""
        models = get_models_info()
        default_key = get_active_model_key()

        lines = ["**Available Image Models:**\n"]
        for model in models:
            is_default = " *(default)*" if model["key"] == default_key else ""
            lines.append(f"**`{model['key']}`** - {model['name']}{is_default}")
            lines.append(f"> {model['description']}")
            lines.append(f"> Max input images: {model['max_input_images']}\n")

        await interaction.response.send_message("\n".join(lines), ephemeral=True)

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
                content="You must be a server admin to clear the conversation history."
            )
            return

        try:
            scope_id = _get_scope_id(interaction)
            agent.reset(scope_id)
            if interaction.guild:
                await interaction.edit_original_response(content="Server conversation history cleared.")
            else:
                await interaction.edit_original_response(content="Conversation history cleared.")
        except Exception as e:
            await interaction.edit_original_response(
                content=f"Failed to clear history: {e}"
            )
