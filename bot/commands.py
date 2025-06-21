import discord

from .handlers import handle_ping


def setup_commands(tree: discord.app_commands.CommandTree) -> None:
    """Set up all slash commands for the bot."""

    @tree.command(name="ping", description="Replies with Pong!")
    async def _ping(
        interaction: discord.Interaction,
    ):  # noqa: D401, N802 — internal callback name
        """Simple health-check slash command."""
        response = await handle_ping()
        await interaction.response.send_message(response, ephemeral=True)
