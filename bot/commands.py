import discord

from .handlers import handle_ping, handle_roll


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
