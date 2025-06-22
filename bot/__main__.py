from dotenv import load_dotenv
import os
import discord

from .commands import setup_commands
from .agent import Agent

load_dotenv()


class Bot:
    def __init__(self):
        # Configure privileged intents so the bot can read message content.
        self.intents = discord.Intents.default()
        self.intents.message_content = True

        # Create low-level client and command tree (for slash commands).
        self.client = discord.Client(intents=self.intents)
        self.tree = discord.app_commands.CommandTree(self.client)

        # Instantiate the shared Agent
        self.agent = Agent()

        # Set up slash commands from the commands module
        setup_commands(self.tree)

        # ----------------------------
        # Event handlers
        # ----------------------------
        @self.client.event
        async def on_ready():  # noqa: D401 — Discord.py callback signature
            """Called when the bot finishes logging in."""
            user = self.client.user
            assert (
                user is not None
            )  # narrow typing: user is always set here after login
            print(f"Logged in as {user} (ID: {user.id})")

            # Sync application commands with Discord so that the slash
            # commands become available. Doing this on_every restart keeps
            # things simple during development.
            try:
                synced = await self.tree.sync()
                print(f"Synced {len(synced)} command(s) with Discord")
            except discord.HTTPException as exc:
                print(f"Failed to sync commands: {exc}")

        @self.client.event
        async def on_message(
            message: discord.Message,
        ):  # noqa: D401 — Discord.py callback signature
            """Basic prefix command to complement the slash command."""
            # Ignore messages from the bot itself to avoid infinite loops.
            if message.author == self.client.user:
                return

            async with message.channel.typing():
                channel_id = str(message.channel.id)
                async for data_type, content in self.agent.respond(
                    channel_id, message.content
                ):
                    if data_type in ["text", "image_url"]:
                        await message.channel.send(content)
                    elif data_type == "file":
                        await message.channel.send(file=discord.File(content))
                    else:
                        raise ValueError(f"Unknown data type: {data_type}")

    # ----------------------------
    # Public helpers
    # ----------------------------
    def run(self) -> None:
        """Start the Discord client using the token from the environment."""
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            raise ValueError("Environment variable DISCORD_TOKEN is not set")
        self.client.run(token)


if __name__ == "__main__":
    Bot().run()
