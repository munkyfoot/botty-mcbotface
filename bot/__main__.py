from dotenv import load_dotenv
import os
import discord
import io
from datetime import datetime, timezone

from bot.utils import prepare_image, chunk_text

from .commands import setup_commands
from .agent import Agent
from .config import load_settings
from .s3 import S3

load_dotenv()


class Bot:
    def __init__(self):
        # Configure privileged intents so the bot can read message content.
        self.intents = discord.Intents.default()
        self.intents.message_content = True

        # Create low-level client and command tree (for slash commands).
        self.client = discord.Client(intents=self.intents)
        self.tree = discord.app_commands.CommandTree(self.client)

        # Load settings
        self.settings = load_settings()

        if not self.settings or None in self.settings.values():
            raise ValueError("Settings are not loaded or are missing values")

        try:
            self.s3 = S3()
        except Exception as e:
            print(f"Error initializing S3: {e}")
            self.s3 = None

        self.agent = Agent(
            model=self.settings["model"],
            instructions=self.settings["instructions"],
            enable_web_search=self.settings["enable_web_search"],
            maximum_turns=self.settings["maximum_turns"],
            maximum_user_messages=self.settings["maximum_user_messages"],
            s3=self.s3,
        )

        self._auto_respond_channels = self.settings["auto_respond_channels"]
        self._dm_whitelist = self.settings["dm_whitelist"]

        # Set up slash commands from the commands module
        setup_commands(self.tree, self.agent, self.s3)

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

            # Only respond to DMs if the user is in the whitelist,
            # or to channels if the channel name is in auto_respond_channels.
            if isinstance(message.channel, discord.DMChannel):
                # DM: check if user is in whitelist
                if message.author.id not in self._dm_whitelist:
                    return
            else:
                # Channel: check if channel name is in auto_respond_channels
                channel_name = getattr(message.channel, "name", None)
                bot_mentioned = (
                    self.client.user in message.mentions if self.client.user else False
                )
                if (
                    channel_name is None
                    or channel_name not in self._auto_respond_channels
                ) and not bot_mentioned:
                    return

            # Get image attachments from the message, if any
            image_attachments = [
                attachment
                for attachment in getattr(message, "attachments", [])
                if getattr(attachment, "content_type", "")
                and attachment.content_type.startswith("image/")
            ]
            image_urls = []

            for attachment in image_attachments:
                image_data = await attachment.read()
                key = f"images/{message.channel.id}/{attachment.id}.jpg"
                image_url, _ = prepare_image(image_data, self.s3, key)
                image_urls.append(image_url)

            async with message.channel.typing():
                channel_id = str(message.channel.id)
                async for data_type, content in self.agent.respond(
                    channel_id, message.content, message.author.name, image_urls
                ):
                    if data_type == "text":
                        for chunk in chunk_text(content, 2000):
                            await message.channel.send(chunk)
                    elif data_type == "image_data":
                        data = content
                        if not data or not isinstance(data, bytes):
                            await message.channel.send(
                                "Failed to download generated image."
                            )
                            return
                        # Sanitize filename and create attachment
                        filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg"
                        file = discord.File(io.BytesIO(data), filename=filename)
                        await message.channel.send(file=file)
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
