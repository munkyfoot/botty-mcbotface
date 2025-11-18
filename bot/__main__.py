from dotenv import load_dotenv
import os
import discord
import io
from datetime import datetime, timedelta, timezone

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
            reasoning_level=self.settings.get("reasoning_level"),
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
        async def on_guild_join(guild: discord.Guild):
            """Called when the bot joins a new server."""
            # Try to find the first channel we can send messages to
            # Try to send a welcome message to the server's system channel or a welcome channel
            welcome_channel = guild.system_channel
            
            # If no system channel, look for a channel named 'welcome' or 'general'
            if not welcome_channel:
                for channel in guild.text_channels:
                    if channel.name.lower() in ['welcome', 'general']:
                        welcome_channel = channel
                        break
            
            # Send the welcome message if we found a suitable channel and have permissions
            if welcome_channel and welcome_channel.permissions_for(guild.me).send_messages:
                try:
                    channel_id = str(welcome_channel.id)
                    intro_message = f"Welcome to the {guild.name} server! Please introduce yourself to the server and tell us what you can do!"
                    await self._send_agent_response(welcome_channel, channel_id, intro_message, "onboarding_bot")
                except discord.HTTPException:
                    pass

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

            channel_id = str(message.channel.id)
            await self._send_agent_response(message.channel, channel_id, message.content, message.author.name, image_urls)

        @self.client.event
        async def on_poll_complete(poll: discord.Poll):
            """Called when a poll is completed."""
            try:
                # Announce the poll results in the same channel
                channel = poll.channel
                if channel and poll.options:
                    poll_message = f"The poll '{poll.question}' has completed! Here are the results:\n"
                    for option in poll.options:
                        poll_message += f"- {option.text}: {option.vote_count} votes\n"
                    
                    # Find the winning option(s) - handle ties
                    max_votes = max(option.vote_count for option in poll.options)
                    winners = [option for option in poll.options if option.vote_count == max_votes]
                    
                    if len(winners) == 1:
                        poll_message += f"\nThe winning option is: {winners[0].text} with {winners[0].vote_count} votes!"
                    elif len(winners) > 1:
                        winner_names = ", ".join(w.text for w in winners)
                        poll_message += f"\nIt's a tie! Winners: {winner_names} (each with {max_votes} votes)"
                    
                    channel_id = str(channel.id)
                    await self._send_agent_response(channel, channel_id, poll_message, "poll_bot")
            except discord.HTTPException:
                pass

    # ----------------------------
    # Helper methods
    # ----------------------------
    async def _send_agent_response(
        self, channel: discord.abc.Messageable, channel_id: str, prompt: str, username: str, image_urls: list[str] = None
    ):
        """Send agent response to a channel, handling text, images, and polls.
        
        Args:
            channel: The Discord channel to send messages to.
            channel_id: The channel ID string for agent context.
            prompt: The prompt to send to the agent.
            username: The username of the requester.
            image_urls: Optional list of image URLs to include in the request.
        """
        async with channel.typing():
            async for data_type, content in self.agent.respond(
                channel_id, prompt, username, image_urls or []
            ):
                if data_type == "text":
                    for chunk in chunk_text(content, 2000):
                        await channel.send(chunk)
                elif data_type == "image_data":
                    data = content
                    if not data or not isinstance(data, bytes):
                        await channel.send(
                            "Failed to download generated image."
                        )
                        return
                    # Sanitize filename and create attachment
                    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg"
                    file = discord.File(io.BytesIO(data), filename=filename)
                    await channel.send(file=file)
                elif data_type == "poll":
                    data = content
                    try:
                        poll = discord.Poll(
                            question=data["question"],
                            duration=timedelta(hours=data.get("duration", 24)),
                            multiple=data.get("multiple", False),
                        )

                        for option in data.get("options", []):
                            poll.add_answer(text=option)

                        await channel.send(poll=poll)
                    except Exception as e:
                        await channel.send(f"Failed to create poll: {e}")
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
