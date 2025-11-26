from dotenv import load_dotenv
import os
import discord
import io
from datetime import datetime, timedelta, timezone

from bot.utils import chunk_text

from .commands import setup_commands
from .agent import Agent
from .config import load_settings
from .storage import create_storage, StorageProvider
from .image_models import initialize_from_settings as initialize_image_model

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

        # Initialize image model from settings
        initialize_image_model(self.settings.get("default_image_model"))

        # Initialize cloud storage (required)
        self.storage = create_storage()
        storage_type = type(self.storage).__name__
        print(f"Cloud storage configured successfully ({storage_type}).")

        self.agent = Agent(
            model=self.settings["model"],
            instructions=self.settings["instructions"],
            enable_web_search=self.settings["enable_web_search"],
            maximum_turns=self.settings["maximum_turns"],
            maximum_history_chars=self.settings.get("maximum_history_chars"),
            reasoning_level=self.settings.get("reasoning_level"),
            storage=self.storage,
        )

        self._auto_respond_channels = self.settings["auto_respond_channels"]
        self._dm_whitelist = self.settings["dm_whitelist"]

        # Set up slash commands from the commands module
        setup_commands(self.tree, self.agent, self.storage)

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
                    auto_respond_info = (
                        f"Your designated auto-respond channels are: {', '.join(self._auto_respond_channels)}."
                        if self._auto_respond_channels
                        else ""
                    )
                    
                    intro_message = f"""Welcome to the {guild.name} server! Please introduce yourself to the server and tell us what you can do!
                    
                    Be sure to gear your introduction towards how you can contribute to this community specifically. Talk about the tools you have access to, and how you can use them to help members of this server.
                    
                    Lastly, inform users that they can interact with you by mentioning you in their messages{', sending a message in one of the designated auto-respond channels,' if self._auto_respond_channels else ''} or using slash commands.
                    
                    Users can mention you by typing '@' followed by your name, {self.client.user.name}, or by clicking on your name in the member list.

                    Your available commands can be accessed by typing '/' in the message input box.

                    {auto_respond_info}"""
                    await self._send_agent_response(welcome_channel, channel_id, intro_message, "onboarding_bot")
                except discord.HTTPException:
                    pass

        async def _handle_message(message: discord.Message, require_new_mention: bool = False, previous_mentions: set = None):
            """Process a message and send agent response if appropriate.
            
            Args:
                message: The Discord message to process.
                require_new_mention: If True, only respond if bot was newly mentioned.
                previous_mentions: Set of previous mentions (for edited messages).
            """
            # Ignore messages from the bot itself to avoid infinite loops.
            if message.author == self.client.user:
                return

            bot_mentioned = (
                self.client.user in message.mentions if self.client.user else False
            )

            # If we require a new mention, check that bot wasn't mentioned before
            if require_new_mention:
                bot_mentioned_before = (
                    self.client.user in previous_mentions if self.client.user and previous_mentions else False
                )
                if not bot_mentioned or bot_mentioned_before:
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
            # Upload user images to cloud storage for permanent URLs
            image_urls = []
            for attachment in image_attachments:
                try:
                    image_data = await attachment.read()
                    image_url = self.storage.public_upload(image_data, attachment.content_type)
                    image_urls.append(image_url)
                except Exception as e:
                    print(f"Failed to upload user image: {e}")

            channel_id = str(message.channel.id)
            await self._send_agent_response(message.channel, channel_id, message.content, message.author.name, image_urls)

        @self.client.event
        async def on_message(
            message: discord.Message,
        ):  # noqa: D401 — Discord.py callback signature
            """Basic prefix command to complement the slash command."""
            await _handle_message(message)

        @self.client.event
        async def on_message_edit(before: discord.Message, after: discord.Message):
            """Called when a message is edited."""
            # Handle case where bot was newly mentioned in the edit
            await _handle_message(after, require_new_mention=True, previous_mentions=set(before.mentions))

            # Check if the message has a poll and it just finished
            if (
                after.poll
                and after.poll.is_finalized()
                and (before.poll and not before.poll.is_finalized())
            ):
                try:
                    poll = after.poll
                    # Announce the poll results in the same channel
                    channel = after.channel
                    if channel and poll.answers:
                        poll_message = f"The poll '{poll.question}' has completed! Here are the results:\n"
                        for option in poll.answers:
                            poll_message += f"- {option.text}: {option.vote_count} votes\n"
                        
                        # Find the winning option(s) - handle ties
                        max_votes = max(option.vote_count for option in poll.answers)
                        winners = [option for option in poll.answers if option.vote_count == max_votes]
                        
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
        server_context = None
        guild = None
        if isinstance(channel, discord.TextChannel):
            guild = channel.guild
            server_context = f"Server name: {guild.name}"
            if guild.description:
                server_context += f"\nServer description: {guild.description}"
            if channel.topic:
                server_context += f"\nChannel topic: {channel.topic}"
            
            # Build list of available channels with their IDs
            available_channels = []
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).send_messages:
                    topic_info = f" - {ch.topic}" if ch.topic else ""
                    available_channels.append(f"  - #{ch.name} (ID: {ch.id}){topic_info}")
            
            if available_channels:
                server_context += f"\n\nAvailable channels you can send messages to:\n" + "\n".join(available_channels)

        async def get_target_channel(target_channel_id: str) -> discord.abc.Messageable | None:
            """Get the target channel, which may be different from the source channel."""
            if target_channel_id == channel_id:
                return channel
            
            # Cross-channel: look up the channel in the guild
            if guild:
                target = guild.get_channel(int(target_channel_id))
                if target and isinstance(target, discord.TextChannel):
                    if target.permissions_for(guild.me).send_messages:
                        return target
                    else:
                        await channel.send(f"I don't have permission to send messages to #{target.name}.")
                        return None
                else:
                    await channel.send(f"Could not find channel with ID {target_channel_id}.")
                    return None
            else:
                await channel.send("Cross-channel messaging is only available in servers, not DMs.")
                return None

        async with channel.typing():
            async for output in self.agent.respond(
                channel_id, prompt, username, image_urls or [], server_context=server_context
            ):
                # Output is now a dict with type, content, channel_id
                data_type = output.get("type")
                content = output.get("content")
                target_channel_id = output.get("channel_id", channel_id)
                
                # Get target channel (may be different from source)
                target_channel = await get_target_channel(target_channel_id)
                if target_channel is None:
                    continue  # Error already reported to source channel
                
                if data_type == "text":
                    for chunk in chunk_text(content, 2000):
                        await target_channel.send(chunk)
                elif data_type == "image_data":
                    data = content
                    if not data or not isinstance(data, bytes):
                        await channel.send(
                            "Failed to download generated image."
                        )
                        continue
                    # Sanitize filename and create attachment
                    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg"
                    file = discord.File(io.BytesIO(data), filename=filename)
                    await target_channel.send(file=file)
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

                        await target_channel.send(poll=poll)
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
