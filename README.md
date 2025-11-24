# Botty McBotface

Botty McBotface is a feature-rich Discord bot powered by OpenAI's advanced language models. It offers conversational capabilities, image generation and editing, meme creation, and utility commands like dice rolling.

## Work in Progress & Customization

This repository contains the Discord bot I run on my own servers. It’s actively developed, so expect some rough edges, experimental features, and occasional breaking changes.

Quick start: drop your credentials into `.env`, adjust `settings.json` to suit your needs, install dependencies, and run the bot. For many use cases this setup will work out of the box.

Want to extend or change behavior? Fork the repo. If you need new commands, additional external API integrations, different models, or anything beyond what `settings.json` supports, create a fork and modify the code. This project is intended as a starting point for customization.

## Features

- **Conversational AI**: Chat with Botty in designated channels or DMs. It maintains context and can handle complex queries.
- **Image Generation**: Generate images from text prompts using the `/image` command.
- **Meme Generation**: Create memes on the fly with the `/meme` command.
- **Image Editing**: Edit existing images using the `/edit` command.
- **Dice Rolling**: Advanced dice roller with modifiers, drop lowest/highest, and multiple dice support via `/roll`.
- **Poll Monitoring**: Automatically announces results when a poll concludes.
- **Welcome Messages**: Greets the server when joining, introducing itself and its capabilities.
- **Configurable**: Highly customizable via `settings.json` and environment variables. Change the bot's name, personality, and other settings to fit your needs.

## Prerequisites
- Python >=3.10, <3.13
- A Discord Bot Token
- OpenAI API Key
- Replicate API Token (for image features)
- AWS S3 Credentials (optional, for image storage)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd botty-mcbotface
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create a `.env` file in the root directory with the following variables:
   ```env
   DISCORD_TOKEN=your_discord_bot_token
   OPENAI_API_KEY=your_openai_api_key
   REPLICATE_API_TOKEN=your_replicate_api_token
   
   # Optional: S3 Configuration for image persistence
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_BUCKET_NAME=your_bucket_name
   AWS_REGION_NAME=your_aws_region
   ```

4. **Configuration:**
   Create or modify `settings.json` in the root directory to customize the bot's behavior.
   ```json
   {
       "model": "gpt-5-mini",
       "instructions": "You are Botty McBotface, a helpful AI assistant.",
       "reasoning_level": "none",
       "enable_web_search": false,
       "maximum_turns": 10,
       "maximum_user_messages": 25,
       "auto_respond_channels": ["general", "bot-chat"],
       "dm_whitelist": [123456789012345678]
   }
   ```

## Usage

Run the bot using the following command:

```bash
python -m bot
```

## Commands

| Command | Description |
|---------|-------------|
| `/ping` | Checks if the bot is responsive. |
| `/roll` | Rolls dice (e.g., `dice_value: 20`, `dice_count: 2`). Supports modifiers and dropping lowest/highest. |
| `/image` | Generates an image based on a text prompt. Supports various aspect ratios. |
| `/meme` | Generates a meme using an image prompt and overlay text. |
| `/edit` | Edits an attached image based on a text prompt. |
| `/clear` | Clears the conversation history for the current channel (Admin/DM only). |

## Configuration Options (`settings.json`)

- **model**: The OpenAI model to use (default: `gpt-5-mini`).
- **instructions**: The system prompt that defines the bot's personality and behavior.
- **enable_web_search**: Enable or disable web search capabilities (if supported by the model/agent).
- **maximum_turns**: The maximum number of turns (tool calls / responses) the bot will take in a row.
- **maximum_user_messages**: The maximum number of user messages to hold in memory.
- **auto_respond_channels**: A list of channel names where the bot will automatically respond to all messages without being mentioned.
- **dm_whitelist**: A list of user IDs allowed to interact with the bot via Direct Messages. (e.g., `[123456789012345678, 987654321098765432]`)

## Data Persistence

### Agent Memory
The bot maintains conversation history using a local SQLite database (`agent_history.db`) located in the project root. This allows the bot to remember context across restarts. The database stores messages per channel and automatically trims older messages based on the `maximum_user_messages` setting in `settings.json` to keep the context window manageable.

### Image Storage
The bot supports two methods for handling generated or edited images:

1. **S3 Storage (Recommended)**:
   - Requires AWS credentials and an S3 bucket.
   - Images are uploaded to the configured S3 bucket.
   - Generates a public URL for the image, which is shared in the chat.
   - Images are persistent and can be accessed later.
   - To enable, configure the AWS variables in your `.env` file.

2. **Base64 Data URLs (Fallback)**:
   - Used automatically if S3 is not configured.
   - Images are compressed and encoded as Base64 data URLs.
   - The image data is embedded directly in the Discord message.
   - **Note**: These images are ephemeral and may not be viewable if the message is too large or on certain clients. They are not stored on a server.

## License

[MIT License](LICENSE)
