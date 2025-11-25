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
- Cloud Storage: Cloudflare R2 (recommended) or AWS S3 (required for image features)

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
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   
   Required variables:
   ```env
   DISCORD_TOKEN=your_discord_bot_token
   OPENAI_API_KEY=your_openai_api_key
   REPLICATE_API_TOKEN=your_replicate_api_token
   ```
   
   Plus one of the cloud storage options below (see [Cloud Storage Setup](#cloud-storage-setup)).

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

The bot **requires** cloud storage for generated and edited images. This is necessary because:
- Generated images need permanent public URLs for the AI to reference them
- Discord CDN URLs expire and can't be used reliably for image history

Choose one of the following options:

#### Option 1: Cloudflare R2 (Recommended)

R2 is recommended because:
- **Generous free tier**: 10GB storage, 10 million reads/month, 1 million writes/month
- **No egress fees**: Unlimited free bandwidth for serving images
- **S3-compatible**: Uses the same API as AWS S3

Setup:
1. Create a [Cloudflare account](https://dash.cloudflare.com) and go to **R2 Object Storage**
2. Create a new bucket (note the bucket name)
3. **Enable public access**:
   - Select your bucket → **Settings**
   - Under **Public Development URL**, click **Enable**
   - Type `allow` to confirm and click **Allow**
   - Copy the **Public Bucket URL** (e.g., `https://pub-xxxx.r2.dev`)
4. **Create an API token**:
   - Go back to the R2 overview page
   - Click **Manage R2 API Tokens** (in the right sidebar)
   - Click **Create API token**
   - Choose **Object Read & Write** permission
   - Optionally scope to your specific bucket
   - Click **Create API Token**
   - Copy the **Access Key ID** and **Secret Access Key** (you won't see the secret again!)
5. Find your **Account ID** in the right sidebar of the R2 overview page
6. Add to your `.env`:
   ```env
   R2_ACCESS_KEY_ID=your_access_key_id
   R2_SECRET_ACCESS_KEY=your_secret_access_key
   R2_BUCKET_NAME=your_bucket_name
   R2_ACCOUNT_ID=your_cloudflare_account_id
   R2_PUBLIC_URL=https://pub-xxxx.r2.dev
   ```

#### Option 2: AWS S3

Traditional cloud storage option. May incur costs for storage and data transfer.

Setup:
1. Create an [AWS account](https://aws.amazon.com)
2. Create an S3 bucket with public read access enabled
3. Create an IAM user with S3 read/write permissions
4. Add to your `.env`:
   ```env
   AWS_ACCESS_KEY_ID=your_access_key_id
   AWS_SECRET_ACCESS_KEY=your_secret_access_key
   S3_BUCKET_NAME=your_bucket_name
   AWS_REGION=us-east-1
   ```

## License

[MIT License](LICENSE)
