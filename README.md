# beachPatrol

🏖️🚓**beachPatrol** is a project by [LidoDAO contributors](https://www.lido.fi) with the purpose of amplifying the eyes and ears of the Lido DAO. It empowers contributors by providing pre-packaged AI meta-aggregation and parsing across articles, governance forums, Twitter profiles, Twitter Spaces, YouTube videos and more.

**Beach Patrol’s purpose** is straightforward: it continuously monitors these sources, processes the data using AI-driven summarization, and posts timely updates either via Discord or Telegram through a persistent, paginated interface. Additionally, the bot can publish summaries directly to a Notion database.

---

## Features

- **Governance Forum Summaries**  
  Ecosystem and governance forum updates summarization (over 70 forums supported).
  
- **Twitter Digest**  
  Generate a digest for a configurable list of Twitter accounts, with categorization by topics.

- **Twitter Account Summary**  
  Summarize tweets for a specific Twitter user over a given timeframe, with categorization and reference legend.

- **On-Demand Media Summarization**  
  Request summaries for articles, tweets, Twitter Spaces, and YouTube videos. The bot downloads/transcribes/scrapes as needed, then summarizes the content.

- **Multiple Interfaces**  
  - **Discord** slash commands (with interactive embeds).  
  - **Telegram** commands (with paginated messages and inline keyboards).

- **Optional Output to Notion**  
  Summaries can be published to a Notion database.

- **Custom Model and Prompt**  
  Certain commands can be run with a custom OpenAI model or a fully custom prompt. Multiple models can be defined in `.env`.

- **Paginated Views & Persistent State**  
  Summaries use multi-page “views” with buttons to navigate pages. Redis saves state so restarts do not break the interactive UI.

- **Basic Password Auth**  
  Both Discord and Telegram require a password to unlock commands (`BOT_PASSWORD` in `.env`).

---


## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart-docker-deployment)
- [Usage](#usage)
  - [Discord Usage](#discord-usage)
  - [Telegram Usage](#telegram-usage)
- [Notion Integration](#notion-integration)
- [Custom Models and Prompts](#custom-models-and-prompts)
- [Solution Design](#solution-design)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Before installing and running **beachPatrol**, you need to set up a Discord bot with the appropriate permissions and invite it to your server.


1. Clone the repository:

```bash
git clone https://github.com/nameisremus/beachPatrol
```

2. Navigate to the project directory:

```bash
cd beachPatrol
```

3.	Create an .env file from the sample:

```bash
cp .env.sample .env
```
Open the .env file and fill in the necessary values, or replace the placeholders with your actual configuration values. Save and close the .env file.

After setting up your .env, review the JSON sample files to add or update Twitter account sources and tweet categories as needed:
- tweet_accounts.sample.json: Use this sample to list the Twitter accounts you want tracked.
- governance_forums.sample.json: Use this sample to list the Governance forums you want tracked. Make sure the forum is running on Discourse.
- content_tags.sample.json: Use this sample to define content categories.

4. 	Run with Docker:
Build and start all services:

```bash
docker-compose build
docker-compose up -d
```

This launches:
- A Redis container (unless you point to an external Redis),
- A Celery worker container,
- Containers for the Discord and/or Telegram bots.

5. Check Logs (optional)
Confirm that the bot connects to Discord/Telegram correctly and is not reporting errors.

```bash
docker-compose logs -f
```

If everything is set up correctly, you can start-up the conversation with the bot in Telegram or Discord.

## Usage

Once the bot is online, it can accept commands in your Discord server or via Telegram. By default, many command parameters are **optional** (e.g., `model`, `prompt`, `parse_comments`, etc.). If you omit them, the bot uses default settings (such as the default model from `.env` and the standard summarization prompt).

---

### Discord Usage

1. **Unlock the Bot**  
   - After adding the bot to your Discord server, type the slash command:
     ```
     /password <BOT_PASSWORD>
     ```
     This authorizes your server to use the bot’s features (assuming `<BOT_PASSWORD>` matches the value in your `.env`).

2. **Available Commands**  
   Once the bot is unlocked, typing `/` in Discord will also show you all available commands, their parameters, and helpful hints. **Note**: all parameters in brackets (`[ ]`) are optional.

   - **`/generate_summary <url> [model=xxx] [prompt="..."]`**  
     Summarize any URL. Can be:
     - An article or PDF  
     - A Twitter Space link  
     - A YouTube video link  
     - A tweet or tweet thread  

     **Minimal Example** (defaults to your `.env` model/prompt):
     ```
     /generate_summary https://example.com/article
     ```
     **With custom model & prompt (Note. Do not forget the quotes when entering the custom prompt)**:
     ```
     /generate_summary https://example.com/article model=gpt-4.5-preview prompt="Please summarize this article in bullet points"
     ```

   - **`/generate_tweet_summary <tweet_url> [parse_comments=True/False] [model=xxx] [prompt="..."]`**  
     Summarize a tweet or thread.  
     - By default, `parse_comments=False` (no replies).
     - Add `parse_comments=True` to include replies.  

     **Minimal Example**:
     ```
     /generate_tweet_summary https://twitter.com/LidoFinance/status/123456789
     ```
     **Custom Example**:
     ```
     /generate_tweet_summary https://twitter.com/LidoFinance/status/123456789 parse_comments=True model=gpt-4.5-preview prompt="Summarize the tweets and then also summarize the sentiment from the comments."
     ```

   - **`/generate_twitter_digest [timeframe=1d/2d] [relevancy_filter=True/False]`**  
     Summarize tweets from a list of pre-configured Twitter accounts, grouped by category with links in a “legend.” Please note that due to the rate limits, this can take multiple hours, depending on the number of accounts tracked.
     - The default `timeframe` is `1d`.
     - The default `relevancy_filter` is `True`.

     **Minimal Example**:
     ```
     /generate_twitter_digest
     ```
     **Custom Example**:
     ```
     /generate_twitter_digest timeframe=2d relevancy_filter=False
     ```

   - **`/generate_twitter_account_summary <username> [timeframe=...] [model=xxx] [prompt="..."]`**  
     Summarize all tweets from a specific user during the specified timeframe.  
     - Default timeframe is `1d`.

     **Minimal Example**:
     ```
     /generate_twitter_account_summary LidoFinance
     ```
     **Custom Example**:
     ```
     /generate_twitter_account_summary LidoFinance timeframe=7d model=gpt-3.5-turbo prompt="Give me a concise, day-by-day summary"
     ```

   - **`/generate_gov_digest [timeframe=1d/2d/7d/etc] [relevancy_filter=True/False]`**  
     Summaries of multiple crypto governance forums  
     - Default timeframe is `1d`.
     - Default relevancy filter is `True`.

     **Minimal Example**:
     ```
     /generate_gov_digest
     ```
     **Custom Example**:
     ```
     /generate_gov_digest timeframe=3d relevancy_filter=True
     ```

---

### Telegram Usage

1. **Unlock the Bot**  
   - In a **Telegram** chat (private or group), enter:
     ```
     /password <BOT_PASSWORD>
     ```
     This unlocks commands in that chat. `<BOT_PASSWORD>` must match your `.env`.

2. **Commands & Examples**  
   All parameters are optional unless otherwise noted. If you leave them out, default values from `.env` or the code are used.

   - **`/generate_summary <url> [model=...] [prompt="..."]`**  
     Summarize any URL (article, PDF, tweet, Twitter Space, YouTube).  
     **Minimal Example**:
     ```
     /generate_summary https://example.com/some_report.pdf
     ```
     **Custom Example**:
     ```
     /generate_summary https://example.com/some_report.pdf model=gpt-4.5-preview prompt="Please give a concise 200-word summary"
     ```

   - **`/generate_tweet_summary <tweet_url> [parse_comments=...] [model=...] [prompt="..."]`**  
     Summarize a tweet or thread, optionally including comments.  
     - Default `parse_comments=False`.

     **Minimal Example**:
     ```
     /generate_tweet_summary https://x.com/LidoFinance/status/1899476896909345019
     ```
     **Custom Example**:
     ```
     /generate_tweet_summary https://x.com/LidoFinance/status/1899476896909345019 parse_comments=True model=gpt-4.5-preview prompt="Please generate a summary for these tweets in 100 words or less, while also extracting the community feedback from the comments."
     ```

   - **`/generate_twitter_account_summary <username> [timeframe=...] [model=...] [prompt="..."]`**  
     Summarize tweets for a single user in a specific timeframe (default: `1d`).
     **Minimal Example**:
     ```
     /generate_twitter_account_summary LidoFinance
     ```
     **Custom Example**:
     ```
     /generate_twitter_account_summary LidoFinance 7d model=gpt-4.5-preview prompt="Please summarize these tweets and extract the most interesting URLs separately."
     ```

   - **`/generate_twitter_digest [timeframe=...] [relevancy_filter=...]`**  
     Summarize multiple pre-configured Twitter accounts.  
     - Default `timeframe=1d`, `relevancy_filter=True`.
     **Minimal Example**:
     ```
     /generate_twitter_digest
     ```
     **Custom Example**:
     ```
     /generate_twitter_digest timeframe=1d relevancy_filter=False
     ```

   - **`/generate_gov_digest [timeframe=...] [relevancy_filter=...]`**  
     Summaries of multiple governance forums.  
     **Minimal Example**:
     ```
     /generate_gov_digest
     ```
     **Custom Example**:
     ```
     /generate_gov_digest timeframe=2d relevancy_filter=False
     ```

3. **Auto-Detection**  
   - In group chats, if you post a tweet link, the bot can detect and summarize it automatically, provided the chat is already unlocked with `/password`.

---

## Notion Integration

1. **Create** a Notion integration: [docs](https://developers.notion.com/docs/create-a-notion-integration#create-your-integration-in-notion).  
2. **Add** the integration to your desired Notion database. 
3. **Set** in `.env`:
   ```bash
   NOTION_KEY=secret_abc123
   NOTION_DATABASE_ID=xxxx...
   NOTION_FULL_SUMMARY_PARENT_ID=xxxx...
4. When you receive a summary, use the **Send to Notion** button in Discord or Telegram.

3. **Auto-Detection**  
   In group chats, if you drop a tweet link, the bot can auto-summarize it if unlocked.

---

## Custom Models and Prompts

- **Custom Model**  
  - In your `.env`, define a comma-separated list of models, e.g. `OPENAI_MODELS=gpt-4o-mini,gpt-4.5-preview`.  
  - The default is `OPENAI_DEFAULT_MODEL=gpt-4o-mini`.  
  - Override via `model=` parameter in commands.

- **Custom Prompt**  
  - Pass a custom prompt that overrides the built-in summarization logic:
    ```
    e.g.: /generate_summary <url> model=gpt-4.5-preview prompt="Summarize this in Shakespearean style"
    ```
  - The raw text is appended to your prompt, and the model follows your instructions.

----------

## Solution Design

**beachPatrol** is architected to efficiently process and summarize diverse content sources, Here's an overview of the solution's design:

1. **Discord Bot**  
   - Utilizes the `discord.py` library to provide slash commands and interactive embeds.  
   - Commands are forwarded as Celery tasks, storing intermediate data in Redis.  
   - Summaries are returned as paginated embeds, with buttons for navigating multiple pages of output.  
   - A password (`BOT_PASSWORD`) is required to unlock commands on each server, preventing unauthorized use.

2. **Telegram Bot**  
   - Uses `python-telegram-bot` to handle commands in Telegram chats or groups.  
   - Similar to Discord, commands and messages are converted into Celery tasks and stored in Redis.  
   - Results are sent back via paginated messages that persist across restarts.  
   - Also requires `/password <BOT_PASSWORD>` to unlock commands in each chat.

3. **Celery Workers**  
   - Perform all heavy lifting (scraping, downloading, transcription, summarization).  
   - Submits and retrieves tasks from Redis (the broker & state store).  

4. **Redis**  
   - Acts as both a **message broker** for Celery tasks and a **persistent store** for summary states (e.g., paginated views).  
   - When new content is summarized, the Celery worker returns results to Redis, which the bots retrieve for final posting.  
   - On restart, existing summary states are reloaded so interactive embeds in both Telegram and Discord can be re-hydrated.

5. **AI Summarization**  
   - Multiple OpenAI models can be used (GPT-3.5, GPT-4, etc.), or custom prompts can override the default prompts.  
   - Summaries are generated differently depending on the media type (article, PDF, Twitter Space, YouTube, tweet, etc.).  
   - Supports both an “executive summary” (short and focused) and a more detailed summary.

6. **Paginated Output & Persistent Views**  
   - Long-form text is chunked across multiple pages, either as Discord embeds or Telegram messages with inline “Next/Prev” buttons.  
   - Each page includes navigation controls for easy reading.  
   - Redis stores the current page index and associated message IDs, so these views remain interactive across bot restarts.

7. **Notion Integration**  
   - Summaries can optionally be sent to a specified Notion database, using a Notion integration token.  
   - A single button in the summary message (`“Send to Notion”`) lets users publish the executive and full summaries directly.

8. **Docker-Based Deployment**  
   - All components—Celery worker, Redis, Discord bot, and Telegram bot—can be containerized via Docker Compose.  
   - The `.env` file defines credentials (Discord, Telegram tokens), DB connections, password, and any custom environment variables.

----------

## Contributing

Contributions are always welcome! To contribute:
1.	Fork the repository and create a new branch for your changes.
2.	Ensure your code follows the project’s style and conventions.
3.	Write clear, concise commit messages.
4.	Thoroughly test your changes before submitting a pull request.
5.	Be responsive to feedback and willing to make adjustments as necessary.

For more detailed guidelines on contributing to open-source projects, refer to [GitHub’s official guide](https://guides.github.com/activities/contributing-to-open-source/).

## License

This project is licensed under the MIT License. For more information, please see the [LICENSE](./LICENSE) file.