# beachPatrol

🏖️🚓**beachPatrol** is a project by [LidoDAO contributors](https://www.lido.fi) with the purpose of amplifying the eyes and ears of the Lido DAO. It empowers contributors by providing pre-packaged AI meta-aggregation and parsing across articles, governance forums, Twitter profiles, Twitter Spaces and YouTube videos.

**Beach Patrol’s purpose** is straightforward: it continuously monitors these sources, processes the data using AI-driven summarization, and posts timely updates in a Discord channel via a persistent, paginated interface.

---

## Features

- **Governance Forum Summaries**. Ecosystem and Governance forum updates summarization, over 70 forums available

- **Twitter Digest**. Generate a digest for a customizable list of Twitter accounts, with categorization by topics

- **Twitter Account Summary**. Summarize tweets for a specific Twitter user over a given timeframe, with categorization and reference legend.

- **On-Demand Media Summarization**. Request summaries for articles, Twitter Spaces, and YouTube videos via Discord commands

- **Paginated Views**. Summaries sent to Discord channels as interactive embeds

- **Persistent State Management**. Summaries and views remain accessible across bot restarts

- **GPT-based Summaries**. Use a stock OpenAI model or refine your own

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
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

3. Install [Poetry](https://python-poetry.org/docs/#installation) if you haven't already

4. Install dependencies:

```bash
poetry install
```

5. [Download](https://redis.io/download) and install Redis if you haven't already. By default, beachPatrol uses port **6380**.

6. Start Redis on the desired port (for example, 6380):

```bash
redis-server --port 6380
```

7. Install `ffmpeg` and `ffprobe`:

**For macOS using Homebrew**:

```bash
brew install ffmpeg
```
More details here: https://bbc.github.io/bbcat-orchestration-docs/installation-mac-manual/

**For Ubuntu/Debian**:

```bash
sudo apt update
sudo apt install ffmpeg
```

**For Windows**:

- Download the latest static build from the [FFmpeg website](https://ffmpeg.org/download.html).
- Extract the downloaded files.
- Add the `bin` directory to your system's PATH environment variable.

8.	Create an .env file from the sample:

```bash
cp .env.sample .env
```
Open the .env file in a text editor and fill in the necessary values, or replace the placeholders with your actual configuration values. Save and close the .env file.

After setting up your .env, review the JSON sample files to add or update Twitter account sources and tweet categories as needed:

	- tweet_accounts.sample.json: Use this sample to list additional Twitter accounts. Follow the provided format to add more sources.
	- tweet_categories.sample.json: Use this sample to define additional tweet categories. Ensure the format remains consistent for proper loading.

9. Start the Celery worker:
Open a new terminal window and run:

```bash
cd src/tasks
celery -A worker worker --loglevel=info
```

10. Run the Discord bot
Open another terminal window and run:

```bash
cd src/interface
poetry run python discord_app.py
```

If everything is set up correctly, the bot should come online in your Discord server.

## Usage

Once the bot is online, it can accept slash commands in your Discord server. Here are the key commands and functionalities:

### Available Slash Commands

Within Discord, once the bot is running, you can use the following slash commands:
- `/generate_summary <url>`
Determines the media type of the provided URL (Twitter Space, YouTube video, or article/PDF), then processes it accordingly:
    - Twitter Spaces: Downloads the audio, transcribes it, and generates summaries.
    - YouTube Videos: Downloads the audio, transcribes it, and generates summaries.
    - Articles/PDFs: Scrapes the text and generates summaries.

The bot responds with both an executive summary and detailed notes, formatted in a paginated embed if necessary.
- `/generate_gov_digest [timeframe=1d/2d/etc] [relevancy_filter=True/False]`
    - Manually triggers the governance forum scraper. It checks multiple governance forums, optionally filters out irrelevant topics, and posts summarized updates in Discord.

- `/generate_twitter_digest [timeframe=1d/2d] [relevancy_filter=True/False]`
    - Gathers tweets from configured Twitter accounts, categorizes them, and returns a digest which includes a legend linking to specific tweets.

- `/generate_twitter_digest [timeframe=1d/2d] [relevancy_filter=True/False]`
    - Summarizes all tweets for a specific Twitter user within the given timeframe. The summary includes categorization and a legend of analyzed tweets, fetching additional pages as needed to cover the entire timeframe.

### Background Tasks

In addition to responding to commands, the bot runs several background tasks:
- `check_tasks` (every 5 seconds)
Monitors Celery tasks to check if any have completed. If a task is finished, it retrieves the result and posts the summary and notes to the appropriate Discord channel.
- `check_watchlist_results` (every 60 seconds)
Checks a Redis list named watchlist_results for new Twitter Spaces marked as live and sends their summaries to the Discord channel.
- `daily_scheduled_tasks` (every minute)
Combines daily scheduling for both governance and Twitter digests:
	- At 10:00 UTC, automatically runs the governance forum scraper.
	- At 07:00 UTC, automatically runs the Twitter digest.

----------

### Folder Structure
```
beachPatrol/
└── src/
    ├── config.py                 # Configuration settings and environment variables
    ├── core/
    │   ├── core.py               # Summarization & models
    │   └── prompts.py            # Prompts for generating the summaries
    ├── extractors/
    │   ├── article_extractor.py      # Article/PDFs logic
    │   ├── governance_extractor.py   # Governance forums logic
    │   ├── twitter/                  # Twitter-related logic
    │   │   ├── spaces_extractor.py   # Twitter Spaces logic
    │   │   ├── digest_extractor.py   # Multi-user Twitter digest logic
    │   │   └── account_extractor.py  # Single-user Twitter account summary logic
    │   └── youtube_extractor.py      # YouTube audio logic
    ├── interface/
    │   └── discord_app.py          # Main Discord bot code (slash commands, task loops)
    ├── tasks/
    │   ├── celery_config.py        # Celery configuration
    │   ├── monitor.py              # Periodic checks (e.g., for Twitter Spaces)
    │   └── worker.py               # Celery task definitions
    └── core/
        └── utils.py                # Utility functions and persistent views
```
----------

### Solution Design

**beachPatrol** is architected to efficiently process and summarize diverse content sources, Here's an overview of the solution's design:

1. **Discord Bot**:
    - Utilizes the `discord.py` library to interact with the Discord API.
    - Listens for slash commands and enqueues tasks to Celery workers via Redis.
    - Posts summaries back to Discord using paginated embeds with interactive buttons. This help with message limits in Discord, while also offerring a friendly UX for this use-case.

2. **Celery Workers**:
    - Handle the heavy lifting of scraping, transcribing, and summarizing content.
    - Communicates with Redis for task queuing and result storage.

3. **Redis**:
    - Serves as both the message broker for Celery and a storage for persistent state.
    - Stores task queues, results, and state information for paginated views to ensure persistence across bot restarts. This means that even though the bot is restarted, the previous messages and views are repopulated.
    - When a summary is posted, its state is saved in Redis, including the message ID and channel ID.
    - On bot startup, existing summaries are reloaded from Redis, and views are reattached to the original Discord messages to maintain interactivity.

4. **AI Summarization**:
    - Leverages various OpenAI models to generate executive summaries and detailed notes.
    - Differentiates between media types to apply appropriate summarization prompts and models.

5. **Paginated Embeds**:
    - Results are displayed in Discord as paginated embeds with navigation buttons, which have unique `custom_id`s to ensure persistent views across bot restarts.
    - State (such as the current page index) is stored in Redis.

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