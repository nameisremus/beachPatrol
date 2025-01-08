import os
import json
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("environment", "development")

# Discord
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", 0))

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_GOVERNANCE_MODEL = os.getenv("OPENAI_GOVERNANCE_MODEL")

# Celery / Broker
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Paths
TEMP_DIR = "/tmp"
PDF_TEMP_PATH = f"{TEMP_DIR}/temp_article.pdf"

# Governance forums from ENV
FORUMS_MAPPING_JSON = os.getenv("GOVERNANCE_FORUMS_MAPPING", "{}")
GOVERNANCE_FORUMS_MAPPING = json.loads(FORUMS_MAPPING_JSON)

# Timeframe for governance forum extraction
GOVERNANCE_EXTRACT_TIMEFRAME = os.getenv("GOVERNANCE_EXTRACT_TIMEFRAME", "1d")

# Twikit login credentials
TWITTER_USER = os.getenv("TWITTER_USER", "")
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD", "")

# Loading Twitter categories from json
TWEET_CATEGORIES_PATH = os.getenv("TWEET_CATEGORIES_PATH", "")
TWEET_CATEGORIES = []
if TWEET_CATEGORIES_PATH:
    try:
        with open(TWEET_CATEGORIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)  # e.g. { "tweet_categories": [{ "category_name": "..."}, ...]}
            TWEET_CATEGORIES = data.get("tweet_categories", [])
    except Exception as e:
        print(f"Error loading tweet categories from {TWEET_CATEGORIES_PATH}: {e}")

# Loading Twitter accounts from json
TWITTER_ACCOUNTS_PATH = os.getenv("TWITTER_ACCOUNTS_PATH", "")
TWITTER_ACCOUNTS = []
if TWITTER_ACCOUNTS_PATH:
    try:
        with open(TWITTER_ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)  # e.g. { "accounts": [ { "username": "...", "display_name": "...", "organization": "..."}]}
            TWITTER_ACCOUNTS = data.get("accounts", [])
    except Exception as e:
        print(f"Error loading twitter accounts from {TWITTER_ACCOUNTS_PATH}: {e}")

# Building accounts dict
TWITTER_ACCOUNTS_DICT = {}
for acct in TWITTER_ACCOUNTS:
    uname = acct.get("username", "").lower()
    if uname:
        TWITTER_ACCOUNTS_DICT[uname] = {
            "display_name": acct.get("display_name", ""),
            "organization": acct.get("organization", "")
        }