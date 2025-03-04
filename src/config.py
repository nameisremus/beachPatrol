import os
import json
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("environment", "development")

# Discord
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", 0))

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_ECOSYSTEM_UPDATES_GROUPID = os.getenv("TG_ECOSYSTEM_UPDATES_GROUPID")

# Bot password
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "changeMe")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL")
OPENAI_GOVERNANCE_MODEL = os.getenv("OPENAI_GOVERNANCE_MODEL")
OPENAI_MODELS_LIST = os.getenv("OPENAI_MODELS", "").split(",")

# Celery / Broker
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Paths
TEMP_DIR = "/tmp"
PDF_TEMP_PATH = f"{TEMP_DIR}/temp_article.pdf"

# Governance forums mapping: load from JSON file if the path is provided.
GOVERNANCE_FORUMS_MAPPING_PATH = os.getenv("GOVERNANCE_FORUMS_MAPPING_PATH", "")
GOVERNANCE_FORUMS_MAPPING = {}
if GOVERNANCE_FORUMS_MAPPING_PATH:
    try:
        with open(GOVERNANCE_FORUMS_MAPPING_PATH, "r", encoding="utf-8") as f:
            GOVERNANCE_FORUMS_MAPPING = json.load(f)
    except Exception as e:
        print(f"Error loading governance forums from {GOVERNANCE_FORUMS_MAPPING_PATH}: {e}")

# Timeframe for governance forum extraction
GOVERNANCE_EXTRACT_TIMEFRAME = os.getenv("GOVERNANCE_EXTRACT_TIMEFRAME", "1d")

# Twikit login credentials
TWITTER_USER = os.getenv("TWITTER_USER", "")
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD", "")

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

# Twitter Proxies
PROXY_IP = os.getenv("PROXY_IP")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")

# Load content_tags.json
CONTENT_TAGS_PATH = os.getenv("CONTENT_TAGS_PATH", "")
CONTENT_TAGS = []
if CONTENT_TAGS_PATH:
    try:
        with open(CONTENT_TAGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_tags = data.get("tags", [])
            # Convert each tag dict to have a key "tag_name" for consistency
            CONTENT_TAGS = [{"tag_name": t.get("tag", "")} for t in raw_tags]
    except Exception as e:
        print(f"Error loading content tags from {CONTENT_TAGS_PATH}: {e}")


# Notion Integration
NOTION_KEY = os.getenv("NOTION_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_FULL_SUMMARY_PARENT_ID = os.getenv("NOTION_FULL_SUMMARY_PARENT_ID")

# Dumping .env variables
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sample_file = os.path.join(PROJECT_ROOT, ".env.sample")

# 3) Build a set of sensitive vars by scanning .env.sample
sensitive_vars = set()
print("Looking for .env.sample at:", sample_file)

try:
    with open(sample_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()

            # If sample’s value is empty, mark that key as sensitive
            if val == "":
                sensitive_vars.add(key)
except FileNotFoundError:
    print("[config.py] WARNING: .env.sample not found, skipping sensitive detection.")

# 4) Dump environment, masking sensitive ones
print("[config.py] Environment vars (masking sensitive ones):")

try:
    with open(sample_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key = line.split("=", 1)[0].strip()
            real_val = os.getenv(key, "(not set)")
            if key in sensitive_vars:
                print(f"  {key} = ******")
            else:
                print(f"  {key} = {real_val}")
except FileNotFoundError:
    pass

print("[config.py] Done loading config.")
