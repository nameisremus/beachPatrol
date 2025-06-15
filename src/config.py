import os
import json
from dotenv import load_dotenv
import subprocess

import logging_config
import logging

logger = logging.getLogger(__name__)

load_dotenv()

ENV = os.getenv("environment")

# Discord
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", 0))
ENABLE_DISCORD_DIGEST = os.getenv("ENABLE_DISCORD_DIGESTS").strip().upper() == "TRUE"

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_ECOSYSTEM_UPDATES_GROUPID = os.getenv("TG_ECOSYSTEM_UPDATES_GROUPID")
ENABLE_DAILY_TELEGRAM_TWITTER_DIGEST = os.getenv("ENABLE_DAILY_TELEGRAM_TWITTER_DIGEST").strip().upper() == "TRUE"

# Bot password
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "changeMe")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL")
OPENAI_GOVERNANCE_MODEL = os.getenv("OPENAI_GOVERNANCE_MODEL")
OPENAI_MODELS_LIST = os.getenv("OPENAI_MODELS", "").split(",")

# Celery / Broker
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")

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
    except Exception:
        logger.error(
            "Error loading governance forums mapping",
            extra={"path": GOVERNANCE_FORUMS_MAPPING_PATH},
            exc_info=True
        )

# Timeframe for governance forum extraction
GOVERNANCE_EXTRACT_TIMEFRAME = os.getenv("GOVERNANCE_EXTRACT_TIMEFRAME", "1d")

# Twitter login credentials and proxies
TWITTER_USERNAMES = [u.strip() for u in os.getenv("TWITTER_USERS", "").split(",") if u.strip()]
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD", "")

PROXIES_PATH = os.getenv("PROXIES_PATH", "")
PROXIES_LIST: list[dict] = []

if PROXIES_PATH:
    try:
        with open(PROXIES_PATH, newline="", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(":")] + ["", "", "", ""]
                host, port_str, user, pwd = parts[:4]

                # Skip a header like "ip:port:user:pass"
                if port_str.lower() == "port":
                    continue

                try:
                    port_int = int(port_str) if port_str else None
                except ValueError:
                    logger.warning(
                        "Proxy line skipped because of non-numeric port",
                        extra={"line": line}
                    )
                    continue

                PROXIES_LIST.append(
                    {
                        "host": host or None,
                        "port": port_int,
                        "username": user or None,
                        "password": pwd or None,
                    }
                )

        logger.info("Loaded proxies", extra={"count": len(PROXIES_LIST)})
    except Exception:
        logger.error(
            "Error loading proxies file",
            extra={"path": PROXIES_PATH},
            exc_info=True
        )

ACCOUNT_PROXY_MAP = list(zip(TWITTER_USERNAMES, PROXIES_LIST))
if TWITTER_USERNAMES and PROXIES_LIST and len(TWITTER_USERNAMES) != len(PROXIES_LIST):
    logger.warning(
        "TWITTER_USERNAMES count != proxies count",
        extra={"usernames": TWITTER_USERNAMES, "proxies": len(PROXIES_LIST)},
    )

# Loading Twitter accounts from json
TWITTER_ACCOUNTS_PATH = os.getenv("TWITTER_ACCOUNTS_PATH", "")
TWITTER_ACCOUNTS = []
if TWITTER_ACCOUNTS_PATH:
    try:
        with open(TWITTER_ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            TWITTER_ACCOUNTS = data.get("accounts", [])
    except Exception:
        logger.error(
            "Error loading twitter accounts",
            extra={"path": TWITTER_ACCOUNTS_PATH},
            exc_info=True
        )

# Building accounts dict
TWITTER_ACCOUNTS_DICT = {}
for acct in TWITTER_ACCOUNTS:
    uname = acct.get("username", "").lower()
    if uname:
        TWITTER_ACCOUNTS_DICT[uname] = {
            "display_name": acct.get("display_name", ""),
            "organization": acct.get("organization", "")
        }

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
    except Exception:
        logger.error(
            "Error loading content tags",
            extra={"path": CONTENT_TAGS_PATH},
            exc_info=True
        )

# Notion Integration
NOTION_KEY = os.getenv("NOTION_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_FULL_SUMMARY_PARENT_ID = os.getenv("NOTION_FULL_SUMMARY_PARENT_ID")

# Dumping .env variables
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sample_file = os.path.join(PROJECT_ROOT, ".env.sample")

# Build a set of sensitive vars by scanning .env.sample
sensitive_vars = set()
logger.info("Looking for .env.sample", extra={"path": sample_file})

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
    logger.warning(
        ".env.sample not found, skipping sensitive detection",
        extra={"path": sample_file}
    )

# Dump environment, masking sensitive ones
logger.info("Environment vars (masking sensitive ones)")

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
            masked_val = "******" if key in sensitive_vars else real_val
            logger.info("Environment var", extra={"key": key, "value": masked_val})
except FileNotFoundError:
    pass

# Git & Prometheus
def _git_value(args: list[str]) -> str:
    try:
        if not (PROJECT_ROOT / ".git").exists():
            return ""
        return subprocess.check_output(["git", *args],
                                       cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return ""

GIT_COMMIT = os.getenv("GIT_COMMIT", _git_value(["rev-parse", "HEAD"]))
GIT_BRANCH = os.getenv("GIT_BRANCH", _git_value(["rev-parse", "--abbrev-ref", "HEAD"]))
GIT_TAG = os.getenv("GIT_TAG", _git_value(["describe", "--tags", "--abbrev=0"]))
VERSION = os.getenv("VERSION", _git_value(["describe", "--tags", "--always"]) or "dev")

PROMETHEUS_PREFIX = os.getenv("PROMETHEUS_PREFIX", "beachPatrol_")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9300"))


logger.info("Done loading config")
