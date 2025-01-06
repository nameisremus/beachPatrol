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