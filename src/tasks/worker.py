import json
import redis
import time
from config import REDIS_HOST, REDIS_PORT, OPENAI_MODEL
from tasks.celery_config import app as celery_app
import logging

from extractors.twitter.spaces_extractor import process_twitter_space
from extractors.youtube_extractor import process_youtube_video
from commands.generate_summary.generate_summary_handler import process_get_summary
from commands.generate_gov_digest.gov_digest_handler import process_gov_digest
from commands.generate_twitter_digest.twitter_digest_handler import process_twitter_digest_command
from commands.generate_twitter_account_summary.account_summary_handler import process_account_summary
from commands.generate_tweet_summary.tweet_summary_handler import process_tweet_summary
from core.core import get_valid_model
import metrics.exporter

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
CACHE_TTL_SECONDS = 3600
logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def scrape_space(self, space_url, model=None, prompt=None, save_to_watchlist_results=False):
    logger.info(
        "Starting space scrape",
        extra={
            "space_url": space_url,
            "model": model,
            "prompt": prompt
        }
    )
    try:
        chosen_model = get_valid_model(model)
        res = process_twitter_space(space_url, "../../cookies.txt", model=chosen_model, prompt=prompt)
        # Optionally store in watchlist_results
        if save_to_watchlist_results:
            r.rpush('watchlist_results', json.dumps(res))

        return {
            "exec_sum": res["exec_sum"],
            "notes": res["notes"],
            "used_model": chosen_model,
            "custom_prompt_used": bool(prompt)
        }

    except Exception as e:
        logger.error(
            "Error processing space",
            extra={"space_url": space_url},
            exc_info=True
        )
        return {"error": "space_processing_failed"}


@celery_app.task(bind=True)
def scrape_youtube_video(self, url, model=None, prompt=None):
    logger.info(
        "Starting YouTube video scrape",
        extra={
            "url": url,
            "model": model,
            "prompt": prompt
        }
    )
    try:
        chosen_model = get_valid_model(model)
        res = process_youtube_video(url, model=chosen_model, prompt=prompt)
        return {
            "exec_sum": res["exec_sum"],
            "notes": res["notes"],
            "used_model": chosen_model,
            "custom_prompt_used": bool(prompt)
        }
    except Exception as e:
        logger.error(
            "Error processing YouTube video",
            extra={"url": url},
            exc_info=True
        )
        return f"Error processing YouTube video: {e}"


@celery_app.task(bind=True)
def scrape_article(self, url, model=None, prompt=None):
    logger.info(
        "Starting article scrape",
        extra={
            "url": url,
            "model": model,
            "prompt": prompt
        }
    )
    try:
        chosen_model = get_valid_model(model)
        res = process_get_summary(url, model=chosen_model, prompt=prompt)
        return {
            "exec_sum": res["exec_sum"],
            "notes": res["summary"],
            "used_model": chosen_model,
            "custom_prompt_used": bool(prompt)
        }
    except Exception as e:
        logger.error(
            "Error processing article",
            extra={"url": url},
            exc_info=True
        )
        return f"Error processing article: {e}"


@celery_app.task(bind=True)
def scrape_governance_forum(self, timeframe="1d", only_relevant=True):
    logger.info(
        "Started scrape_governance_forum task",
        extra={
            "timeframe": timeframe,
            "only_relevant": only_relevant
        }
    )
    try:
        # 1. Build a cache key from timeframe + only_relevant
        cache_key = f"gov_digest:{timeframe}:{only_relevant}"

        # 2. Check Redis for existing data
        cached_data = r.get(cache_key)
        if cached_data:
            cached_json = json.loads(cached_data)
            cached_ts = cached_json.get("ts", 0)

            # If it's fresher than CACHE_TTL_SECONDS, return immediately
            if time.time() - cached_ts < CACHE_TTL_SECONDS:
                logger.info(
                    "Returning cached gov_digest",
                    extra={
                        "timeframe": timeframe,
                        "only_relevant": only_relevant,
                        "cache_age_secs": time.time() - cached_ts
                    }
                )
                return {
                    "exec_sum": cached_json["exec_sum"],
                    "notes": cached_json["notes"],
                    "used_model": cached_json["used_model"],
                    "custom_prompt_used": cached_json["custom_prompt_used"]
                }

        # If no valid cache found, proceed as normal
        res = process_gov_digest(timeframe=timeframe, only_relevant=only_relevant)
        final_data = {
            "exec_sum": res["exec_sum"],
            "notes": res["notes"],
            "used_model": OPENAI_MODEL,
            "custom_prompt_used": False
        }

        # 3. Store new result in Redis with current timestamp
        final_data_with_ts = {
            **final_data,
            "ts": time.time()
        }
        r.set(cache_key, json.dumps(final_data_with_ts))

        return final_data
    except Exception as e:
        logger.error(
            "Error processing governance forums",
            extra={
                "timeframe": timeframe,
                "only_relevant": only_relevant
            },
            exc_info=True
        )
        return f"Error processing governance forums: {e}"


@celery_app.task(bind=True)
def scrape_twitter_digest(self, timeframe="1d", only_relevant=True):
    logger.info(
        "Started scrape_twitter_digest task",
        extra={
            "timeframe": timeframe,
            "only_relevant": only_relevant
        }
    )
    try:
        # No custom prompt for twitter_digest
        exec_sum, notes = process_twitter_digest_command(timeframe=timeframe, only_relevant=only_relevant)
        return {
            "exec_sum": exec_sum,
            "notes": notes,
            "used_model": OPENAI_MODEL,
            "custom_prompt_used": False
        }
    except Exception as e:
        logger.error(
            "Error processing twitter digest",
            extra={
                "timeframe": timeframe,
                "only_relevant": only_relevant
            },
            exc_info=True
        )
        return f"Error processing twitter digest: {e}"


@celery_app.task(bind=True)
def scrape_twitter_account_summary(self, username: str, timeframe="1d", model=None, prompt=None):
    logger.info(
        "Started scrape_twitter_account_summary task",
        extra={
            "username": username,
            "timeframe": timeframe,
            "model": model,
            "prompt": prompt
        }
    )
    try:
        chosen_model = get_valid_model(model)
        exec_sum, notes = process_account_summary(username, timeframe, model=chosen_model, prompt=prompt)
        return {
            "exec_sum": exec_sum,
            "notes": notes,
            "used_model": chosen_model,
            "custom_prompt_used": bool(prompt)
        }
    except Exception as e:
        logger.error(
            "Error processing single-user Twitter summary",
            extra={"username": username},
            exc_info=True
        )
        return f"Error processing single-user Twitter summary: {e}"


@celery_app.task(bind=True)
def scrape_tweet_summary(self, url, parse_comments: bool = False, model=None, prompt=None):
    logger.info(
        "Started scrape_tweet_summary task",
        extra={
            "url": url,
            "parse_comments": parse_comments,
            "model": model,
            "prompt": prompt
        }
    )
    try:
        chosen_model = get_valid_model(model)
        res = process_tweet_summary(url, parse_comments, model=chosen_model, prompt=prompt)
        return {
            "exec_sum": res["exec_sum"],
            "notes": res["summary"],
            "used_model": chosen_model,
            "custom_prompt_used": bool(prompt)
        }
    except Exception as e:
        logger.error(
            "Error processing tweet summary",
            extra={"url": url, "parse_comments": parse_comments},
            exc_info=True
        )
        return f"Error processing tweet summary: {e}"
