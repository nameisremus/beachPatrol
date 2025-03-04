import json
import redis
from config import REDIS_HOST, REDIS_PORT
from tasks.celery_config import app as celery_app

from extractors.twitter.spaces_extractor import process_twitter_space
from extractors.youtube_extractor import process_youtube_video
from commands.generate_summary.generate_summary_handler import process_get_summary
from commands.generate_gov_digest.gov_digest_handler import process_gov_digest
from commands.generate_twitter_digest.twitter_digest_handler import process_twitter_digest_command
from commands.generate_twitter_account_summary.account_summary_handler import process_account_summary
from commands.generate_tweet_summary.tweet_summary_handler import process_tweet_summary
from core.core import get_valid_model

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

@celery_app.task(bind=True)
def scrape_space(self, space_url, model=None, prompt=None, save_to_watchlist_results=False):
    print(f"Processing space URL {space_url} with model={model} prompt={prompt}")
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
        return f"Error processing space: {e}"

@celery_app.task(bind=True)
def scrape_youtube_video(self, url, model=None, prompt=None):
    print(f"Processing YouTube video {url} with model={model} prompt={prompt}")
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
        return f"Error processing YouTube video: {e}"

@celery_app.task(bind=True)
def scrape_article(self, url, model=None, prompt=None):
    print(f"Processing article {url} with model={model} prompt={prompt}")
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
        return f"Error processing article: {e}"

@celery_app.task(bind=True)
def scrape_governance_forum(self, timeframe="1d", only_relevant=True):
    print("Started scrape_governance_forum task.")
    try:
        # This particular function does not accept a custom prompt, so custom_prompt_used=False
        res = process_gov_digest(timeframe=timeframe, only_relevant=only_relevant)
        # We can still pass a model if desired, but for now if no param was given, we do default:
        chosen_model = "gpt-3.5-turbo"  # or your default
        return {
            "exec_sum": res["exec_sum"],
            "notes": res["notes"],
            "used_model": chosen_model,
            "custom_prompt_used": False
        }
    except Exception as e:
        return f"Error processing governance forums: {e}"

@celery_app.task(bind=True)
def scrape_twitter_digest(self, timeframe="1d", only_relevant=True):
    print("Started scrape_twitter_digest task.")
    try:
        # No custom prompt in your code for twitter_digest
        chosen_model = "gpt-3.5-turbo"  # or whichever
        exec_sum, notes = process_twitter_digest_command(timeframe=timeframe, only_relevant=only_relevant)
        return {
            "exec_sum": exec_sum,
            "notes": notes,
            "used_model": chosen_model,
            "custom_prompt_used": False
        }
    except Exception as e:
        return f"Error processing twitter digest: {e}"

@celery_app.task(bind=True)
def scrape_twitter_account_summary(self, username: str, timeframe="1d", model=None, prompt=None):
    print(f"Started scrape_twitter_account_summary task for @{username}, timeframe={timeframe}, model={model}, prompt={prompt}")
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
        return f"Error processing single-user Twitter summary: {e}"

@celery_app.task(bind=True)
def scrape_tweet_summary(self, url, parse_comments: bool = False, model=None, prompt=None):
    print(f"Started scrape_tweet_summary task for URL {url} with parse_comments={parse_comments}, model={model}, prompt={prompt}")
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
        return f"Error processing tweet summary: {e}"
