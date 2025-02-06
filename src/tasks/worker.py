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

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

@celery_app.task(bind=True)
def scrape_space(self, space_url, save_to_watchlist_results=False):
    print(f"Processing space URL {space_url}...")
    try:
        res = process_twitter_space(space_url, "../../cookies.txt")
        if save_to_watchlist_results:
            r.rpush('watchlist_results', json.dumps(res))
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        return f"Error processing space: {e}"

@celery_app.task(bind=True)
def scrape_youtube_video(self, url):
    print(f"Processing YouTube video {url}...")
    try:
        res = process_youtube_video(url)
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        return f"Error processing YouTube video: {e}"

@celery_app.task(bind=True)
def scrape_article(self, url):
    print(f"Processing article {url}...")
    try:
        res = process_get_summary(url)
        return (res["exec_sum"], res["summary"])
    except Exception as e:
        return f"Error processing article: {e}"

@celery_app.task(bind=True)
def scrape_governance_forum(self, timeframe="1d", only_relevant=True):
    print("Started scrape_governance_forum task.")
    try:
        res = process_gov_digest(timeframe=timeframe, only_relevant=only_relevant)
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        return f"Error processing governance forums: {e}"

@celery_app.task(bind=True)
def scrape_twitter_digest(self, timeframe="1d", only_relevant=True):
    print("Started scrape_twitter_digest task.")
    try:
        exec_sum, notes = process_twitter_digest_command(timeframe=timeframe, only_relevant=only_relevant)
        return (exec_sum, notes)
    except Exception as e:
        return f"Error processing twitter digest: {e}"

@celery_app.task(bind=True)
def scrape_twitter_account_summary(self, username: str, timeframe="1d"):
    print(f"Started scrape_twitter_account_summary task for @{username}, timeframe={timeframe}")
    try:
        exec_sum, notes = process_account_summary(username, timeframe)
        return (exec_sum, notes)
    except Exception as e:
        return f"Error processing single-user Twitter summary: {e}"

@celery_app.task(bind=True)
def scrape_tweet_summary(self, url, parse_comments: bool = False):
    print(f"Started scrape_tweet_summary task for URL {url} with parse_comments={parse_comments}")
    try:
        res = process_tweet_summary(url, parse_comments)
        # We now expect the result to have keys 'exec_sum' and 'notes'
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        return f"Error processing tweet summary: {e}"
