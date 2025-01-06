import json
import redis
from config import REDIS_HOST, REDIS_PORT
from tasks.celery_config import app as celery_app
from extractors.twitter_extractor import process_twitter_space
from extractors.youtube_extractor import process_youtube_video
from extractors.article_extractor import process_article
from extractors.governance_extractor import process_governance_forum

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

@celery_app.task
def scrape_space(space_url, save_to_watchlist_results=False):
    print(f"Processing space URL {space_url}...")
    try:
        res = process_twitter_space(space_url, "cookies.txt")
        if save_to_watchlist_results:
            r.rpush('watchlist_results', json.dumps(res))
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        print(f"Error processing {space_url}: {e}")
        return "Error processing space"

@celery_app.task
def scrape_youtube_video(url):
    print(f"Processing YouTube video {url}...")
    try:
        res = process_youtube_video(url)
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        print(f"Error processing YouTube video {url}: {e}")
        return "Error processing YouTube video"

@celery_app.task
def scrape_article(url):
    print(f"Processing article {url}...")
    try:
        res = process_article(url)
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        print(f"Error processing article {url}: {e}")
        return "Error processing article"

@celery_app.task
def scrape_governance_forum(timeframe="1d", only_relevant=True):
    print("Started scrape_governance_forum task.")
    try:
        res = process_governance_forum(timeframe=timeframe, only_relevant=only_relevant)
        print("Completed scrape_governance_forum task successfully.")
        return (res["exec_sum"], res["notes"])
    except Exception as e:
        print(f"Error processing governance forums: {e}")
        return "Error processing governance forums"