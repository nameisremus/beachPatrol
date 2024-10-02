from twitter import process_twitter_space
from celery_config import app as celery_app
from monitor import check_if_live
import redis
import json
from youtube import process_youtube_video
from article import process_article

r = redis.Redis(host='localhost', port=6380)


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