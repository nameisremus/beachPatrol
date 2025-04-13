from celery import shared_task
import redis
from config import REDIS_HOST, REDIS_PORT
from tasks.celery_config import app as celery_app
from extractors.twitter.spaces_extractor import get_twitter_space_if_live

import logging

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
logger = logging.getLogger(__name__)

@shared_task
def check_if_live():
    logger.info("Checking watchlist for live spaces")
    try:
        urls_to_check = r.lrange('watchlist', 0, -1)
        for url in urls_to_check:
            url_str = url.decode('utf-8')
            tw_space = get_twitter_space_if_live(url_str, "../../cookies.txt")
            if tw_space is not None:
                logger.info(
                    "Space is live",
                    extra={
                        "space_url": url_str,
                        "live_url": tw_space["url"]
                    }
                )
                if r.get(f"scraping:{url_str}") == b"True":
                    logger.info(
                        "Already scraping, skipping",
                        extra={"space_url": url_str}
                    )
                    continue
                r.set(f"scraping:{url_str}", "True")
                celery_app.send_task(
                    'worker.scrape_space',
                    args=[tw_space["url"], True]
                )
                logger.info(
                    "Dispatched scrape_space task",
                    extra={"space_url": tw_space["url"]}
                )
    except Exception as e:
        logger.error(
            "Error in check_if_live task",
            exc_info=True
        )
