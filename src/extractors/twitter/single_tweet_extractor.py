import re
import asyncio
from datetime import timezone

from tweety import Twitter
from tweety.types.twDataTypes import Tweet, SelfThread, ConversationThread
from core.core import summarize_transcript, get_executive_summary
from .digest_extractor import _expand_tweet_and_threads
from config import TWITTER_USER, TWITTER_PASSWORD

def parse_tweet_id_from_url(tweet_url: str) -> str:
    pattern = r"/status/(\d+)"
    match = re.search(pattern, tweet_url)
    return match.group(1) if match else ""

async def fetch_single_tweet_and_thread(tweet_id: str) -> list[Tweet]:
    app = Twitter("session")
    if TWITTER_USER and TWITTER_PASSWORD:
        print(f"[fetch_single_tweet_and_thread] Logging in as {TWITTER_USER}...")
        await app.sign_in(TWITTER_USER, TWITTER_PASSWORD)
        print(f"[fetch_single_tweet_and_thread] Logged in user: {app.me}")

    detail_obj = await app.tweet_detail(tweet_id)
    tweets_list = []

    if isinstance(detail_obj, SelfThread):
        await detail_obj.expand()
        tweets_list.extend(detail_obj.tweets)
    elif detail_obj.threads:
        for tw in detail_obj.threads:
            tweets_list.append(tw)
    else:
        tweets_list.append(detail_obj)

    expanded = []
    for tw in tweets_list:
        expansion_res = await _expand_tweet_and_threads(app, tw)
        for item in expansion_res["expanded_list"]:
            expanded.append(item["tweet_obj"])

    return expanded

async def fetch_tweet_comments(original_tweet: Tweet, pages=1) -> list[Tweet]:
    """
    Fetch all comments (replies) posted in response to a single Tweet.
    This returns a list of TWEET objects if we flatten the ConversationThreads.
    """
    if not original_tweet:
        return []
    try:
        convo_threads = await original_tweet.get_comments(pages=pages, wait_time=2)
        if not convo_threads:
            return []
        
        # Flatten them from ConversationThread -> list[Tweet]
        flattened_tweets = []
        for cthread in convo_threads:
            # cthread is a ConversationThread with cthread.tweets: list[Tweet]
            flattened_tweets.extend(cthread.tweets)
            # If needed, also flatten deeper levels:
            # e.g. for subReply in cthread.replies: ...
        
        return flattened_tweets
    except Exception as e:
        print(f"[fetch_tweet_comments] Error fetching comments for tweet {original_tweet.id}: {e}")
        return []

def extract_single_tweet_details(tweet_url: str, fetch_comments: bool = False) -> dict:
    tweet_id = parse_tweet_id_from_url(tweet_url)
    if not tweet_id:
        return {
            'text': "Error: Could not parse tweet ID",
            'summary': "",
            'authors': [],
            'publish_date': None,
        }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        expanded_tweets = loop.run_until_complete(fetch_single_tweet_and_thread(tweet_id))
        comments_list = []
        if fetch_comments and expanded_tweets:
            root_tweet = expanded_tweets[0]
            # fetch 1 page of comments, or more if you like
            comments_list = loop.run_until_complete(fetch_tweet_comments(root_tweet, pages=1))
    except Exception as e:
        return {
            'text': f"Error fetching tweet ID={tweet_id}: {e}",
            'summary': "",
            'authors': [],
            'publish_date': None,
        }
    finally:
        loop.close()

    # Merge main tweets + optional comment tweets
    all_tweets = expanded_tweets[:]
    if fetch_comments and comments_list:
        all_tweets.extend(comments_list)

    sorted_tweets = sorted(all_tweets, key=lambda t: t.created_on or 0)
    snippet_list = []
    authors = set()
    publish_date = None

    for tw in sorted_tweets:
        # If created_on is None for some reason, skip or treat as 1970
        dt = tw.created_on
        if dt is None:
            # fallback
            dt_str = "(no date)"
        else:
            dt_utc = dt.replace(tzinfo=timezone.utc)
            if publish_date is None or dt_utc < publish_date:
                publish_date = dt_utc
            dt_str = dt_utc.isoformat()

        user_name = "unknown"
        if tw.author and tw.author.username:
            user_name = tw.author.username

        authors.add(user_name)
        snippet_list.append(f"@{user_name} ({dt_str}): {tw.text}")

    full_text = "\n\n".join(snippet_list)
    return {
        'text': full_text,
        'summary': "",
        'authors': list(authors),
        'publish_date': publish_date,
    }
