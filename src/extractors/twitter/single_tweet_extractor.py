import re
import asyncio
import os
from datetime import timezone

from tweety import Twitter, TwitterAsync
from tweety.types.twDataTypes import Tweet, SelfThread
from tweety.types import Proxy, PROXY_TYPE_HTTP
from .digest_extractor import _expand_tweet_and_threads
from config import TWITTER_USER, TWITTER_PASSWORD, PROXY_IP, PROXY_PORT, PROXY_USERNAME, PROXY_PASSWORD
from core.utils import safe_filename
import yt_dlp
from extractors.youtube_extractor import chunk_file_if_needed, transcribe_segments


def parse_tweet_id_for_video_check(tweet_url: str) -> str:
    """
    Helper to parse the tweet ID from a URL.
    Used by get_twitter_video_transcript for the yt-dlp logic.
    """
    pattern = r"/status/(\d+)"
    match = re.search(pattern, tweet_url)
    return match.group(1) if match else ""


def get_twitter_video_transcript(tweet_url: str) -> str:
    """
    Checks if the tweet contains a video by running yt-dlp in simulation mode.
    If a supported video/audio format is present, downloads and transcribes it.
    Otherwise, returns an empty string.
    Uses a safe filename to avoid errors with too-long file names.
    """
    tweet_id = parse_tweet_id_for_video_check(tweet_url)
    if not tweet_id:
        return ""
    base_name = safe_filename(f"twitter_video_{tweet_id}")
    outtmpl = f"/tmp/{base_name}.%(ext)s"

    # Simulation (metadata only)
    ydl_opts_simulate = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'cookiefile': "../../cookies.txt",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts_simulate) as ydl:
            info = ydl.extract_info(tweet_url, download=False)
        supported_formats = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
        if info.get("ext", "") not in supported_formats:
            formats_list = info.get("formats", [])
            if not any(fmt.get("ext", "") in supported_formats for fmt in formats_list):
                return ""
    except Exception:
        return ""

    # Actual download and transcription if we have a supported format
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outtmpl,
        'quiet': True,
        'no_warnings': True,
        'cookiefile': "../../cookies.txt",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(tweet_url, download=True)
            file_path = ydl.prepare_filename(info)
        if not os.path.exists(file_path):
            return ""
        chunks = chunk_file_if_needed(file_path)
        transcript = transcribe_segments(chunks)
        return transcript
    except Exception:
        return ""


def parse_tweet_id_from_url(tweet_url: str) -> str:
    pattern = r"/status/(\d+)"
    match = re.search(pattern, tweet_url)
    return match.group(1) if match else ""


async def fetch_single_tweet_and_thread(tweet_id: str) -> list[Tweet]:
    proxy = Proxy(host=PROXY_IP, port=PROXY_PORT, proxy_type=PROXY_TYPE_HTTP, username=PROXY_USERNAME, password=PROXY_PASSWORD)
    app = TwitterAsync("session", proxy=proxy)
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
        return flattened_tweets
    except Exception as e:
        print(f"[fetch_tweet_comments] Error fetching comments for tweet {original_tweet.id}: {e}")
        return []


def extract_single_tweet_details(tweet_url: str, fetch_comments: bool = False) -> dict:
    """
    Extract the text from a single tweet or tweet-thread link.
    Returns { 'text': joined_text, 'summary': '', 'authors': [], ... }.
    This function is used by the generate_summary command.
    """
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
            comments_list = loop.run_until_complete(fetch_tweet_comments(root_tweet, pages=2))
    except Exception as e:
        return {
            'text': f"Error fetching tweet ID={tweet_id}: {e}",
            'summary': "",
            'authors': [],
            'publish_date': None,
        }
    finally:
        loop.close()

    # Merge main tweets + optional comment tweets and build a single text
    all_tweets = expanded_tweets[:]
    if fetch_comments and comments_list:
        all_tweets.extend(comments_list)

    sorted_tweets = sorted(all_tweets, key=lambda t: t.created_on or 0)
    snippet_list = []
    authors = set()
    publish_date = None

    for tw in sorted_tweets:
        dt = tw.created_on
        if dt is None:
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

        # Build tweet text the same way as before
        snippet_text = f"@{user_name} ({dt_str}): {tw.text}"
        tweet_url_for_video = (
            f"https://x.com/{user_name}/status/{tw.id}"
            if user_name != "unknown"
            else f"https://x.com/status/{tw.id}"
        )
        transcript = get_twitter_video_transcript(tweet_url_for_video)
        if transcript:
            snippet_text += "\n\nVideo Transcript:\n" + transcript

        snippet_list.append(snippet_text)

    full_text = "\n\n".join(snippet_list)
    return {
        'text': full_text,
        'summary': "",
        'authors': list(authors),
        'publish_date': publish_date,
    }


def build_text_from_tweets(tweets: list[Tweet]) -> str:
    """
    Helper function that sorts a list of tweets and joins their text.
    For each tweet, it also checks if there is an associated video by calling
    get_twitter_video_transcript and appends the transcript.
    """
    if not tweets:
        return ""
    sorted_tweets = sorted(tweets, key=lambda t: t.created_on or 0)
    snippet_list = []
    for tw in sorted_tweets:
        dt_str = "(no date)"
        if tw.created_on:
            dt_utc = tw.created_on.replace(tzinfo=timezone.utc)
            dt_str = dt_utc.isoformat()
        user_name = "unknown"
        if tw.author and tw.author.username:
            user_name = tw.author.username
        # Construct tweet URL using the username if available.
        tweet_url = (
            f"https://x.com/{user_name}/status/{tw.id}"
            if user_name != "unknown"
            else f"https://x.com/status/{tw.id}"
        )
        base_text = f"@{user_name} ({dt_str}): {tw.text}"
        # Check for a video transcript for this tweet.
        transcript = get_twitter_video_transcript(tweet_url)
        if transcript:
            base_text += "\n\nVideo Transcript:\n" + transcript
        snippet_list.append(base_text)
    return "\n\n".join(snippet_list)


def extract_tweet_and_comments_text(tweet_url: str, fetch_comments: bool = False) -> dict:
    """
    Extracts and returns two separate texts from a tweet URL:
      - 'main_text': text from the main tweet (or its thread)
      - 'comments_text': text from the tweet’s comments (if asked for)
    """
    tweet_id = parse_tweet_id_from_url(tweet_url)
    if not tweet_id:
        return {
            'main_text': "Error: Could not parse tweet ID",
            'comments_text': ""
        }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        main_tweets = loop.run_until_complete(fetch_single_tweet_and_thread(tweet_id))
        comments = []
        if fetch_comments and main_tweets:
            root_tweet = main_tweets[0]
            comments = loop.run_until_complete(fetch_tweet_comments(root_tweet, pages=1))
    except Exception as e:
        return {
            'main_text': f"Error fetching tweet ID={tweet_id}: {e}",
            'comments_text': ""
        }
    finally:
        loop.close()

    main_text = build_text_from_tweets(main_tweets)
    comments_text = build_text_from_tweets(comments) if fetch_comments else ""
    return {
        'main_text': main_text,
        'comments_text': comments_text
    }
