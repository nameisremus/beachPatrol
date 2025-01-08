import asyncio
import openai
import re
from datetime import datetime, timedelta, timezone

from tweety import Twitter
from tweety.types.twDataTypes import Tweet, SelfThread

from .digest_extractor import (
    parse_timeframe,
    _replace_large_headings,
    discordify_headings,
    _expand_tweet_and_threads,
    categorize_tweet,
)

from core.core import (
    summarize_transcript,
    get_executive_summary
)
from config import (
    OPENAI_API_KEY,
    TWITTER_USER,
    TWITTER_PASSWORD
)

openai.api_key = OPENAI_API_KEY

async def _fetch_pages_until_cutoff(app: Twitter, username: str, cutoff_dt: datetime) -> list[Tweet | SelfThread]:
    """
    Continually fetch more pages until we encounter a tweet older than cutoff_dt
    or we get no more pages back.
    """
    collected = []
    page = 1

    while True:
        try:
            batch = await app.get_tweets(
                username,
                pages=page,
                replies=False,
                wait_time=15
            )
            if not batch:
                break  # no more tweets

            # Convert to a list if it's not already
            if isinstance(batch, list):
                new_tweets = batch
            else:
                new_tweets = batch.tweets

            if not new_tweets:
                break

            collected.extend(new_tweets)

            # Check if we have any older tweet
            older_found = any(
                t.created_on.replace(tzinfo=timezone.utc) < cutoff_dt
                for t in new_tweets
                if isinstance(t, Tweet)
            )
            if older_found:
                # Stop fetching further pages
                break

            page += 1

            # Optional: put a maximum safety if you like, e.g. if page > 10 then break
        except Exception as e:
            print(f"[fetch_pages_until_cutoff] Error fetching page={page} for {username}: {e}")
            break

    return collected


async def _expand_and_filter_tweets(app: Twitter, all_raw, cutoff_dt: datetime) -> dict:
    """
    Expand & filter the raw Tweets / SelfThreads, building a final list of tweet dicts.
    Returns a dict with merged_tweets, normal_count, retweet_count.
    """
    merged_ids = set()
    final_list = []

    # Deduplicate first
    for tw in all_raw:
        if isinstance(tw, Tweet):
            if tw.id not in merged_ids:
                merged_ids.add(tw.id)
                final_list.append(tw)
        elif isinstance(tw, SelfThread):
            for sub_tweet in tw.tweets:
                if sub_tweet.id not in merged_ids:
                    merged_ids.add(sub_tweet.id)
                    final_list.append(sub_tweet)

    merged_tweets = []
    merged_normal = 0
    merged_retweets = 0

    # Now expand each tweet
    for raw_tweet in final_list:
        try:
            expansion_res = await _expand_tweet_and_threads(app, raw_tweet)
            merged_normal += expansion_res["normal_count"]
            merged_retweets += expansion_res["retweet_count"]

            # Filter out older than cutoff
            for item in expansion_res["expanded_list"]:
                tw = item["tweet_obj"]
                if (tw.created_on.replace(tzinfo=timezone.utc) < cutoff_dt):
                    if item["is_retweet"]:
                        merged_retweets -= 1
                    else:
                        merged_normal -= 1
                else:
                    merged_tweets.append({
                        "id": tw.id,
                        "text": tw.text,
                        "date": tw.created_on.isoformat(),
                        "screen_name": raw_tweet.author if hasattr(raw_tweet, "author") else "",
                        "is_retweet": item["is_retweet"]
                    })
        except Exception as e:
            print(f"[expand_and_filter_tweets] Error expanding tweet for single-user: {e}")
            continue

    # No negatives
    if merged_normal < 0:
        merged_normal = 0
    if merged_retweets < 0:
        merged_retweets = 0

    return {
        "merged_tweets": merged_tweets,
        "normal_count": merged_normal,
        "retweet_count": merged_retweets
    }


def process_twitter_account_summary(username: str, timeframe: str = "1d") -> tuple[str, str]:
    """
    Summarize tweets for a single user within a timeframe, returning (exec_summary, notes).
    We'll also:
      - keep fetching pages until we see older tweets or no more data.
      - build a 'legend' referencing each tweet by [1], [2], etc. 
      - replicate the category logic from your multi-user digest.
    """
    from celery import current_task

    cutoff_dt = datetime.now(timezone.utc) - parse_timeframe(timeframe)
    print(f"[process_twitter_account_summary] username={username}, timeframe={timeframe}, cutoff={cutoff_dt.isoformat()}")

    async def run_async():
        # 1) Create Tweety client
        app = Twitter("session")
        if TWITTER_USER and TWITTER_PASSWORD:
            print(f"[process_twitter_account_summary] Logging in as {TWITTER_USER}...")
            await app.sign_in(TWITTER_USER, TWITTER_PASSWORD)
            print(f"[process_twitter_account_summary] Logged in user: {app.me}")

        # 2) Keep fetching more pages until we see older tweets
        all_raw_tweets = await _fetch_pages_until_cutoff(app, username, cutoff_dt)
        if not all_raw_tweets:
            return ("No tweets found", "No tweets found within timeframe.")

        # 3) Expand + filter out older
        result_dict = await _expand_and_filter_tweets(app, all_raw_tweets, cutoff_dt)
        final_tweets = result_dict["merged_tweets"]
        normal_count = result_dict["normal_count"]
        retweet_count = result_dict["retweet_count"]

        if not final_tweets:
            return ("No tweets found", "No tweets found within timeframe.")

        # Optional: progress
        if current_task:
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "processed": 1,
                    "total": 1,
                    "username": username,
                    "timeframe": timeframe
                }
            )

        # 4) Build snippet list + categorize for the legend
        cat_map = {}
        root_ids = []
        tweet_idx = 1
        final_snippets = []

        for tw in final_tweets:
            snippet = f"@{username} ({tw['date']}): {tw['text']}"
            # In your single-user approach, we are not filtering by relevancy 
            # but if you do want to skip irrelevant tweets, 
            # you can call: if not check_tweet_relevance(snippet): continue

            # categorize
            tweet_category = categorize_tweet(snippet)
            if tweet_category not in cat_map:
                cat_map[tweet_category] = []
            cat_map[tweet_category].append(tweet_idx)

            final_snippets.append(snippet)
            root_ids.append(tw["id"])
            tweet_idx += 1

        if not final_snippets:
            return ("No relevant tweets found.", "No tweets found or none were relevant within timeframe.")

        # 5) Summarize all 
        joined_text = "\n\n".join(final_snippets)
        # We can reuse "article" or define a new media_type. We'll reuse "article".
        summary = summarize_transcript(joined_text, media_type="article")
        summary = _replace_large_headings(summary)
        summary = discordify_headings(summary)

        exec_summary = get_executive_summary(summary, media_type="article")
        exec_summary = _replace_large_headings(exec_summary)
        exec_summary = discordify_headings(exec_summary)

        # 5b) Build legend: referencing each tweet by [1], [2], etc.
        dedup_ids = list(dict.fromkeys(root_ids))
        link_map = {}
        for i, rid in enumerate(dedup_ids, start=1):
            link_map[i] = f"[{i}](https://x.com/{username}/status/{rid})"

        legend_lines = []
        if cat_map:
            legend_lines.append("Analyzed tweets, retweets, and threads by category:")
            for category, idx_list in cat_map.items():
                link_strs = []
                for idx in idx_list:
                    if idx in link_map:
                        link_strs.append(link_map[idx])
                if link_strs:
                    joined_links = ", ".join(link_strs)
                    legend_lines.append(f"- {category}: {joined_links}")
        else:
            legend_lines.append("**[+] No tweets analyzed.**")

        legend_text = "\n".join(legend_lines)
        exsumm_legend = f"{exec_summary}\n\n**:information_source:** {legend_text}"

        # 6) Build normal vs retweets line
        if normal_count == 0 and retweet_count == 0:
            line_label = "No tweets"
        elif normal_count > 0 and retweet_count > 0:
            line_label = f"{normal_count} tweets and {retweet_count} retweets"
        elif normal_count == 0:
            line_label = f"{retweet_count} retweets"
        elif retweet_count == 0:
            line_label = f"{normal_count} tweets"
        else:
            line_label = f"{normal_count} tweets and {retweet_count} retweets"

        # 7) Build final notes
        # We keep it similar to the multi-user approach
        # Example: Title line might be "### 1. [@username] - X tweets" etc.
        final_notes = f"# Single-User Twitter Summary (Last {timeframe})\n\n"
        final_notes += f"### [@{username}](https://x.com/{username}) - {line_label} within the timeframe\n\n"
        final_notes += summary

        # exec_summary includes the legend 
        final_exec = exsumm_legend
        final_exec += f"\n\n**Processed {normal_count} normal tweets and {retweet_count} retweets within the timeframe.**"

        return (final_exec.strip(), final_notes.strip())

    # 8) Run async
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        exec_sum, notes = loop.run_until_complete(run_async())
        return (exec_sum, notes)
    except Exception as e:
        print(f"[process_twitter_account_summary] Error: {e}")
        return ("Error processing tweets", "Error processing tweets")
    finally:
        loop.close()