import asyncio
import openai
import re
from datetime import datetime, timedelta, timezone

from tweety import Twitter
from tweety.types.twDataTypes import Tweet, SelfThread

from core.core import (
    summarize_transcript,
    get_executive_summary,
    check_tweet_relevance,
    categorize_tweet
)
from config import (
    OPENAI_API_KEY,
    TWITTER_ACCOUNTS,
    TWITTER_ACCOUNTS_DICT,
    TWITTER_USER,
    TWITTER_PASSWORD
)

openai.api_key = OPENAI_API_KEY

def parse_timeframe(timeframe_str: str) -> timedelta:
    """
    Parses strings like '1d', '7d', '30d' into a timedelta.
    Defaults to 1 day if unrecognized.
    """
    if timeframe_str.endswith('d'):
        try:
            days = int(timeframe_str[:-1])
            return timedelta(days=days)
        except ValueError:
            pass
    return timedelta(days=1)

async def _expand_tweet_and_threads(app: Twitter, tweet: Tweet | SelfThread) -> dict:
    """
    Given a Tweet or SelfThread object, expand all threads or retweets so we
    get a complete list of tweets that belong together.

    Returns:
      {
        "expanded_list": [ { "tweet_obj": <Tweet>, "is_retweet": bool }, ... ],
        "retweet_count": <int>,
        "normal_count": <int>
      }
    """
    expanded = []
    normal_ct = 0
    retweet_ct = 0

    async def make_tweet_dict(tw: Tweet, force_retweet=False) -> dict:
        nonlocal normal_ct, retweet_ct
        if force_retweet:
            retweet_ct += 1
        else:
            if tw.is_retweet:
                retweet_ct += 1
            else:
                normal_ct += 1
        return {
            "tweet_obj": tw,
            "is_retweet": force_retweet or tw.is_retweet
        }

    if isinstance(tweet, SelfThread):
        # The user started a self-thread
        await tweet.expand()
        for tw in tweet.tweets:
            expanded.append(await make_tweet_dict(tw))
    elif isinstance(tweet, Tweet):
        if tweet.is_retweet and tweet.retweeted_tweet:
            # Expand the original tweet if it's a retweet
            original_tweet = tweet.retweeted_tweet
            detailed = await app.tweet_detail(original_tweet.id)
            if isinstance(detailed, SelfThread):
                await detailed.expand()
                for tw in detailed.tweets:
                    expanded.append(await make_tweet_dict(tw, force_retweet=True))
            elif detailed.threads:
                for tw in detailed.threads:
                    expanded.append(await make_tweet_dict(tw, force_retweet=True))
            else:
                expanded.append(await make_tweet_dict(detailed, force_retweet=True))
        else:
            # Normal standalone tweet
            expanded.append(await make_tweet_dict(tweet))

    return {
        "expanded_list": expanded,
        "retweet_count": retweet_ct,
        "normal_count": normal_ct
    }

async def _fetch_tweets_simple(app: Twitter, screen_name: str, pages: int) -> list[Tweet]:
    """
    Fetch tweets from Tweety using a specified `pages` argument (1 => ~21 tweets, 2 => ~41, etc.)
    Excluding replies by default. Returns the raw Tweet objects.
    """
    try:
        return await app.get_tweets(
            screen_name,
            pages=pages,
            replies=False,
            wait_time=15
        )
    except Exception as e:
        print(f"[_fetch_tweets_simple] Error fetching {pages} pages for {screen_name}: {e}")
        return []

async def fetch_user_tweets_with_threads(
    app: Twitter,
    screen_name: str,
    cutoff_dt: datetime
) -> dict:
    """
    1) Fetch 1 "page" ~21 tweets.
       - If we see *any* tweet older than cutoff_dt in that batch, we skip the second fetch.
       - Otherwise, fetch again with pages=2 (~41 tweets), merge them (deduplicate).
    2) Expand threads/retweets, filter out older than cutoff_dt, tally normal vs retweets.

    Returns:
      {
        "merged_tweets": [
          {
            "id": <int>,
            "text": <str>,
            "date": <str>,
            "screen_name": <str>,
            "is_retweet": bool
          },
          ...
        ],
        "normal_count": <int>,
        "retweet_count": <int>
      }
    """
    # Step 1: fetch 1 page
    first_batch = await _fetch_tweets_simple(app, screen_name, pages=1)
    if not first_batch:
        return {"merged_tweets": [], "normal_count": 0, "retweet_count": 0}

    # Check if any tweet is older than cutoff => older_found
    older_found = any(
        isinstance(t, Tweet) and t.created_on.replace(tzinfo=timezone.utc) < cutoff_dt
        for t in first_batch
    )

    # If NOT older_found => do a second fetch with pages=2
    if not older_found:
        second_batch = await _fetch_tweets_simple(app, screen_name, pages=2)
    else:
        second_batch = []

    first_batch_tweets = first_batch if isinstance(first_batch, list) else first_batch.tweets
    second_batch_tweets = second_batch if isinstance(second_batch, list) else second_batch.tweets

    # Merge and deduplicate the lists
    combined_ids = set()
    combined_list = []

    # Handle both Tweet and SelfThread objects
    for tw in (first_batch_tweets + second_batch_tweets):
        if isinstance(tw, Tweet):
            # If it's a Tweet, just check its ID
            if tw.id not in combined_ids:
                combined_ids.add(tw.id)
                combined_list.append(tw)
        elif isinstance(tw, SelfThread):
            # If it's a SelfThread, expand the thread and add the individual tweets to combined_list
            for sub_tweet in tw.tweets:
                if sub_tweet.id not in combined_ids:
                    combined_ids.add(sub_tweet.id)
                    combined_list.append(sub_tweet)

    # Now proceed with the expansion and filtering logic as before
    merged_tweets = []
    merged_normal = 0
    merged_retweets = 0

    for raw_tweet in combined_list:
        try:
            expansion_res = await _expand_tweet_and_threads(app, raw_tweet)
            merged_normal += expansion_res["normal_count"]
            merged_retweets += expansion_res["retweet_count"]

            # Filter out older than cutoff
            for item in expansion_res["expanded_list"]:
                tw = item["tweet_obj"]
                if isinstance(tw, Tweet) and tw.created_on.replace(tzinfo=timezone.utc) < cutoff_dt:
                    if item["is_retweet"]:
                        merged_retweets -= 1
                    else:
                        merged_normal -= 1
                else:
                    merged_tweets.append({
                        "id": tw.id,
                        "text": tw.text,
                        "date": tw.created_on.isoformat(),
                        "screen_name": screen_name,
                        "is_retweet": item["is_retweet"]
                    })
        except Exception as e:
            print(f"[fetch_user_tweets_with_threads] Error expanding tweet {raw_tweet.id} for {screen_name}: {e}")
            continue

    if merged_normal < 0:
        merged_normal = 0
    if merged_retweets < 0:
        merged_retweets = 0

    return {
        "merged_tweets": merged_tweets,
        "normal_count": merged_normal,
        "retweet_count": merged_retweets
    }


def _replace_large_headings(text: str) -> str:
    """
    Replace top-level # or ## with ### in any Markdown headings.
    Ensures we never get huge headings in Discord.
    Then final pass: if we see #### or more, also turn them into ###.
    """
    text = re.sub(r'^(#{1,2})(\s+)', r'###\2', text, flags=re.MULTILINE)
    text = re.sub(r'^(#{4,})(\s+)', r'###\2', text, flags=re.MULTILINE)
    return text

def discordify_headings(text: str) -> str:
    """
    Convert lines starting with '### ' into '**Heading**'
    so they stand out more in Discord (optional convenience).
    """
    return re.sub(r'(?m)^###\s+(.*)', r'**\1**', text)

def process_twitter_digest(timeframe: str = "1d", only_relevant: bool = True):
    """
    1. Create an authenticated Tweety client (Twitter).
    2. For each user in TWITTER_ACCOUNTS, fetch 1 page => if no older tweets, also fetch 2 pages => deduplicate => expand => filter.
    3. If only_relevant == True, filter out tweets not relevant to Lido.
    4. Summarize each user's combined tweet text.
    5. Use categorize_tweet(...) to group them by category in the final legend.
    6. Enhance the title line with user display_name & organization (if present).
    7. Return final exec summary + notes with discordified formatting.
    """
    # For progress updates
    from celery import current_task

    cutoff_dt = datetime.now(timezone.utc) - parse_timeframe(timeframe)
    print(f"[process_twitter_digest] timeframe={timeframe}, onlyRelevancy={only_relevant}, cutoff={cutoff_dt.isoformat()}")

    async def run_async():
        # 1) create Tweety client
        app = Twitter("session")
        if TWITTER_USER and TWITTER_PASSWORD:
            print(f"[process_twitter_digest] Logging in as {TWITTER_USER}...")
            await app.sign_in(TWITTER_USER, TWITTER_PASSWORD)
            print(f"[process_twitter_digest] Logged in user: {app.me}")

        summaries = []

        total_accounts = len(TWITTER_ACCOUNTS)
        processed_accounts = 0

        # 2) iterate each configured account
        for acct in TWITTER_ACCOUNTS:
            # Increase processed counter
            processed_accounts += 1

            # If we're running inside a Celery task, update progress
            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={
                        "processed": processed_accounts,
                        "total": total_accounts,
                        "timeframe": timeframe,
                        "only_relevant": only_relevant
                    }
                )

            # manual delay between accounts fetching to avoid hitting rate limits
            await asyncio.sleep(30)

            # handle dict or string from config
            if isinstance(acct, dict):
                username_raw = acct.get("username", "")
            else:
                username_raw = str(acct)

            username = username_raw.strip().lower()
            if not username:
                continue

            print(f"[process_twitter_digest] fetching tweets for @{username}")
            # Perform the custom "two-step fetch" approach
            result_dict = await fetch_user_tweets_with_threads(app, username, cutoff_dt)
            all_tweets = result_dict["merged_tweets"]
            normal_count = result_dict["normal_count"]
            retweet_count = result_dict["retweet_count"]

            if not all_tweets:
                continue

            # 3) filter by relevancy + categorize each tweet
            final_tweet_blocks = []
            root_ids = []
            cat_map = {}
            tweet_idx = 1

            for tw in all_tweets:
                snippet = f"@{username} ({tw['date']}): {tw['text']}"
                if only_relevant:
                    if not check_tweet_relevance(snippet):
                        continue

                # categorize the tweet
                tweet_category = categorize_tweet(snippet)
                if tweet_category not in cat_map:
                    cat_map[tweet_category] = []
                cat_map[tweet_category].append(tweet_idx)

                final_tweet_blocks.append(snippet)
                root_ids.append(tw["id"])
                tweet_idx += 1

            if not final_tweet_blocks:
                # means none relevant
                continue

            # 4) Summarize
            combined_text = "\n\n".join(final_tweet_blocks)
            summary = summarize_transcript(combined_text, media_type="twitter_digest")
            summary = _replace_large_headings(summary)
            summary = discordify_headings(summary)

            exec_summary = get_executive_summary(summary, media_type="twitter_digest")
            exec_summary = _replace_large_headings(exec_summary)
            exec_summary = discordify_headings(exec_summary)

            # 5) Build legend with categories => links
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
                        # map idx to link
                        if idx in link_map:
                            link_strs.append(link_map[idx])
                    if link_strs:
                        joined_links = ", ".join(link_strs)
                        legend_lines.append(f"- {category}: {joined_links}")
            else:
                legend_lines.append("**[+] No tweets analyzed.**")

            legend_text = "\n".join(legend_lines)
            exsumm_legend = f"{exec_summary}\n\n**:information_source:** {legend_text}"

            # 6) Build the line label for how many normal vs retweets
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

            # 7) Enhance top line with user display_name & organization
            from config import TWITTER_ACCOUNTS_DICT
            display_name = ""
            organization = ""
            if username in TWITTER_ACCOUNTS_DICT:
                display_name = TWITTER_ACCOUNTS_DICT[username].get("display_name", "")
                organization = TWITTER_ACCOUNTS_DICT[username].get("organization", "")

            link_to_user = f"https://x.com/{username}"
            if display_name or organization:
                parts = []
                if display_name:
                    parts.append(display_name)
                parts.append(f"(@{username})")
                if organization:
                    parts.append(organization)
                combined_identity = ", ".join(parts)
                title_line = (
                    f"### {{index}}. [{combined_identity}]({link_to_user}) - {line_label} within the timeframe\n\n"
                )
            else:
                title_line = (
                    f"### {{index}}. [@{username}]({link_to_user}) - {line_label} within the timeframe\n\n"
                )

            summaries.append((title_line, summary, exsumm_legend))

        # If no final items, return no updates
        if not summaries:
            return ("No relevant tweets found.", "No tweets found or none were relevant within timeframe.")

        # Build final outputs
        final_notes = f"# Twitter Digest (Last {timeframe})\n"
        final_exec = f"**Twitter Digest (Last {timeframe})**\n\n"

        for i, (title_line, summ, exsumm_legend) in enumerate(summaries, start=1):
            rendered_title = title_line.format(index=i)
            final_notes += f"{rendered_title}{summ}\n\n"
            final_exec += f"{rendered_title}{exsumm_legend}\n\n"

        return (final_exec.strip(), final_notes.strip())

    # Run async logic
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        exec_sum, notes = loop.run_until_complete(run_async())
        return (exec_sum, notes)
    except Exception as e:
        print(f"[process_twitter_digest] Error: {e}")
        return ("Error processing tweets", "Error processing tweets")
    finally:
        loop.close()