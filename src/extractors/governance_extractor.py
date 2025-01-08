import requests
import html
import re
from datetime import datetime, timedelta, timezone

from config import GOVERNANCE_FORUMS_MAPPING, OPENAI_MODEL
from core.core import summarize_transcript, get_executive_summary, check_topic_relevance, categorize_governance_topic

def remove_html_tags(text):
    return re.sub('<[^<]+?>', '', text)

def parse_timeframe(timeframe_str: str) -> timedelta:
    """
    Parses strings like '1d', '7d', '30d' into a datetime.timedelta.
    If unrecognized, defaults to 1 day.
    """
    if timeframe_str.endswith('d'):
        try:
            days = int(timeframe_str[:-1])
            return timedelta(days=days)
        except ValueError:
            pass
    return timedelta(days=1)

def fetch_forum_topics(forum_url):
    """
    Retrieve /latest.json from Discourse forums to get recent topics.
    Returns a list of dicts or an empty list on errors.
    """
    try:
        print(f"Fetching topics from {forum_url}...")
        resp = requests.get(f"{forum_url}/latest.json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        topics = data.get('topic_list', {}).get('topics', [])
        print(f"Found {len(topics)} topics in {forum_url}")
        return topics
    except Exception as e:
        print(f"Error fetching from {forum_url}: {e}")
        return []

def get_topic_details(forum_url, topic_id):
    """
    Fetch the first post's 'cooked' HTML from the topic_id,
    strip HTML tags, and return plain text.
    """
    try:
        print(f"Fetching details for topic {topic_id} from {forum_url}")
        resp = requests.get(f"{forum_url}/t/{topic_id}.json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get('post_stream', {}).get('posts', [])
        if posts and posts[0].get('cooked'):
            cooked_html = posts[0]['cooked']
            return html.unescape(remove_html_tags(cooked_html))
    except Exception as e:
        print(f"Error fetching topic details from {forum_url}: {e}")
    return ""

def process_governance_forum(
    timeframe: str = "1d",
    only_relevant: bool = True
):
    """
    1) Determine cutoff from timeframe
    2) For each forum in GOVERNANCE_FORUMS_MAPPING, fetch recent topics
       - For each topic, fetch the full post content
       - Combine title + excerpt + body
       - If relevancy filter is True, run check_topic_relevance
       - If relevant, categorize the topic, then summarize
    3) Produce combined notes & executive summary grouped by category
    """
    print("Starting governance forum processing...")

    # Convert timeframe (e.g. '7d') to a timedelta
    timeframe_delta = parse_timeframe(timeframe)
    cutoff = datetime.now(timezone.utc) - timeframe_delta
    print(f"Timeframe={timeframe}, onlyRelevant={only_relevant}")
    print(f"Using timeframe cutoff of {timeframe_delta} -> {cutoff.isoformat()}")

    # We'll accumulate all relevant topics here
    relevant_topics = []

    # Iterate each forum in the mapping
    for forum_url, forum_name in GOVERNANCE_FORUMS_MAPPING.items():
        print(f"Processing forum: {forum_url} ({forum_name})")
        topics = fetch_forum_topics(forum_url)
        processed_count = 0
        relevant_count = 0

        for t in topics:
            created_at = t.get('created_at')
            if not created_at:
                continue

            # Convert 'created_at' to offset-aware datetime
            try:
                naive_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                created_dt = naive_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Skip older topics
            if created_dt < cutoff:
                continue

            title = t.get('title', '')
            excerpt = t.get('excerpt', '')
            topic_id = t.get('id')

            if not title or not topic_id:
                continue

            processed_count += 1
            print(f"Checking topic '{title}' in {forum_name}")

            # Fetch full text for summarization + category
            full_content = get_topic_details(forum_url, topic_id)
            entire_text_for_relevance = f"{title}\n\n{excerpt}\n\n{full_content}"

            # If only_relevant is True, check if relevant to Lido
            if only_relevant:
                if not check_topic_relevance(entire_text_for_relevance):
                    print(f"Topic '{title}' in {forum_name} not relevant.")
                    continue

            # Now we categorize the topic
            category = categorize_governance_topic(entire_text_for_relevance)

            # Summaries
            summary = summarize_transcript(
                entire_text_for_relevance,
                media_type="governance_forum_summary"
            )
            exec_summary = get_executive_summary(summary, media_type="governance_forum")

            print(f"Topic '{title}' in {forum_name} is included and summarized.")
            relevant_count += 1

            # Build direct link
            thread_link = f"{forum_url}/t/{topic_id}"

            # Adjust headings
            summary = summary.replace("##", "###")
            # Second pass: anything >= #### => ###
            summary = re.sub(r'^(#{4,})(\s+)', r'###\2', summary, flags=re.MULTILINE)
            # Discordify formatting
            summary = re.sub(r'(?m)^###\s+(.*)', r'**\1**', summary, flags=re.MULTILINE)

            relevant_topics.append({
                'forum_name': forum_name,
                'thread_link': thread_link,
                'title': title,
                'summary': summary,
                'exec_summary': exec_summary,
                'category': category
            })

        print(f"Finished processing {forum_name}: Processed={processed_count}, Included={relevant_count}.")

    # Group final results
    if not relevant_topics:
        combined_notes = "No new relevant governance topics found in the given timeframe."
        combined_exec = "No updates."
    else:
        # Known categories, in desired order
        CATEGORY_ORDER = [
            "Lido Related Updates 💧",
            "Competitor Updates 🥊",
            "Lending Markets 🏦",
            "Layer2s 🔗",
            "DEXes 💱",
            "Grants and Funding 💸",
            "Misc. 🌀"
        ]
        # Build a dict category -> list of topics
        grouped = {cat: [] for cat in CATEGORY_ORDER}

        # Sort each topic into the correct bucket
        for rt in relevant_topics:
            cat = rt["category"]
            # If GPT returned something unexpected, default to Misc.
            if cat not in grouped:
                cat = "Misc. 🌀"
            grouped[cat].append(rt)

        # Build final combined notes & exec
        combined_notes = f"## Ecosystem and Governance Updates (Last {timeframe}), model={OPENAI_MODEL}\n\n"
        combined_exec = f"**Ecosystem and Governance Updates past {timeframe}**\n\n"

        for cat in CATEGORY_ORDER:
            cat_items = grouped[cat]
            if not cat_items:
                continue  # skip empty category

            # Add category heading
            combined_notes += f"### {cat}\n\n"
            combined_exec += f"### {cat}\n\n"

            for i, rt in enumerate(cat_items, start=1):
                combined_notes += (
                    f"{i}.) [{rt['title']} - {rt['forum_name']}](<{rt['thread_link']}>)\n"
                    f"   {rt['summary']}\n\n"
                )
                combined_exec += (
                    f"{i}. [{rt['title']} - {rt['forum_name']}](<{rt['thread_link']}>) - {rt['exec_summary']}\n\n"
                )

    print("Governance forum processing completed.")
    return {
        'exec_sum': combined_exec.strip(),
        'notes': combined_notes.strip()
    }