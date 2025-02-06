from commands.generate_summary.generate_summary_router import get_summary_router
from core.core import summarize_transcript, get_executive_summary
from extractors.twitter.single_tweet_extractor import extract_single_tweet_details

def process_tweet_summary(url: str, parse_comments: bool = False) -> dict:
    """
    Summarizes a tweet or tweet thread URL, optionally including comments.
    """
    single_tweet_data = extract_single_tweet_details(url, fetch_comments=parse_comments)

    # The text from the tweet (and optionally comments)
    tweet_text = single_tweet_data.get("text", "")
    # Summarize
    summary = summarize_transcript(tweet_text, media_type="article")
    exec_summary = get_executive_summary(summary, media_type="article")

    # Build the final result dict:
    result = {
        "tweet_url": url,
        "exec_sum": exec_summary,
        "notes": summary,
        "parse_comments": parse_comments,
    }

    # Optional: build some formatted structure
    if parse_comments:
        formatted = {
            "title": "Tweet Summary (with Comments)",
            "description": exec_summary,
            "notes": summary + "\n\n[Comments included]"
        }
    else:
        formatted = {
            "title": "Tweet Summary",
            "description": exec_summary,
            "notes": summary
        }

    result["formatted"] = formatted
    return result
