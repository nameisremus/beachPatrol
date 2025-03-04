from commands.generate_summary.generate_summary_router import get_summary_router
from core.core import summarize_transcript, get_executive_summary, do_custom_prompt, get_valid_model
from extractors.twitter.single_tweet_extractor import extract_tweet_and_comments_text

def process_tweet_summary(url: str, parse_comments: bool = False, model=None, prompt=None) -> dict:
    """
    Summarizes a tweet or tweet thread URL.
    If parse_comments is False, it uses get_summary_router (existing logic).
    Otherwise, separate summaries for main tweet vs. comments.

    If `prompt` is given, we do a single custom prompt that uses all extracted text 
    and sets exec_sum=summary to that result, overriding normal logic.

    NOTE: Video checking and transcription is now handled in single_tweet_extractor.py,
    so that logic no longer appears here. The text returned by extract_tweet_and_comments_text
    may already include any video transcripts.
    """
    if not parse_comments:
        # If no custom prompt:
        if not prompt:
            result = get_summary_router(url)
            result['tweet_url'] = url
            result['parse_comments'] = False

            # If a model override is provided without prompt, we re-run summarization
            if model:
                chosen_model = get_valid_model(model)
                # forcibly re-summarize
                full_text = result.get('article_details', {}).get('text', '')
                new_summary = summarize_transcript(full_text, media_type="tweet_or_thread")
                new_exec = get_executive_summary(new_summary, media_type="tweet_or_thread")
                result['exec_sum'] = new_exec
                result['summary'] = new_summary
            return result

        # If prompt is provided, we do a single custom approach
        text_only = extract_tweet_and_comments_text(url, fetch_comments=False)
        main_text = text_only.get("main_text", "")
        chosen_model = get_valid_model(model)
        combined_res = do_custom_prompt(main_text, prompt, chosen_model)
        return {
            "tweet_url": url,
            "exec_sum": combined_res,
            "summary": combined_res,
            "parse_comments": False
        }

    else:
        # parse_comments == True
        # If prompt is given, do a single custom call for main + comments
        data = extract_tweet_and_comments_text(url, fetch_comments=True)
        main_text = data.get("main_text", "")
        comments_text = data.get("comments_text", "")
        combined_text = main_text + "\n\n--- COMMENTS ---\n\n" + comments_text

        if prompt:
            chosen_model = get_valid_model(model)
            single_res = do_custom_prompt(combined_text, prompt, chosen_model)
            return {
                "tweet_url": url,
                "exec_sum": single_res,
                "summary": "",
                "parse_comments": True
            }
        else:
            # existing multi-step logic
            summary_main = summarize_transcript(main_text, media_type="tweet_or_thread")
            exec_main = get_executive_summary(summary_main, media_type="tweet_or_thread")
            summary_comments = summarize_transcript(comments_text, media_type="tweet_comments")
            exec_comments = get_executive_summary(summary_comments, media_type="tweet_comments")

            combined_exec = (
                "**Executive Summary (Main Tweet):**\n" + exec_main +
                "\n\n**Executive Summary (Comments):**\n" + (exec_comments if exec_comments else "No comments.")
            )
            combined_summary = (
                "**Full Summary (Main Tweet):**\n" + summary_main +
                "\n\n**Full Summary (Comments):**\n" + (summary_comments if summary_comments else "No comments.")
            )

            # If a model override (w/o prompt), we could re-run if needed
            if model:
                chosen_model = get_valid_model(model)
                # re-run each piece:
                summary_main2 = summarize_transcript(main_text, media_type="tweet_or_thread")
                exec_main2 = get_executive_summary(summary_main2, media_type="tweet_or_thread")
                summary_comments2 = summarize_transcript(comments_text, media_type="tweet_comments")
                exec_comments2 = get_executive_summary(summary_comments2, media_type="tweet_comments")
                combined_exec = (
                    "**Executive Summary (Main Tweet):**\n" + exec_main2 +
                    "\n\n**Executive Summary (Comments):**\n" + (exec_comments2 if exec_comments2 else "No comments.")
                )
                combined_summary = (
                    "**Full Summary (Main Tweet):**\n" + summary_main2 +
                    "\n\n**Full Summary (Comments):**\n" + (summary_comments2 if summary_comments2 else "No comments.")
                )

            return {
                "tweet_url": url,
                "exec_sum": combined_exec,
                "summary": combined_summary,
                "parse_comments": True
            }
