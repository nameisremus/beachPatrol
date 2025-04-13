from .generate_summary_router import get_summary_router
import logging

logger = logging.getLogger(__name__)


def save_summary(article_details: dict, save_path: str) -> bool:
    """
    Saves the summary details to a file, which the user can download
    """
    logger.info(
        "Saving summary to file",
        extra={
            "article_url": article_details.get("article_url"),
            "save_path": save_path,
        }
    )
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("Article URL: {}\n".format(article_details.get('article_url')))
            f.write("\n--- Executive Summary ---\n")
            f.write(article_details.get('exec_sum', '') + "\n")
            f.write("\n--- Full Summary ---\n")
            f.write(article_details.get('summary', '') + "\n")
        return True
    except Exception as e:
        logger.error(
            "Error saving summary",
            extra={
                "article_url": article_details.get("article_url"),
                "save_path": save_path,
            },
            exc_info=True
        )
        return False


def process_get_summary(
    url: str,
    save: bool = False,
    save_path: str = "summary.txt",
    interface: str = "discord",
    model: str = None,
    prompt: str = None
) -> dict:
    """
    Handler for the get_summary command.

    If `prompt` is provided, we skip the default multi-step logic
    and do a single call with the user prompt + the extracted article text.
    Otherwise, proceed with the normal summarize + get_executive_summary pipeline.
    """
    logger.info(
        "Processing get_summary request",
        extra={
            "url": url,
            "save": save,
            "save_path": save_path,
            "interface": interface,
            "model": model,
            "prompt": prompt,
        }
    )

    from core.core import do_custom_prompt, summarize_transcript, get_executive_summary, get_valid_model
    # Extract raw article details first
    raw_result = get_summary_router(url)

    final_exec = raw_result.get('exec_sum', '')
    final_summary = raw_result.get('summary', '')

    if prompt:
        # Single-call approach
        full_text = raw_result.get('article_details', {}).get('text', '')
        # run custom prompt
        chosen_model = get_valid_model(model)
        combined_result = do_custom_prompt(full_text, prompt, chosen_model)
        final_exec = combined_result
        final_summary = ""
    else:
        # normal approach uses existing logic
        if model:
            # if user specified a model, we forcibly re-summarize with that model
            chosen_model = get_valid_model(model)
            # We'll do a forced re-summarization
            article_text = raw_result.get('article_details', {}).get('text', '')
            # first do normal summary
            new_summary = summarize_transcript(article_text, media_type="article")
            # then do exec summary
            new_exec = get_executive_summary(new_summary, media_type="article")
            final_summary = new_summary
            final_exec = new_exec

    result = {
        "article_url": url,
        "exec_sum": final_exec,
        "summary": final_summary,
        "article_details": raw_result
    }
    if save:
        saved_ok = save_summary(result, save_path)
        result["saved"] = saved_ok

    # Format for interface, by default the interface is discord and TG format gets processed after it
    if interface.lower() == "discord":
        formatted = {
            "title": "Article Summary",
            "description": final_exec,
            "notes": final_summary
        }
    elif interface.lower() == "telegram":
        formatted = {
            "text": (
                f"*Article Summary:*\n\n"
                f"{final_exec}\n\n"
                f"*Full Summary:*\n{final_summary}"
            )
        }
    else:
        formatted = result

    result["formatted"] = formatted
    return result
