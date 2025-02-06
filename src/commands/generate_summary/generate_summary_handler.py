from .generate_summary_router import get_summary_router

def save_summary(article_details: dict, save_path: str) -> bool:
    """
    Saves the summary details to a file.
    This implementation writes the URL, executive summary, and 
    full summary to a text file which the user can download
    """
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("Article URL: {}\n".format(article_details.get('article_url')))
            f.write("\n--- Executive Summary ---\n")
            f.write(article_details.get('exec_sum', '') + "\n")
            f.write("\n--- Full Summary ---\n")
            f.write(article_details.get('summary', '') + "\n")
        return True
    except Exception as e:
        print(f"Error saving summary: {e}")
        return False

def process_get_summary(url: str, save: bool = False, save_path: str = "summary.txt", interface: str = "discord") -> dict:
    """
    Handler for the get_summary command.
    
    Parameters:
      - url: the URL of the article to summarize.
      - save: whether to save the summary to a file.
      - save_path: the file path where the summary should be saved if save is True.
      - interface: a string indicating which interface is requesting the summary 
                   (e.g. "discord" or "telegram") so that the formatting can be adapted.
    
    Returns a dictionary with:
      - 'article_url'
      - 'exec_sum'
      - 'summary'
      - 'formatted': the output formatted appropriately for the interface.
      - Any additional keys as needed.
    """
    # Get the raw summary details from the router.
    result = get_summary_router(url)
    
    # Optionally save the summary.
    if save:
        saved = save_summary(result, save_path)
        result['saved'] = saved
    
    # Format the output based on the interface.
    if interface.lower() == "discord":
        # For Discord, we use an embed
        formatted = {
            "title": "Article Summary",
            "description": result.get("exec_sum", ""),
            "notes": result.get("summary", "")
        }
    elif interface.lower() == "telegram":
        # For Telegram, a plain-text format is common.
        formatted = {
            "text": (
                f"*Article Summary:*\n\n"
                f"{result.get('exec_sum', '')}\n\n"
                f"*Full Summary:*\n{result.get('summary', '')}"
            )
        }
    else:
        # Default to returning the raw result.
        formatted = result
    
    result['formatted'] = formatted
    return result
