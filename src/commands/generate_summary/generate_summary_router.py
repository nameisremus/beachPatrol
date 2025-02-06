from extractors.extractor_router import process_article
from core.core import summarize_transcript

def get_summary_router(url: str) -> dict:
    """
    Processes a get_summary command
    Uses the shared extractor router (which selects the appropriate extractor)
    and then runs the summarization.
    
    Returns a dictionary with:
      - 'article_url'
      - 'exec_sum': the executive summary
      - 'summary': the main summary
      - 'article_details': all extracted details
    """
    # Pass summarizer (summarize_transcript) to the process_article method.
    result = process_article(url, summarize_transcript)
    return {
        'article_url': url,
        'exec_sum': result.get('exec_sum', ''),
        'summary': result.get('summary', ''),
        'article_details': result
    }