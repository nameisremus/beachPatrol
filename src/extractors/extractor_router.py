from .generic_extractor import extract_generic_article_details
from .pdf_extractor import extract_pdf_details
from .github_extractor import extract_github_advisory_details
from .twitter.single_tweet_extractor import extract_single_tweet_details
from core.core import summarize_transcript, get_executive_summary



def extract_article_details(url: str) -> dict:
    """
    Routes the fetching of the URL contents for summarization to the correct 
    extractor. If the source is unidentified, then request is routed to the generic extractor
    """
    url_lower = url.lower()
    # If URL is a GitHub advisory page (contains both "github" and "advisories")
    if "github" in url_lower and "advisories" in url_lower:
        return extract_github_advisory_details(url)
    # If URL ends with .pdf or contains "arxiv", use PDF extractor
    elif url_lower.endswith(".pdf") or "arxiv" in url_lower:
        return extract_pdf_details(url)
    elif ("twitter.com/" in url_lower or "x.com/" in url_lower) and "/status/" in url_lower:
    # handle single tweet or thread
        return extract_single_tweet_details(url)
    else:
        return extract_generic_article_details(url)

def process_article(url: str, summarizer_func) -> dict:
    """
    Process the article: extract details then summarize.
    The summarizer_func that accepts the full text and returns a summary.
    """
    details = extract_article_details(url)
    full_text = details.get('text', '')
    # Uses the core.summarize_transcript function via summarizer_func
    summary = summarize_transcript(full_text, media_type="article")
    exec_summary = summarizer_func(summary)
    exec_summary = get_executive_summary(summary, media_type="article")
    details.update({
        'summary': summary,
        'exec_sum': exec_summary
    })
    return details
