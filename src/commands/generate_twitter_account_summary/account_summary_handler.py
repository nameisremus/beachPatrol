from extractors.twitter.account_extractor import process_twitter_account_summary

def process_account_summary(username: str, timeframe: str = "1d", model=None, prompt=None) -> tuple[str, str]:
    """
    Processes a Twitter account summary command.
    If prompt is provided, do a single custom call, otherwise two-step calls
    """
    return process_twitter_account_summary(username, timeframe, model=model, prompt=prompt)
