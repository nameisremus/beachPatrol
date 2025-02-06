from extractors.twitter.account_extractor import process_twitter_account_summary

def process_account_summary(username: str, timeframe: str = "1d") -> tuple[str, str]:
    """
    Processes a Twitter account summary command.
    Returns a tuple: (exec_summary, notes)
    """
    return process_twitter_account_summary(username, timeframe)
