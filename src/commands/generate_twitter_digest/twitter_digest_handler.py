from extractors.twitter.digest_extractor import process_twitter_digest

def process_twitter_digest_command(timeframe: str = "1d", only_relevant: bool = True) -> tuple[str, str]:
    """
    Processes a Twitter digest command.
    Returns a tuple: (exec_summary, notes)
    """
    return process_twitter_digest(timeframe=timeframe, only_relevant=only_relevant)
