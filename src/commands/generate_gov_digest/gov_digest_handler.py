from extractors.governance_extractor import process_governance_forum

def process_gov_digest(timeframe: str = "1d", only_relevant: bool = True) -> dict:
    """
    Processes a governance forum digest.
    Returns a dictionary with keys:
      - 'exec_sum'
      - 'notes'
    """
    return process_governance_forum(timeframe=timeframe, only_relevant=only_relevant)
