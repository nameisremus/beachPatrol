import requests
from bs4 import BeautifulSoup
from .IExtractor import IExtractor

class GitHubAdvisoryExtractor(IExtractor):
    def extract_details(self, url: str) -> dict:
        """
        If the extraction target is a github advisory page, use BeautifulSoup
        to extract the content from the <main> tag
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception as e:
            raise Exception(f"Error fetching URL {url}: {e}")
        
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        
        # Look for the <main> element, which contains all the advisory content.
        main_content = soup.find("main")
        if main_content:
            advisory_text = main_content.get_text(separator="\n", strip=True)
        else:
            advisory_text = soup.get_text(separator="\n", strip=True)
        
        return {
            'text': advisory_text,
            'summary': "",
            'authors': [],
            'publish_date': None,
            'top_image': None,
            'movies': [],
            'keywords': [],
        }

def extract_github_advisory_details(url: str) -> dict:
    extractor = GitHubAdvisoryExtractor()
    return extractor.extract_details(url)
