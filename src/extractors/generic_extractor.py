from newspaper import Article
from .IExtractor import IExtractor

class GenericExtractor(IExtractor):
    def extract_details(self, url: str) -> dict:
        """
        If the source is unidentified, falls back to extracting the details
        using the newspaper library
        """
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        return {
            'text': article.text,
            'summary': article.summary,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'top_image': article.top_image,
            'movies': article.movies,
            'keywords': article.keywords,
        }

# Helper method
def extract_generic_article_details(url: str) -> dict:
    extractor = GenericExtractor()
    return extractor.extract_details(url)
