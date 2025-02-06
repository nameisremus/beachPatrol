from abc import ABC, abstractmethod

class IExtractor(ABC):
    @abstractmethod
    def extract_details(self, url: str) -> dict:
        """
        Given a URL, extract the relevant details and return them in a dictionary.
        The dictionary should include:
          - 'text': The full extracted text.
          - 'summary': A (possibly preliminary) summary.
          - Other keys such as 'authors', 'publish_date', etc.
        """
        pass
