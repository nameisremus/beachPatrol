import requests
import fitz  # PyMuPDF
from config import PDF_TEMP_PATH
from .IExtractor import IExtractor

class PDFExtractor(IExtractor):
    def extract_details(self, url: str) -> dict:
        """
        If the extraction target is a pdf, use PyMuPDF to extract the content
        """
        response = requests.get(url)
        pdf_content = response.content

        with open(PDF_TEMP_PATH, 'wb') as f:
            f.write(pdf_content)

        pdf_text = ""
        with fitz.open(PDF_TEMP_PATH) as pdf_doc:
            for page in pdf_doc:
                pdf_text += page.get_text()

        return {
            'text': pdf_text,
            'summary': "",
            'authors': [],
            'publish_date': None,
            'top_image': None,
            'movies': [],
            'keywords': [],
        }

def extract_pdf_details(url: str) -> dict:
    extractor = PDFExtractor()
    return extractor.extract_details(url)
