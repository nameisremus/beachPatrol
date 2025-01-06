import newspaper
import fitz  # PyMuPDF for PDF parsing
import requests
from core.core import summarize_transcript, get_executive_summary
from config import PDF_TEMP_PATH

def extract_pdf_details(pdf_url):
    response = requests.get(pdf_url)
    pdf_content = response.content

    with open(PDF_TEMP_PATH, 'wb') as f:
        f.write(pdf_content)

    pdf_text = ""
    with fitz.open(PDF_TEMP_PATH) as pdf_doc:
        for page in pdf_doc:
            pdf_text += page.get_text()

    return {
        'text': pdf_text,
        'authors': [],
        'publish_date': None,
        'top_image': None,
        'movies': [],
        'keywords': [],
        'summary': ""
    }

def extract_article_details(url):
    if url.endswith(".pdf") or "arxiv" in url:
        return extract_pdf_details(url)
    else:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        article_details = {
            'authors': article.authors,
            'publish_date': article.publish_date,
            'text': article.text,
            'top_image': article.top_image,
            'movies': article.movies
        }
        article.nlp()
        article_details['keywords'] = article.keywords
        article_details['summary'] = article.summary
        return article_details

def process_article(url):
    article_details = extract_article_details(url)
    summary = summarize_transcript(article_details['text'], media_type="article")
    exec_summary = get_executive_summary(summary, media_type="article")

    return {
        'article_url': url,
        'exec_sum': exec_summary,
        'notes': summary,
        'article_details': article_details
    }