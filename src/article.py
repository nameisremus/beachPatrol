import newspaper
import fitz  # PyMuPDF
import requests
from twitter import summarize_transcript, get_executive_summary

# Function to extract article details using newspaper or PyMuPDF for PDFs
def extract_article_details(url):
    if url.endswith(".pdf") or "arxiv" in url:
        # Handle PDF case
        return extract_pdf_details(url)
    else:
        # Handle regular articles using newspaper
        article = newspaper.Article(url)
        article.download()  # Download the article
        article.parse()  # Parse the article

        # Fetch important details
        article_details = {
            'authors': article.authors,
            'publish_date': article.publish_date,
            'text': article.text,
            'top_image': article.top_image,
            'movies': article.movies
        }

        # Perform NLP for summarization and keyword extraction
        article.nlp()
        article_details['keywords'] = article.keywords
        article_details['summary'] = article.summary

        return article_details

# Function to extract text from a PDF file using PyMuPDF
def extract_pdf_details(pdf_url):
    # Download the PDF
    response = requests.get(pdf_url)
    pdf_content = response.content

    # Save the PDF to a temporary file
    temp_pdf_path = "/tmp/temp_article.pdf"
    with open(temp_pdf_path, 'wb') as f:
        f.write(pdf_content)

    # Extract text from the PDF using PyMuPDF
    pdf_text = ""
    with fitz.open(temp_pdf_path) as pdf_doc:
        for page in pdf_doc:
            pdf_text += page.get_text()

    return {
        'text': pdf_text,
        'authors': [],  # You might need to extract authors separately
        'publish_date': None,  # You might need to extract publish date separately
        'top_image': None,
        'movies': [],
        'keywords': [],
        'summary': ""  # Will generate summary after processing
    }

# Updated function to process an article
def process_article(url):
    # Extract article content and details
    article_details = extract_article_details(url)

    # Perform summarization using the text content
    summary = summarize_transcript(article_details['text'])  # Reuse the summarization function

    # Generate executive summary
    exec_summary = get_executive_summary(summary)

    # Return the article details along with the generated summaries
    return {
        'article_url': url,
        'exec_sum': exec_summary,
        'notes': summary,
        'article_details': article_details
    }
