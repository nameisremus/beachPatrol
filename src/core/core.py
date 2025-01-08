import openai
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_GOVERNANCE_MODEL
)
from core.prompts import PromptManager
from langchain.schema import Document, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai.api_key = OPENAI_API_KEY

def query_openai(prompt):
    llm = ChatOpenAI(temperature=0, model_name=OPENAI_MODEL)
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def _get_model_for_media_type(media_type: str) -> str:
    """
    If media_type='governance_forum' and OPENAI_GOVERNANCE_MODEL is set,
    use the specialized governance model. Otherwise use OPENAI_MODEL.
    """
    if media_type == "governance_forum" and OPENAI_GOVERNANCE_MODEL:
        return OPENAI_GOVERNANCE_MODEL
    return OPENAI_MODEL

def check_topic_relevance(topic_full_text: str) -> bool:
    """
    Check if the entire text is relevant to Lido. (Used for governance topics.)
    """
    relevance_prompt = PromptManager.get_governance_forum_relevance_prompt()
    content = (
        f"Below is the entire content from a governance forum topic:\n\n"
        f"{topic_full_text}\n\n"
        f"{relevance_prompt}"
    )
    answer = query_openai(content)
    answer = answer.lower()
    return "yes" in answer

def categorize_governance_topic(topic_full_text: str) -> str:
    """
    Calls OpenAI with a specialized prompt to categorize a governance forum topic
    into exactly ONE of the 7 categories (plus default 'Misc. 🌀').
    """
    prompt_text = PromptManager.get_governance_forum_category_prompt()

    combined_text = (
        f"Full post:\n\n{topic_full_text}\n\n"
        f"{prompt_text}"
    )
    try:
        answer = query_openai(combined_text).strip()
        # Sanity-check. If GPT somehow returns multiple lines,
        # take the first line only:
        category_line = answer.splitlines()[0].strip()
        return category_line
    except Exception as e:
        print(f"[categorize_governance_topic] Error: {e}")
        return "Misc. 🌀"  # fallback

def check_tweet_relevance(tweet_text: str) -> bool:
    """
    Use GPT to decide if a tweet is relevant to Lido.
    """
    short_prompt = PromptManager.get_twitter_relevance_prompt()
    content = f"Tweet:\n{tweet_text}\n\n{short_prompt}"

    try:
        answer = query_openai(content).lower()
        return "yes" in answer
    except Exception as e:
        print(f"[check_tweet_relevance] Error: {e}")
        return False

def categorize_tweet(tweet_full_text: str) -> str:
    """
    Calls OpenAI with a specialized prompt to categorize a tweet
    into exactly ONE category from a provided list (plus default 'Misc. 🌀').
    Uses PromptManager.get_twitter_category_prompt().
    """
    prompt_text = PromptManager.get_twitter_category_prompt()
    combined_text = (
        f"Below is the entire tweet text:\n\n"
        f"{tweet_full_text}\n\n"
        f"{prompt_text}"
    )
    try:
        answer = query_openai(combined_text).strip()
        # Take the first line if multiple
        category_line = answer.splitlines()[0].strip()
        return category_line
    except Exception as e:
        print(f"[categorize_tweet] Error: {e}")
        return "Misc. 🌀"  # fallback

def summarize_transcript(transcript, media_type="article"):
    """
    Summarize the given transcript using the specified media_type's prompt.
    """
    try:
        used_model = _get_model_for_media_type(media_type)
        doc = Document(page_content=transcript)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,
            chunk_overlap=500
        )
        docs = text_splitter.split_documents([doc])
        question_prompt, refine_prompt = PromptManager.get_summary_prompts(media_type)

        llm = ChatOpenAI(temperature=0, model_name=used_model)
        chain = load_summarize_chain(
            llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text"
        )
        result = chain({"input_documents": docs}, return_only_outputs=True)
        return result["output_text"]
    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        return "Error summarizing transcript"

def get_executive_summary(summary, media_type="article"):
    """
    Produce an executive summary for the given summary text,
    using the prompt logic for the specified media type.
    """
    try:
        used_model = _get_model_for_media_type(media_type)
        doc = Document(page_content=summary)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,
            chunk_overlap=500
        )
        docs = text_splitter.split_documents([doc])
        question_prompt, refine_prompt = PromptManager.get_executive_prompts(media_type)

        llm = ChatOpenAI(temperature=0, model_name=used_model)
        chain = load_summarize_chain(
            llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text"
        )
        result = chain({"input_documents": docs}, return_only_outputs=True)
        return result["output_text"]
    except Exception as e:
        print(f"Error generating executive summary: {e}")
        return "Error generating executive summary"