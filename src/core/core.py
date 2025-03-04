import openai
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_GOVERNANCE_MODEL,
    CONTENT_TAGS,
    OPENAI_MODELS_LIST
)
from core.prompts import PromptManager
from langchain.schema import Document, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai.api_key = OPENAI_API_KEY

def get_valid_model(model_candidate: str) -> str:
    """
    Checks if `model_candidate` is in OPENAI_MODELS_LIST (case-sensitive).
    If so, returns that candidate. Otherwise, returns the default OPENAI_MODEL.
    """
    if not model_candidate:
        return OPENAI_MODEL
    if model_candidate.strip() in [m.strip() for m in OPENAI_MODELS_LIST if m.strip()]:
        return model_candidate.strip()
    return OPENAI_MODEL

def query_openai(prompt):
    llm = ChatOpenAI(temperature=0, model_name=OPENAI_MODEL)
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def do_custom_prompt(full_text: str, user_prompt: str, model: str) -> str:
    """
    If a user supplies `prompt`, we use it directly instead of PromptManager logic.
    We feed the full_text plus the user_prompt into one OpenAI call.
    Returns the entire response as a single string.
    """
    model_used = get_valid_model(model)
    llm = ChatOpenAI(temperature=0, model_name=model_used)
    # We simply combine everything into a single message:
    combined_prompt = (
        f"{user_prompt}\n\n"
        f"Raw content:\n\n{full_text}\n\n"
        f"Please follow only the user's prompt above for that raw content."
    )
    response = llm([HumanMessage(content=combined_prompt)])
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
        return "Misc. 🌀"

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

def escape_markdown_v2(text: str) -> str:
    """
    Escapes Telegram MarkdownV2 special characters.
    """
    text = text.replace("\\", "\\\\")
    reserved_chars = "_*[]()~`>#+-=|{}.!"
    for char in reserved_chars:
        text = text.replace(char, f"\\{char}")
    return text

def format_for_telegram(text: str) -> str:
    """
    Prompts OpenAI to convert Discord-formatted text into Telegram-friendly HTML.
    """
    if len(text.strip()) < 5:
        return text
    prompt_template = PromptManager.get_telegram_format_prompt()
    prompt = prompt_template.format(text=text)
    return query_openai(prompt)

def get_content_tags(full_text: str, media_type: str) -> list:
    """
    Uses GPT to pick the top relevant tags from our CONTENT_TAGS list
    for the given text. 
    """
    if not CONTENT_TAGS:
        return []
    possible_tags = [tag.get("tag_name", "Misc") for tag in CONTENT_TAGS]
    joined_tags = "\n".join([f"- {t}" for t in possible_tags])
    prompt_template = PromptManager.get_content_tags_prompt()
    prompt = prompt_template.format(media_type=media_type, full_text=full_text, possible_tags=joined_tags)
    try:
        raw_answer = query_openai(prompt)
        chosen = [x.strip() for x in raw_answer.split(",") if x.strip()]
        final_tags = []
        for c in chosen:
            for tag in possible_tags:
                if c.lower() == tag.lower():
                    final_tags.append(tag)
                    break
        return final_tags
    except Exception as e:
        print(f"[get_content_tags] Error: {e}")
        return []
