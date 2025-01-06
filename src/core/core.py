import openai
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_GOVERNANCE_MODEL
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
    Check if the entire topic content is relevant to Lido.
    """
    relevance_prompt = PromptManager.get_governance_forum_relevance_prompt()
    
    # Combine the full text and the relevance prompt
    content = (
        f"Below is the entire content from a governance forum topic:\n\n"
        f"{topic_full_text}\n\n"
        f"{relevance_prompt}"
    )
    
    answer = query_openai(content)
    answer = answer.lower()
    return "yes" in answer


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