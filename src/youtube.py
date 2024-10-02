import os
import yt_dlp
import subprocess
import openai
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Function to download YouTube video as audio using yt-dlp
def download_youtube_video(url, output_format="/tmp/youtube/%(id)s.%(ext)s"):
    """
    Downloads the audio from a YouTube video using yt-dlp.
    """
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best available audio format
        'outtmpl': output_format,  # Save in the output format
        'noplaylist': True,  # Ensure that only a single video is downloaded
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info_dict)  # yt-dlp automatically adds the correct extension
            return file_path
    except Exception as e:
        print(f"Error downloading YouTube video {url}: {e}")
        return None


# Function to split large audio files into smaller chunks if necessary
def chunk_file_if_needed(file_path, max_size_mb=10):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        # Return the original path in a list if it does not exceed the max size
        return [file_path]
    else:
        # Calculate segment time based on estimated file size
        duration = get_audio_duration(file_path)
        estimated_segment_time = int(duration * (max_size_mb / file_size_mb))

        # Create a directory to store segments
        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        segments_dir = f"/tmp/youtube/{base_name}/segments"
        os.makedirs(segments_dir, exist_ok=True)

        # Use FFmpeg to split file into chunks
        cmd = (
            f"ffmpeg -i {file_path} -f segment -segment_time {estimated_segment_time} "
            f"-acodec libmp3lame -b:a 192k {segments_dir}/segment%09d.mp3"
        )

        os.system(cmd)

        # Return the list of chunk file paths
        return [os.path.join(segments_dir, f) for f in os.listdir(segments_dir) if f.startswith("segment")]


# Helper function to get audio duration using ffprobe
def get_audio_duration(file_path):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {file_path}"
    result = subprocess.check_output(cmd, shell=True)
    return float(result)


# Function to transcribe each audio segment using OpenAI's Whisper API
def transcribe_segments(segments, prompt=None):
    """
    Transcribe the audio segments using OpenAI's Whisper API.

    :param segments: List of paths to audio segment files.
    :param prompt: A prompt to provide context for the transcription.
    :return: Transcribed text from all segments combined.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    transcript = ""

    for segment in segments:
        with open(segment, "rb") as audio_file:
            res = openai.Audio.transcribe("whisper-1", audio_file)
            transcript += str(res['text'])

    return transcript


# Summarization template and methods
summary_template = """
    You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a transcript of a YouTube video related to Crypto/Web3 that may or may not be related to Lido.
    You are writing structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

    Your notes should be concise, detailed, and structured by topics. You know what information is especially important, and what is less important.

    Here is the transcript:
    {text}
    
    YOUR NOTES:
"""

refine_summary_template = """
    You are refining structured notes in markdown format for Lido Finance. Think of your notes as key takeaways, TLDRs, and executive summaries.

    Here is the existing note:
    {existing_answer}
    
    We have the opportunity to refine the existing note with some more context below:
    -----
    {text}
    -----
    
    Refine the original note with the new context. If the new context isn't useful, return the original summary.
    
    Your notes should be concise, detailed, and structured by topics.
"""

summary_prompt = PromptTemplate.from_template(summary_template)
refine_summary_prompt = PromptTemplate.from_template(refine_summary_template)


# Summarize the transcript using OpenAI's chat model
def summarize_transcript(transcript):
    try:
        doc = Document(page_content=transcript)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000, chunk_overlap=500, length_function=len, is_separator_regex=False
        )
        docs = text_splitter.split_documents([doc])
        llm = ChatOpenAI(temperature=0, model_name=os.getenv("OPENAI_MODEL"))
        chain = load_summarize_chain(
            llm, chain_type="refine", question_prompt=summary_prompt, refine_prompt=refine_summary_prompt
        )

        result = chain({"input_documents": docs}, return_only_outputs=True)

        return result["output_text"]

    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        return "Error summarizing transcript"


# Executive summary template
executive_template = """
    Given the summary of a YouTube video:
    {text}

    Generate an extremely brief executive summary for Lido executives. It should be concise, focused, and only contain information relevant to Lido Finance.
"""

refine_executive_template = """
    Refine an executive summary in markdown format based on the following summary:

    Existing summary:
    {existing_answer}

    New context:
    -----
    {text}
    -----

    Your refined executive summary:
"""

executive_prompt = PromptTemplate.from_template(executive_template)
refine_executive_prompt = PromptTemplate.from_template(refine_executive_template)


# Function to generate the executive summary
def get_executive_summary(summary):
    try:
        doc = Document(page_content=summary)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000, chunk_overlap=500, length_function=len, is_separator_regex=False
        )
        docs = text_splitter.split_documents([doc])
        llm = ChatOpenAI(temperature=0, model_name=os.getenv("OPENAI_MODEL"))
        chain = load_summarize_chain(
            llm, chain_type="refine", question_prompt=executive_prompt, refine_prompt=refine_executive_prompt
        )

        result = chain({"input_documents": docs}, return_only_outputs=True)
        return result["output_text"]
    except Exception as e:
        print(f"Error generating executive summary: {e}")
        return "Error generating executive summary"


# Main function to process a YouTube video
def process_youtube_video(url):
    """
    Downloads a YouTube video, transcribes the audio, and generates both summaries and an executive summary.
    """
    audio_file = download_youtube_video(url)

    if not audio_file:
        return "Error processing YouTube video"

    chunks = chunk_file_if_needed(audio_file)

    transcript = transcribe_segments(chunks, "YouTube video about Crypto, Web3, Liquid Staking, and Lido Finance")
    if not transcript:
        return "Error processing YouTube video transcript"

    summary = summarize_transcript(transcript)
    executive_summary = get_executive_summary(summary)

    return {'youtube_url': url, 'exec_sum': executive_summary, 'notes': summary}
