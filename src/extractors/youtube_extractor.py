import os
import openai
import subprocess
from dotenv import load_dotenv
from core.core import summarize_transcript, get_executive_summary, do_custom_prompt, get_valid_model
import yt_dlp
from config import OPENAI_API_KEY

import logging
logger = logging.getLogger(__name__)

openai.api_key = OPENAI_API_KEY

def download_youtube_video(url, output_format="/tmp/youtube/%(id)s.%(ext)s"):
    """
    Download a YouTube video's audio via yt-dlp.
    Returns the local file path on success, or None on failure.
    """
    logger.info(
        "Downloading YouTube video",
        extra={"url": url, "output_format": output_format}
    )
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_format,
        'noplaylist': True,
        'cookiefile': "/app/cookies.txt",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info_dict)
            logger.info(
                "Download complete",
                extra={"url": url, "file_path": file_path}
            )
            return file_path
    except Exception:
        logger.error(
            "Error downloading YouTube video",
            extra={"url": url},
            exc_info=True
        )
        return None

def get_audio_duration(file_path):
    """
    Return the duration of an audio file (in seconds) via ffprobe.
    """
    cmd = (
        f"ffprobe -v error -show_entries format=duration "
        f"-of default=noprint_wrappers=1:nokey=1 {file_path}"
    )
    try:
        result = subprocess.check_output(cmd, shell=True)
        return float(result)
    except Exception:
        logger.error(
            "Error getting audio duration",
            extra={"file_path": file_path, "cmd": cmd},
            exc_info=True
        )
        return 0.0

def chunk_file_if_needed(file_path, max_size_mb=10):
    """
    If the downloaded file is larger than `max_size_mb`, split it into segments via ffmpeg,
    disabling any video track with `-vn` and converting to mp3 chunks.
    Otherwise, return the single file path in a list.
    """
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        logger.error(
            "Error checking file size",
            extra={"file_path": file_path},
            exc_info=True
        )
        return [file_path]

    if file_size_mb <= max_size_mb:
        logger.debug(
            "File under size threshold; no chunking needed",
            extra={"file_path": file_path, "size_mb": file_size_mb}
        )
        return [file_path]

    duration = get_audio_duration(file_path)
    if duration <= 0:
        logger.warning(
            "Skipping chunking due to invalid duration",
            extra={"file_path": file_path, "duration": duration}
        )
        return [file_path]

    estimated_segment_time = int(duration * (max_size_mb / file_size_mb))
    base_name = os.path.basename(file_path).rsplit('.', 1)[0]
    segments_dir = f"/tmp/youtube/{base_name}/segments"
    try:
        os.makedirs(segments_dir, exist_ok=True)
    except Exception:
        logger.error(
            "Error creating segments directory",
            extra={"segments_dir": segments_dir},
            exc_info=True
        )
        return [file_path]

    cmd = (
        f"ffmpeg -i '{file_path}' -f segment -segment_time {estimated_segment_time} "
        f"-acodec libmp3lame -b:a 192k '{segments_dir}/segment%09d.mp3'"
    )
    try:
        logger.info(
            "Running ffmpeg to chunk file",
            extra={"file_path": file_path, "cmd": cmd}
        )
        os.system(cmd)
    except Exception:
        logger.error(
            "Error running ffmpeg for chunking",
            extra={"file_path": file_path, "cmd": cmd},
            exc_info=True
        )

    try:
        segments = [
            os.path.join(segments_dir, f)
            for f in os.listdir(segments_dir)
            if f.startswith("segment")
        ]
        logger.info(
            "Chunking complete",
            extra={"file_path": file_path, "num_segments": len(segments)}
        )
        return segments
    except Exception:
        logger.error(
            "Error listing chunked segments",
            extra={"segments_dir": segments_dir},
            exc_info=True
        )
        return [file_path]

def transcribe_segments(segments, prompt=None):
    """
    Use OpenAI Whisper to transcribe each segment, concatenating into a single transcript string.
    """
    transcript = ""
    for segment in segments:
        try:
            with open(segment, "rb") as audio_file:
                if prompt:
                    res = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        prompt=prompt
                    )
                else:
                    res = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                transcript += res.text
        except Exception:
            logger.error(
                "Error transcribing segment",
                extra={"segment": segment},
                exc_info=True
            )
    return transcript


def process_youtube_video(url, model=None, prompt=None):
    """
    High level handler: download, chunk, transcribe, and summarize a YouTube video.
    """
    logger.info(
        "Processing YouTube video",
        extra={"url": url, "model": model, "prompt": bool(prompt)}
    )

    audio_file = download_youtube_video(url)
    if not audio_file:
        logger.error(
            "Download failed",
            extra={"url": url}
        )
        return {
            "youtube_url": url,
            "exec_sum": "",
            "notes": "Error processing YouTube video download"
        }

    try:
        chunks = chunk_file_if_needed(audio_file)
    except Exception:
        logger.error(
            "Error during chunking",
            extra={"file_path": audio_file},
            exc_info=True
        )
        return {
            "youtube_url": url,
            "exec_sum": "",
            "notes": "Error processing audio chunks"
        }

    try:
        transcript = transcribe_segments(chunks, prompt)
    except Exception:
        logger.error(
            "Error during transcription",
            extra={"youtube_url": url},
            exc_info=True
        )
        return {
            "youtube_url": url,
            "exec_sum": "",
            "notes": "Error processing YouTube video transcript"
        }

    if prompt:
        try:
            chosen_model = get_valid_model(model)
            combined_res = do_custom_prompt(transcript, prompt, chosen_model)
            return {
                "youtube_url": url,
                "exec_sum": combined_res,
                "notes": ""
            }
        except Exception:
            logger.error(
                "Error during custom prompt processing",
                extra={"youtube_url": url, "model": model},
                exc_info=True
            )
            return {
                "youtube_url": url,
                "exec_sum": "",
                "notes": "Error processing custom prompt"
            }
    else:
        try:
            summary = summarize_transcript(transcript, media_type="youtube")
            executive_summary = get_executive_summary(summary, media_type="youtube")
            return {
                "youtube_url": url,
                "exec_sum": executive_summary,
                "notes": summary
            }
        except Exception:
            logger.error(
                "Error generating summaries",
                extra={"youtube_url": url},
                exc_info=True
            )
            return {
                "youtube_url": url,
                "exec_sum": "",
                "notes": "Error generating YouTube video summary"
            }
