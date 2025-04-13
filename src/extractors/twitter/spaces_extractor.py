import os
import subprocess
import openai
import yt_dlp
from core.core import summarize_transcript, get_executive_summary, do_custom_prompt, get_valid_model
from config import OPENAI_API_KEY

import logging

logger = logging.getLogger(__name__)

openai.api_key = OPENAI_API_KEY


def get_audio_duration(file_path):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {file_path}"
    try:
        result = subprocess.check_output(cmd, shell=True)
        return float(result)
    except Exception as e:
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
    except Exception as e:
        logger.error(
            "Error checking file size",
            extra={"file_path": file_path},
            exc_info=True
        )
        return [file_path]

    if file_size_mb <= max_size_mb:
        return [file_path]

    # need to chunk
    duration = get_audio_duration(file_path)
    if duration <= 0:
        logger.warning(
            "Skipping chunking due to invalid duration",
            extra={"file_path": file_path, "duration": duration}
        )
        return [file_path]

    estimated_segment_time = int(duration * (max_size_mb / file_size_mb))

    base_name = os.path.basename(file_path).rsplit('.', 1)[0]
    segments_dir = f"/tmp/spaces/{base_name}/segments"
    try:
        os.makedirs(segments_dir, exist_ok=True)
    except Exception as e:
        logger.error(
            "Error creating segments directory",
            extra={"segments_dir": segments_dir},
            exc_info=True
        )
        return [file_path]

    # -vn disables video, -acodec libmp3lame converts to mp3
    cmd = (
        f"ffmpeg -protocol_whitelist file,https,httpproxy,tls,tcp "
        f"-i '{file_path}' -vn -f segment -segment_time {estimated_segment_time} "
        f"-acodec libmp3lame -b:a 192k '{segments_dir}/segment%09d.mp3'"
    )
    try:
        os.system(cmd)
    except Exception as e:
        logger.error(
            "Error running ffmpeg for chunking",
            extra={"file_path": file_path, "cmd": cmd},
            exc_info=True
        )

    try:
        return [
            os.path.join(segments_dir, f)
            for f in os.listdir(segments_dir)
            if f.startswith("segment")
        ]
    except Exception as e:
        logger.error(
            "Error listing chunked segments",
            extra={"segments_dir": segments_dir},
            exc_info=True
        )
        return [file_path]


def transcribe_segments(segments):
    """
    Use OpenAI whisper to transcribe each segment, concatenating into a single transcript string.
    """
    transcript = ""
    for segment in segments:
        try:
            with open(segment, "rb") as audio_file:
                res = openai.Audio.transcribe("whisper-1", audio_file)
                transcript += str(res["text"])
        except Exception as e:
            logger.error(
                "Error transcribing segment",
                extra={"segment": segment},
                exc_info=True
            )
    return transcript


def download_space_with_ytdlp(space_url: str, cookies_file: str, outdir: str = "/tmp/spaces") -> str:
    """
    Download the Twitter Space using yt-dlp.
    Return the local file path on success, or '' if it fails or the file is too small.
    """
    try:
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
    except Exception as e:
        logger.error(
            "Error creating output directory",
            extra={"outdir": outdir},
            exc_info=True
        )
        return ""

    ydl_opts = {
        "cookiefile": cookies_file,
        "outtmpl": f"{outdir}/%(id)s.%(ext)s",
        "format": "bestaudio/best",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(space_url, download=True)
            file_path = ydl.prepare_filename(info)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
                return file_path
            else:
                logger.warning(
                    "Downloaded file missing or too small",
                    extra={"file_path": file_path}
                )
    except Exception as e:
        logger.error(
            "Error downloading space with yt-dlp",
            extra={"space_url": space_url},
            exc_info=True
        )

    return ""


def process_twitter_space(space_url, cookies_path, model=None, prompt=None):
    downloaded_file = download_space_with_ytdlp(space_url, cookies_path)
    if not downloaded_file:
        return {
            "space_url": space_url,
            "exec_sum": "",
            "notes": "Error: Failed to download Twitter Space. Possibly region-locked or no replay."
        }

    # 2) Chunk if needed
    try:
        chunks = chunk_file_if_needed(downloaded_file)
    except Exception as e:
        logger.error(
            "Error during chunking",
            extra={"file_path": downloaded_file},
            exc_info=True
        )
        return {
            "space_url": space_url,
            "exec_sum": "",
            "notes": "Error processing audio chunks."
        }

    # 3) Transcribe
    try:
        transcript = transcribe_segments(chunks)
    except Exception as e:
        logger.error(
            "Error during transcription",
            extra={"space_url": space_url},
            exc_info=True
        )
        return {
            "space_url": space_url,
            "exec_sum": "",
            "notes": "Error transcribing audio."
        }

    if prompt:
        try:
            chosen_model = get_valid_model(model)
            combined_res = do_custom_prompt(transcript, prompt, chosen_model)
            return {
                "space_url": space_url,
                "exec_sum": combined_res,
                "notes": ""
            }
        except Exception as e:
            logger.error(
                "Error during custom prompt processing",
                extra={"space_url": space_url, "model": model},
                exc_info=True
            )
            return {
                "space_url": space_url,
                "exec_sum": "",
                "notes": "Error processing custom prompt."
            }
    else:
        try:
            summary = summarize_transcript(transcript, media_type="twitter_space")
            executive_summary = get_executive_summary(summary, media_type="twitter_space")
            return {
                "space_url": space_url,
                "exec_sum": executive_summary,
                "notes": summary
            }
        except Exception as e:
            logger.error(
                "Error generating summaries",
                extra={"space_url": space_url},
                exc_info=True
            )
            return {
                "space_url": space_url,
                "exec_sum": "",
                "notes": "Error generating summary."
            }
