import os
import subprocess
import time
import openai
from dotenv import load_dotenv
import yt_dlp
from core.core import summarize_transcript, get_executive_summary

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_audio_duration(file_path):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {file_path}"
    result = subprocess.check_output(cmd, shell=True)
    return float(result)


def chunk_file_if_needed(file_path, max_size_mb=10):
    """
    If the downloaded file is larger than `max_size_mb`, split it into segments via ffmpeg,
    disabling any video track with `-vn` and converting to mp3 chunks.
    Otherwise, return the single file path in a list.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb <= max_size_mb:
        return [file_path]
    else:
        duration = get_audio_duration(file_path)
        estimated_segment_time = int(duration * (max_size_mb / file_size_mb))

        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        segments_dir = f"/tmp/spaces/{base_name}/segments"
        os.makedirs(segments_dir, exist_ok=True)

        # -vn disables video, -acodec libmp3lame converts to mp3
        cmd = (
            f"ffmpeg -protocol_whitelist file,https,httpproxy,tls,tcp "
            f"-i '{file_path}' -vn -f segment -segment_time {estimated_segment_time} "
            f"-acodec libmp3lame -b:a 192k '{segments_dir}/segment%09d.mp3'"
        )
        os.system(cmd)

        return [
            os.path.join(segments_dir, f)
            for f in os.listdir(segments_dir)
            if f.startswith("segment")
        ]


def transcribe_segments(segments):
    """
    Use OpenAI whisper to transcribe each segment, concatenating into a single transcript string.
    """
    transcript = ""
    for segment in segments:
        with open(segment, "rb") as audio_file:
            res = openai.Audio.transcribe("whisper-1", audio_file)
            transcript += str(res["text"])
    return transcript


def download_space_with_ytdlp(space_url: str, cookies_file: str, outdir: str = "/tmp/spaces") -> str:
    """
    Attempt to download the Twitter Space using yt-dlp.
    Return the local file path on success, or '' if it fails or the file is too small.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # 'format': 'bestaudio/best' typically gets the best audio stream
    ydl_opts = {
        "cookiefile": cookies_file,
        "outtmpl": f"{outdir}/%(id)s.%(ext)s",
        "format": "bestaudio/best",
        # 'quiet': True,
        # 'geo_bypass': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(space_url, download=True)
            file_path = ydl.prepare_filename(info)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
                return file_path  # success
    except Exception as e:
        print(f"[yt-dlp] Error downloading space: {e}")

    return ""


def process_twitter_space(space_url, cookies_path):
    """
    1. Download the Twitter Space using yt-dlp (and cookies).
    2. If the file is missing/too small, return an error message.
    3. Otherwise chunk it if needed, transcribe with whisper, and summarize.
    4. Return final 'exec_sum' and 'notes'.
    """
    # 1) Download with yt-dlp
    downloaded_file = download_space_with_ytdlp(space_url, cookies_path)
    if not downloaded_file:
        return {
            "space_url": space_url,
            "exec_sum": "",
            "notes": "Error: Failed to download Twitter Space. Possibly region-locked or no replay."
        }

    # 2) Chunk if needed
    chunks = chunk_file_if_needed(downloaded_file)

    # 3) Transcribe
    transcript = transcribe_segments(chunks)

    # 4) Summarize
    summary = summarize_transcript(transcript, media_type="twitter_space")
    executive_summary = get_executive_summary(summary, media_type="twitter_space")

    return {
        "space_url": space_url,
        "exec_sum": executive_summary,
        "notes": summary
    }
