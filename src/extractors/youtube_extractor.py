import os
import openai
import subprocess
from dotenv import load_dotenv
from core.core import summarize_transcript, get_executive_summary
import yt_dlp
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def download_youtube_video(url, output_format="/tmp/youtube/%(id)s.%(ext)s"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_format,
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info_dict)
            return file_path
    except Exception as e:
        print(f"Error downloading YouTube video {url}: {e}")
        return None

def get_audio_duration(file_path):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {file_path}"
    result = subprocess.check_output(cmd, shell=True)
    return float(result)

def chunk_file_if_needed(file_path, max_size_mb=10):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        return [file_path]
    else:
        duration = get_audio_duration(file_path)
        estimated_segment_time = int(duration * (max_size_mb / file_size_mb))
        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        segments_dir = f"/tmp/youtube/{base_name}/segments"
        os.makedirs(segments_dir, exist_ok=True)

        cmd = (
            f"ffmpeg -i {file_path} -f segment -segment_time {estimated_segment_time} "
            f"-acodec libmp3lame -b:a 192k {segments_dir}/segment%09d.mp3"
        )
        os.system(cmd)
        return [os.path.join(segments_dir, f) for f in os.listdir(segments_dir) if f.startswith("segment")]

def transcribe_segments(segments, prompt=None):
    transcript = ""
    for segment in segments:
        with open(segment, "rb") as audio_file:
            res = openai.Audio.transcribe("whisper-1", audio_file)
            transcript += str(res['text'])
    return transcript

def process_youtube_video(url):
    audio_file = download_youtube_video(url)
    if not audio_file:
        return "Error processing YouTube video"

    chunks = chunk_file_if_needed(audio_file)
    transcript = transcribe_segments(chunks, "YouTube video about Crypto, Web3, Liquid Staking, and Lido Finance")
    if not transcript:
        return "Error processing YouTube video transcript"

    summary = summarize_transcript(transcript, media_type="youtube")
    executive_summary = get_executive_summary(summary, media_type="youtube")

    return {'youtube_url': url, 'exec_sum': executive_summary, 'notes': summary}