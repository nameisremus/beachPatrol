import os
import subprocess
import openai
from dotenv import load_dotenv
from twspace_dl.api import API
from twspace_dl.cookies import load_cookies
from twspace_dl.twspace import Twspace
from twspace_dl.twspace_dl import TwspaceDL
from core.core import summarize_transcript, get_executive_summary

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_twitter_space_if_live(user_url, cookies_path):
    API.init_apis(load_cookies(cookies_path))
    try:
        twspace = Twspace.from_user_avatar(user_url)
        return {"url": twspace.url} if twspace else None
    except Exception as e:
        print(e)
        return None

def download_twitter_space_direct(space_url, cookie_file, output_format="/tmp/spaces/%(creator_id)s-%(id)s"):
    API.init_apis(load_cookies(cookie_file))
    twspace = Twspace.from_space_url(space_url)
    twspace_dl = TwspaceDL(twspace, output_format)
    file_save_path = twspace_dl.filename + ".m4a"

    if not os.path.exists("/tmp/spaces/"):
        os.makedirs("/tmp/spaces/")

    try:
        twspace_dl.download()
        twspace_dl.embed_cover()
    except KeyboardInterrupt:
        print("Download Interrupted by user")
    finally:
        twspace_dl.cleanup()

    return file_save_path if os.path.exists(file_save_path) else None

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
        segments_dir = f"/tmp/spaces/{base_name}/segments"
        os.makedirs(segments_dir, exist_ok=True)

        cmd = (
            f"ffmpeg -i {file_path} -f segment -segment_time {estimated_segment_time} "
            f"-acodec libmp3lame -b:a 192k {segments_dir}/segment%09d.mp3"
        )
        os.system(cmd)

        return [os.path.join(segments_dir, f) for f in os.listdir(segments_dir) if f.startswith("segment")]

def transcribe_segments(segments):
    transcript = ""
    for segment in segments:
        with open(segment, "rb") as audio_file:
            res = openai.Audio.transcribe("whisper-1", audio_file)
            transcript += str(res['text'])
    return transcript

def process_twitter_space(space_url, cookies_path):
    transcript_location = download_twitter_space_direct(space_url, cookies_path)
    chunks = chunk_file_if_needed(transcript_location)
    transcript = transcribe_segments(chunks)
    summary = summarize_transcript(transcript, media_type="twitter_space")
    executive_summary = get_executive_summary(summary, media_type="twitter_space")
    return {'space_url': space_url, 'exec_sum': executive_summary, 'notes': summary}