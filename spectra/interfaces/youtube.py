# Specify the output file name for the audio
import ffmpeg
import os
from uuid import uuid4
import torchaudio
import yt_dlp

class SpectraYTInterface:
    def __init__(self, output_dir: str, file_name: str = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_name = file_name if file_name else f"audio_{str(uuid4())}"
        self.output_file = os.path.join(self.output_dir, self.file_name)

    def cache_audio(self, youtube_url: str) -> str:
        raise NotImplementedError()
    
    def download_audio(self, youtube_url: str) -> str:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",   # could use "flac" for lossless, or "mp3"
                "preferredquality": "192", # kbps for lossy formats like mp3
            }],
            "concurrent_fragment_downloads": 5,
            "outtmpl": self.output_file,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Load with torchaudio
        waveform, sample_rate = torchaudio.load(self.output_file)
        return self.output_file, waveform, sample_rate