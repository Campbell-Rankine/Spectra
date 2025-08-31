# Specify the output file name for the audio
import ffmpeg
import os
from uuid import uuid4
import torchaudio
import yt_dlp
from pytube import YouTube
import torch

from spectra.interfaces.base import _BaseInterface

class SpectraYTubeInterface(_BaseInterface):
    @property
    def backend(self) -> str:
        return "pytube"
    
    def cache_audio(self, youtube_url: str) -> tuple[str, torch.Tensor, int] :
        raise NotImplementedError()
    
    def download_audio(self, youtube_url: str) -> str:
        _audio = YouTube(youtube_url).streams.filter(only_audio=True, file_extension="wav").first()
        # Convert to wav

        # Load with torchaudio
        waveform, sample_rate = torchaudio.load(_audio)
        return self.output_file, waveform, sample_rate

class SpectraYTDLPInterface:
    @property
    def backend(self) -> str:
        return "yt-dlp"

    def cache_audio(self, youtube_url: str) -> tuple[str, torch.Tensor, int] :
        raise NotImplementedError()
    
    def download_audio(self, youtube_url: str) -> str:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",   # could use "flac" for lossless, or "mp3"
                "preferredquality": "96", # kbps for lossy formats like mp3
            }],
            "concurrent_fragment_downloads": 10,
            "outtmpl": self.output_file,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Load with torchaudio
        waveform, sample_rate = torchaudio.load(self.output_file)
        return self.output_file, waveform, sample_rate