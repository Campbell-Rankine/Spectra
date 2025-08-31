import ffmpeg
import os
from uuid import uuid4
import torchaudio
from abc import ABC, abstractmethod
import torch

class _BaseInterface(ABC):
    def __init__(self, output_dir: str, file_name: str = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_name = file_name if file_name else f"audio_{str(uuid4())}"
        self.output_file = os.path.join(self.output_dir, self.file_name)
    
    @abstractmethod
    def cache_audio(self, youtube_url: str) -> tuple[str, torch.Tensor, int] :
        pass

    @abstractmethod
    def download_audio(self, youtube_url: str) -> str:
        pass

    @property
    def backend(self) -> str:
        pass