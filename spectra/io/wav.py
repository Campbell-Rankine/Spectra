import numpy as np
import os
from typing import Optional, List, Any, Dict
from scipy.io import wavfile

class WavReader:
    backend: str = "scipy"

    def __init__(self, path: str):
        assert os.path.exists(path)
        self.path = path

    def __call__(self, as_stream: Optional[bool] = False):
        match as_stream:
            case False:
                samplerate, data = wavfile.read(self.path)
                return samplerate, data
            case _:
                raise ValueError(f"Invalid stream input")