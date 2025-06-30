import numpy as np
import os
from typing import Optional, List, Any, Dict
from scipy.io import wavfile

class WavReader:
    backend: str = "scipy"

    def __init__(self, path: str):
        assert os.path.exists(path)
        self.path = path

    def __call__(self):
        samplerate, data = wavfile.read(self.path)
        return samplerate, data
    
class WavStreamReader(WavReader):
    backend: str = "..." # TODO: Find wav streaming

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def read(self):
        raise NotImplementedError