import numpy as np
import os
from typing import Optional, List, Any, Dict
from scipy.io import wavfile
import pyaudio
import wave

from spectra.io.stream import _AudioStream


class WavReader:
    backend: str = "scipy"

    def __init__(self, path: str):
        assert os.path.exists(path)
        self.path = path

    def __call__(self):
        samplerate, data = wavfile.read(self.path)
        return samplerate, data


class FFTStreamReader(_AudioStream):
    backend: str = "pyAudio"  # TODO: Find wav streaming

    def __init__(
        self,
        rate: Optional[int] = 44100,
        chunk_size: Optional[int] = 1024,
        stream_fmt: Optional[Any] = pyaudio.paInt16,
        num_channels: Optional[int] = 1,
    ):
        super().__init__(rate, chunk_size, stream_fmt, num_channels)
        self.fft_chunks = []

    def read_spectra(self, **kw):
        chunk = super().read(**kw)
        samples = np.frombuffer(chunk, dtype=np.int16)
        fft_data = np.abs(np.fft.rfft(samples))
        self.fft_chunks.append(fft_data)
        return fft_data
