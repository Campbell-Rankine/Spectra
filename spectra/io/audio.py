from pydub import AudioSegment
import numpy as np
from typing import Any, Optional
import os


class AudioIO:
    backend: str = "pydub"

    def __init__(
        self,
        path: str,
        target_sample_rate: Optional[int] = 44100,
        mono: Optional[bool] = True,
        normalize: Optional[bool] = True,
        chunk_size: Optional[int] = 1024,
        hop_size: Optional[int] = 512,
        dtype: Optional[Any] = np.float32,
    ):
        assert os.path.exists(path)
        self.path = path
        self.sample_rate = target_sample_rate
        self.mono = mono
        self.normalize = normalize
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.hop_size = hop_size

    def __normalize(self, samples):
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, self.dtype)
        samples /= np.max(np.abs(samples))
        return samples
    
    def read

    def read(self, verbose=False, truncate: Optional[int] = None):
        audio: AudioSegment = AudioSegment.from_file(self.path, format="wav")
        if self.mono:
            audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        else:
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.split_to_mono()
            left = audio[0].get_array_of_samples()
            right = audio[1].get_array_of_samples()
            # audio = audio.get_array_of_samples()
            samples = np.array([left, right]).astype(np.float32)
        if self.normalize:
            samples = self.__normalize(samples)

        if verbose:
            print(
                f"Sample shape: {samples.shape}, Audio type: {type(audio)}, Num Channels: {self.num_channels}, Sample Rate: {self.sample_rate}"
            )
        if truncate is None:
            return samples, self.sample_rate
        elif not self.mono:
            return samples[:, :truncate], self.sample_rate
        else:
            return samples[:truncate], self.sample_rate
