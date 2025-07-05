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
        num_channels: Optional[int] = 2,
        normalize: Optional[bool] = True,
        dtype: Optional[Any] = np.float32,
    ):
        assert os.path.exists(path)
        self.path = path
        self.sample_rate = target_sample_rate
        self.num_channels = num_channels
        self.normalize = normalize
        self.dtype = dtype

    def __normalize(self, samples):
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, self.dtype)
        samples /= np.max(np.abs(samples))
        return samples

    def read(self, verbose=False):
        audio = AudioSegment.from_file(self.path)
        audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        if self.normalize:
            samples = self.__normalize(samples)

        if verbose:
            print(
                f"Sample shape: {samples.shape}, Audio type: {type(audio)}, Num Channels: {self.num_channels}, Sample Rate: {self.sample_rate}"
            )
        return samples, self.sample_rate
