from click import Option
import numpy as np
from typing import Optional, Any, Iterable
from scipy.fft import rfft, rfftfreq, irfft, fftshift
from zmq import has


def compute_fft_frames(samples, rate, frame_size=1024, hop_size=512):
    num_frames = (len(samples) - frame_size) // hop_size
    frames = [
        samples[i * hop_size : i * hop_size + frame_size] * np.hanning(frame_size)
        for i in range(num_frames)
    ]
    fft_frames = [np.abs(rfft(frame)) for frame in frames]
    freqs = rfftfreq(frame_size, d=1 / rate)
    return fft_frames, freqs


class _BaseFFT:
    def __init__(
        self,
        sample_rate: Optional[int] = 44100,
        frame_size: Optional[int] = 1024,
        hop_size: Optional[int] = 512,
        shift_frames: Optional[bool] = False,
        zoom: Optional[int] = 21,
        smoothing: Optional[float] = 0.05,
    ):
        # init generator data
        if sample_rate is None:
            self.sample_rate = 2 * self.nyquist_frequency
        else:
            self.sample_rate = sample_rate

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.shift_frames = shift_frames
        self.zoom = zoom
        self.smoothing = smoothing
        self.prev = {}

        # init transform data
        self.transforms = {}
        self.transform_kw = {}

    @property
    def nyquist_frequency(self):
        return 22050  # hertz

    def smooth(self, value1, value2):
        if np.sum(value1) == 0:
            return value2
        elif np.sum(value2) == 0:
            return value1
        smoothed_value = (self.smoothing * value1) + ((1 - self.smoothing) * value2)
        return smoothed_value

    def register_transform(
        self, key: str, fn: callable, inverse_fn: Optional[callable] = None, **kw
    ):
        self.transforms[key] = fn
        self.transform_kw[key] = kw

    def apply(self, x: Iterable, dim_reduction=False):
        result = x
        if not isinstance(result, np.ndarray):
            result = np.asarray(result)
        for k, v in self.transforms.items():
            _transformed_x = v(x, **self.transform_kw[k])
            if (not dim_reduction) and (not _transformed_x.shape == x.shape):
                raise ValueError(
                    f"After transform={k}, shape of x: {x.shape} changed to {_transformed_x.shape}. If this is expected set dim_reduction=True"
                )
            else:
                result = _transformed_x
        return result


class Real1DFFT(_BaseFFT):
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        frame_size: Optional[int] = 1024,
        hop_size: Optional[int] = 512,
        shift_frames: Optional[bool] = False,
        zoom: Optional[int] = 21,
        smoothing: Optional[float] = 0.05,
    ):
        super().__init__(
            sample_rate, frame_size, hop_size, shift_frames, zoom, smoothing
        )

    def __call__(
        self,
        x: Iterable,
        apply_transforms: Optional[bool] = True,
        transform_step: Optional[str] = "after",
        dim_reduction: Optional[bool] = False,
    ):
        if transform_step == "before" and apply_transforms:
            x = self.apply(x, dim_reduction=dim_reduction)

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        if len(x.shape) > 1:
            x = np.mean(x, axis=1)  # convert to mono

        num_frames = (len(x) - self.frame_size) // self.hop_size
        frames = [
            x[i * self.hop_size : i * self.hop_size + self.frame_size]
            * np.hanning(self.frame_size)
            for i in range(num_frames)
        ]
        fft_frames = [np.abs(rfft(frame)) for frame in frames]

        freqs = rfftfreq(self.frame_size, d=1 / self.sample_rate)

        if self.shift_frames:
            fft_frames = fftshift(fft_frames)
            freqs = fftshift(freqs)

        if transform_step == "after" and apply_transforms:
            fft_frames = self.apply(x, dim_reduction)

        return fft_frames, freqs


class Real1DLogFFT(_BaseFFT):
    allowed_bases: list = [np.e, 2, 10]
    backend: str = "scipy"

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        frame_size: Optional[int] = 1024,
        hop_size: Optional[int] = 512,
        shift_frames: Optional[bool] = False,
        zoom: Optional[int] = 21,
        log_base: Optional[float] = np.e,
        min_frequency: Optional[int] = 10,
        max_frequency: Optional[int | str] = "nyquist",
        num_log_bins: Optional[int] = 1024,
        smoothing: Optional[float] = 0.05,
    ):
        super().__init__(
            sample_rate, frame_size, hop_size, shift_frames, zoom, smoothing
        )

        assert log_base in self.allowed_bases
        self.log_base = log_base
        self.num_log_bins = num_log_bins
        self.min_frequency = min_frequency

        if isinstance(max_frequency, str) and max_frequency == "nyquist":
            self.max_frequency = self.nyquist_frequency
        elif isinstance(max_frequency, (float, int)):
            self.max_frequency = max_frequency
        else:
            raise ValueError(
                f"Invalid frequency type for max frequency={type(max_frequency)}, {max_frequency}"
            )
        self.curr_frame = -1
        self.previous_value = -1

    def _get_logspace_freqs(self, freqs):

        log_bin_edges = (
            np.logspace(
                np.log(self.min_frequency),
                np.log(self.max_frequency),
                num=self.num_log_bins + 1,
                base=self.log_base,
            )
            * self.zoom
        )

        log_bin_indices = np.digitize(freqs, log_bin_edges) - 1  # Map freqs to bins
        return log_bin_edges, log_bin_indices

    def aggregate_log_bins(self, fft_frame, log_bin_indices, normalize=True):
        log_magnitudes = np.zeros(self.num_log_bins)
        for i in range(self.num_log_bins):
            bin_mask = log_bin_indices == i
            if np.any(bin_mask):
                log_magnitudes[i] = np.mean(fft_frame[bin_mask])
        if normalize:
            log_magnitudes /= np.max(log_magnitudes)
        return log_magnitudes

    def load(
        self,
        x,
        convert_to_mono: Optional[bool] = False,
    ):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        if len(x.shape) > 1 and convert_to_mono:
            x = np.mean(x, axis=1)  # convert to mono

        num_frames = (len(x) - self.frame_size) // self.hop_size
        frames = [
            x[i * self.hop_size : i * self.hop_size + self.frame_size]
            * np.hanning(self.frame_size)
            for i in range(num_frames)
        ]
        self.fft_frames = [np.abs(rfft(frame)) for frame in frames]
        self.freqs = rfftfreq(self.frame_size, d=1 / self.sample_rate)
        self.log_edges, self.log_bins = self._get_logspace_freqs(self.freqs)

    def __iter__(self):
        return next(iter(self))

    def __next__(self):
        self.curr_frame = self.curr_frame + 1
        assert (
            hasattr(self, "fft_frames")
            and hasattr(self, "freqs")
            and hasattr(self, "log_edges")
            and hasattr(self, "log_bins")
        )
        if self.curr_frame >= len(self.fft_frames):
            print("Stopping on %i" % self.curr_frame)
            raise StopIteration()
        else:
            frame = self.fft_frames[self.curr_frame]
            log_fft = self.aggregate_log_bins(frame, self.log_bins)
            if self.previous_value == -1:
                self.previous_value = np.zeros_like(log_fft)
            smoothed_log_fft = self.smooth(self.previous_value, log_fft)
            self.previous_value = log_fft
            yield smoothed_log_fft
