import numpy as np
from manim import *
from scipy.fft import rfft, rfftfreq
import os
import logging
from typing import Any, Optional
from scipy.interpolate import interp1d
from matplotlib import cm
from manim import color

from spectra.io.audio import AudioIO
from spectra.cmaps import spectra, spectra_warm
from spectra.fft import compute_fft_frames, Real1DLogFFT

class _BaseAudioVisualizer(Scene):
    backend: str = "manim"
    logger: logging.Logger = None
    animation_counter: int = 0
    num_animations: int = None

    def __init__(
        self,
        path: str,
        num_bins: int,
        log_scale: Optional[bool] = True,
        log_base: Optional[float] = None,
        min_frequency: Optional[int] = 10,
        max_frequency: Optional[int | str] = "nyquist",
        max_amplitude: Optional[float] = 6,
        fps: Optional[int] = 20,
        chunk_size: Optional[int] = 1024,
        hop_size: Optional[int] = 512,
        opacity: Optional[float] = 0.8,
        spacing: Optional[float] = 0.05,
        downsampling: Optional[int] = 2,
        height_clipping: Optional[float] = 3.0,
        min_height: Optional[float] = 0.01,
        zoom_out: Optional[int] = 21,
        debug: Optional[bool] = False,
        **kw,
    ):
        super().__init__(**kw)

        # Setup audio IO
        self.io = AudioIO(path)

        # Set up self attributes
        self.min_frequency = int(min_frequency)

        # calc nyquist frequency
        match max_frequency:
            case "nyquist":
                self.max_frequency = int(round(self.io.sample_rate / 2))
            case float() | int():
                self.max_frequency = max_frequency
            case _:
                raise ValueError(
                    f"Invalid frequency type for max frequency={type(max_frequency)}, {max_frequency}"
                )
        self.num_bins = num_bins
        self.log_scale = log_scale
        self.log_base = log_base
        self.max_amplitude = max_amplitude
        self.fps = fps
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.opacity = opacity
        self.spacing = spacing
        self.colormap = spectra_warm.cmap
        self.downsample = downsampling
        self.height_clipping = height_clipping
        self.min_height = min_height
        self.zoom_out = zoom_out

        self.samples, self.sample_rate = self.io.read()
        self.fft_frames, self.frequencies = self.compute_fft_frames(self.samples, self.sample_rate)

        # create plot axes
        self.axes = Axes(
            x_range=[0, self.max_frequency, 1],
            y_range=[-self.max_amplitude-1, self.max_amplitude+1, 1],
            axis_config={"color": WHITE},
        )

        if debug:
            self.samples = self.samples[:5]
            self.fft_frames = self.fft_frames[:5]
            self.frequencies = self.frequencies[:5]

    def log(self, msg: str, level: str):
        if self.logger is None:
            print(msg)
        else:
            logging_fn = self.logger.__getattribute__(level)
            logging_fn(msg)
    def register(self, name: str, object: Any):
        self.__setattr__(name, object)

    def color_for_amplitude(self, val):
        rgba = list(self.colormap(np.clip(val, 0, 1)))  # Returns (r, g, b, a)
        rgba[-1] = 1.0
        return color.rgb_to_color(rgb=tuple(rgba))  # Drop alpha

    def inverted_color_for_amplitude(self, val):
        inverse = self.colormap.reversed()
        rgba = list(inverse(np.clip(val, 0, 1)))  # Returns (r, g, b, a)
        rgba[-1] = 1.0
        return color.rgb_to_color(rgb=tuple(rgba))  # Drop alpha
    
    def compute_fft_frames(self, samples, rate):
        self.log(f"Computing fft frames", "info")
        num_frames = (len(samples) - self.chunk_size) // self.hop_size
        frames = [
            samples[i * self.hop_size : i * self.hop_size + self.chunk_size] \
             * np.hanning(self.chunk_size) \
            for i in range(num_frames)
        ]
        fft_frames = [np.abs(rfft(frame)) for frame in frames]
        freqs = rfftfreq(self.chunk_size, d=1 / rate)
        self.log(f"Done computing fft frames", "info")
        return fft_frames, freqs
    
    def clip(self, chunk: Any, min: Optional[int]=0):
        return np.clip(chunk, min, self.height_clipping)
    
    def get_bin_indices(self, freqs: np.ndarray, base: Optional[int]=10) -> np.ndarray:
        self.log(f"Computing bin indices", "info")
        if self.log_scale:
            log_bin_edges = (
                np.logspace(
                    np.log(self.min_frequency),
                    np.log(self.max_frequency),
                    num=self.num_bins + 1,
                    base=base,
                )
                * self.zoom_out
            )
            log_bin_indices = np.digitize(freqs, log_bin_edges) - 1  # Map freqs to bins
            self.log(f"Done computing bin indices", "info")
            return log_bin_indices
        else:
            linear_bin_edges = np.linspace(
                self.min_frequency,
                self.max_frequency,
                num=self.num_bins + 1,
            ) * self.zoom_out
            linear_bin_indices = np.digitize(freqs, linear_bin_edges) - 1
            self.log(f"Done computing bin indices", "info")
            return linear_bin_indices
        
    def aggregate_bins(self, bin_indices: np.ndarray, fft_frame: np.ndarray) -> np.ndarray:
        log_magnitudes = np.zeros(self.num_bins)
        for i in range(self.num_bins):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                log_magnitudes[i] = np.mean(fft_frame[bin_mask])
        log_magnitudes /= np.max(log_magnitudes)
        return log_magnitudes

