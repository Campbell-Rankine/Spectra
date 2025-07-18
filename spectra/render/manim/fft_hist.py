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


class FFTFileVisualizer(Scene):
    backend: str = "manim"
    logger: logging.Logger = None
    max_amp: float = None

    def __init__(
        self,
        path: str,
        max_height: Optional[int] = 6,
        frames_per_second: Optional[int] = 20,
        chunk_size: Optional[int] = 1024,
        hop_size: Optional[int] = 512,
        smoothing: Optional[float] = 0.2,
        opacity: Optional[float] = 0.8,
        spacing: Optional[float] = 0.05,
        downsampling: Optional[int] = 2,
        height_clipping: Optional[int] = 3.0,
        min_height: Optional[float] = 0.01,
        zoom_out: Optional[int] = 21,
        **kw,
    ):
        super().__init__(**kw)
        self.io = AudioIO(path)
        self.max_h = max_height
        self.frames_per_second = frames_per_second
        self.smoothing = smoothing
        self.opacity = opacity
        self.spacing = spacing
        self.colormap = spectra_warm.cmap
        self.downsample = downsampling
        self.height_clipping = height_clipping
        self.animate_counter: int = 0
        self.min_height = min_height
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.zoom_out = zoom_out
        self.fft = self._create_fft_(**kw)

    def _create_fft_(self, **kw):
        shift_frames = kw.get("shift_frames", False)
        log_base = kw.get("log_base", np.e)
        min_freq = kw.get("min_frequency", 10)
        max_freq = kw.get("max_frequency", "nyquist")
        num_log_bins = kw.get("num_log_bins", 512)
        transformer = Real1DLogFFT(
            44100,
            self.chunk_size,
            self.hop_size,
            shift_frames,
            self.zoom_out,
            log_base,
            min_freq,
            max_freq,
            num_log_bins,
        )
        return transformer

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

    def construct(self):
        # Load and process audio
        self.animate_counter += 1
        samples, rate = self.io.read()

        # calculate fft
        fft_frames, freqs = compute_fft_frames(samples, rate)
        self.log(
            f"Samples shape: {samples.shape}, Sample rate: {rate}, Number of Animations: {self.animate_counter}",
            "info",
        )

        # Convert linear frequency bins to log bins
        min_freq = 10  # Minimum frequency of interest (Hz)
        max_freq = 22050  # Nyquist frequency
        num_log_bins = 1024  # Set number of perceptual bins

        log_bin_edges = np.logspace(
            np.log(min_freq), np.log(max_freq), num=num_log_bins + 1
        )
        log_bin_indices = np.digitize(freqs, log_bin_edges) - 1  # Map freqs to bins

        # For each frame, we will average magnitudes in each log bin
        def aggregate_log_bins(fft_frame):
            log_magnitudes = np.zeros(num_log_bins)
            for i in range(num_log_bins):
                bin_mask = log_bin_indices == i
                if np.any(bin_mask):
                    log_magnitudes[i] = np.mean(fft_frame[bin_mask])
            log_magnitudes /= np.max(log_magnitudes)
            return log_magnitudes

        # --- Frequency Bins ---
        trackers = [ValueTracker(0.1) for _ in range(self.fft.num_log_bins)]

        # Create bars for initial frame
        bars = VGroup()
        for i, tracker in enumerate(trackers):
            bar = always_redraw(
                lambda t=tracker, i=i: Rectangle(width=0.05, height=t.get_value())
                .set_fill(
                    self.color_for_amplitude(t.get_value() / self.max_h),
                    self.opacity,
                )
                .set_stroke(width=0)
                .align_to(ORIGIN, DOWN)
                .stretch_to_fit_height(t.get_value())
            )
            bars.add(bar)
        self.add(bars)
        bars.arrange(RIGHT, buff=1e-7)

        bars.move_to((ORIGIN))
        bars.move_to((6 * RIGHT))

        # Animate Frames
        for fft in fft_frames:  # downsample for speed
            log_fft = aggregate_log_bins(fft)
            norm = np.clip(log_fft, 0, self.height_clipping)
            for idx, (bar, h) in enumerate(zip(bars, norm)):
                # TODO: convert this if else statement to be an envelope
                if (
                    idx > num_log_bins - (num_log_bins / 2)
                    or idx < num_log_bins + (num_log_bins / 2)
                    and h * 3.775 < self.height_clipping
                ):
                    h = h * 2
                if idx < num_log_bins / 4 or h > self.height_clipping - 1:
                    h = h / 2
                bar.stretch_to_fit_height(max(h, self.min_height))
                bar.move_to([bar.get_x(), bar.get_y(), 0])
                bar.set_fill(self.color_for_amplitude(h))

            self.wait(1 / self.frames_per_second)


class FFTFilePlaneVisualizer(Scene):
    backend: str = "manim"
    logger: logging.Logger = None
    max_amp: float = None

    def __init__(
        self,
        path: str,
        max_height: Optional[int] = 6,
        frames_per_second: Optional[int] = 20,
        smoothing: Optional[float] = 0.2,
        opacity: Optional[float] = 0.85,
        spacing: Optional[float] = 0.025,
        downsampling: Optional[int] = 2,
        **kw,
    ):
        super().__init__(**kw)
        self.io = AudioIO(path)
        self.max_h = max_height
        self.frames_per_second = frames_per_second
        self.smoothing = smoothing
        self.opacity = opacity
        self.spacing = spacing
        self.colormap = cm.get_cmap("nipy_spectral")
        self.downsample = downsampling
        self.animate_counter: int = 0

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

    def construct(self):
        # Load and process audio
        self.animate_counter += 1
        samples, rate = self.io.read()

        # convert to mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # calculate fft
        fft_frames, freqs = compute_fft_frames(samples, rate)
        self.log(
            f"Samples shape: {samples.shape}, Sample rate: {rate}, Number of Animations: {self.animate_counter}",
            "info",
        )

        # convert audio samples to log space for better alignment with human hearing
        num_bins = len(freqs)

        # --- Frequency Bins ---
        trackers = [ValueTracker(0.1) for _ in range(num_bins)]
        self.max_amp = np.max(fft_frames)

        # Create bars for initial frame
        bars = VGroup()
        for i, tracker in enumerate(trackers):
            bar = always_redraw(
                lambda t=tracker, i=i: Rectangle(width=0.01, height=t.get_value())
                .set_fill(
                    self.color_for_amplitude(t.get_value() / self.max_h),
                    self.opacity,
                )
                .set_stroke(width=0)
                .align_to(ORIGIN, DOWN)
            )
            bars.add(bar)

        bars.arrange(RIGHT, buff=0.01)

        bars.move_to(ORIGIN)
        self.add(bars)

        # Animate Frames
        for fft in fft_frames[:: self.downsample]:  # downsample for speed
            # Normalize heights
            norm = np.clip(fft / np.max(fft), 0, 1)
            for bar, h in zip(bars, norm):
                bar.scale_to_fit_height(self.max_h)
                bar.set_fill(self.color_for_amplitude(h))
                bar.move_to([bar.get_x(), bar.get_y(), 0]).align_to(ORIGIN, DOWN)
            self.wait(1 / self.frames_per_second)  # 30 FPS
