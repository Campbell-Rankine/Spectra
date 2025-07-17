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


def compute_fft_frames(samples, rate, frame_size=1024, hop_size=512):
    num_frames = (len(samples) - frame_size) // hop_size
    frames = [
        samples[i * hop_size : i * hop_size + frame_size] * np.hanning(frame_size)
        for i in range(num_frames)
    ]
    fft_frames = [np.abs(rfft(frame)) for frame in frames]
    freqs = rfftfreq(frame_size, d=1 / rate)
    return fft_frames, freqs


class FFTFileRadialVisualizer(Scene):
    backend: str = "manim"
    logger: logging.Logger = None
    max_amp: float = None

    def __init__(
        self,
        path: str,
        max_height: Optional[int] = 6,
        frames_per_second: Optional[int] = 20,
        smoothing: Optional[float] = 0.2,
        opacity: Optional[float] = 0.8,
        spacing: Optional[float] = 0.05,
        downsampling: Optional[int] = 2,
        height_clipping: Optional[int] = 3.0,
        min_height: Optional[float] = 0.01,
        **kw,
    ):
        super().__init__(**kw)
        self.io = AudioIO(path)
        self.max_h = max_height
        self.frames_per_second = frames_per_second
        self.smoothing = smoothing
        self.opacity = opacity
        self.spacing = spacing
        self.colormap = cm.get_cmap("plasma")
        self.downsample = downsampling
        self.height_clipping = height_clipping
        self.animate_counter: int = 0
        self.min_height = min_height

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
        samples, rate = self.io.read()
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        fft_frames, freqs = compute_fft_frames(samples, rate)
        num_bins = len(freqs)

        min_freq = 10  # Minimum frequency of interest (Hz)
        max_freq = 21000  # Nyquist frequency
        num_log_bins = 512  # Set number of perceptual bins

        log_bin_edges = (
            np.logspace(
                np.log2(min_freq),
                np.log2(max_freq),
                num=num_log_bins + 1,
                base=2,
            )
            * 30
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

        center = ORIGIN
        radius = self.max_h
        trackers = [ValueTracker(0.1) for _ in range(num_bins)]
        bars = VGroup()

        angle_step = 360 / (num_log_bins)

        for i, tracker in enumerate(trackers):
            angle = i * angle_step

            bar = always_redraw(
                lambda t=tracker, i=i: Rectangle(width=0.05, height=t.get_value())
                .set_fill(
                    self.color_for_amplitude(t.get_value() / self.max_h),
                    self.opacity,
                )
                .set_stroke(width=0)
            )

            bars.add(bar)
        bars.arrange(RIGHT, buff=1e-7)
        bars.move_to((ORIGIN))
        self.add(bars)

        for fft in fft_frames[:: self.downsample]:  # downsample for speed
            # Normalize heights
            log_fft = aggregate_log_bins(fft)
            norm = np.clip(log_fft, 0, self.height_clipping)
            for idx, (bar, h) in enumerate(zip(bars, norm)):
                angle = (idx + 1) * angle_step
                angle = angle + 90
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                bar.stretch_to_fit_height(max(h, self.min_height))
                bar.align_to([x, y, 0])
                # bar.stretch_to_fit_width(0.05)
                bar.rotate(angle, about_point=bar.get_center())
                bar.move_to(
                    center + radius * np.array([np.cos(angle), np.sin(angle), 0])
                )

                bar.set_fill(self.color_for_amplitude(h))

            self.wait(1 / self.frames_per_second)
