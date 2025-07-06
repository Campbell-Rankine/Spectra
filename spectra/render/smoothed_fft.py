import numpy as np
from manim import *
from scipy.fft import rfft, rfftfreq
import os
import logging
from typing import Any, Optional
from scipy.interpolate import interp1d
from matplotlib import cm
from manim import Color

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


class FFTFileVisualizer(Scene):
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
        interpolation_frames: Optional[int] = 3,
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
        self.interpolation_frames = interpolation_frames
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
        rgba = self.colormap(np.clip(val / self.max_amp, 0, 1))  # Returns (r, g, b, a)
        return Color(rgb=rgba[:3])  # Drop alpha

    def inverted_color_for_amplitude(self, val):
        inverse = self.colormap.reversed()
        rgba = inverse(np.clip(val / self.max_amp, 0, 1))  # Returns (r, g, b, a)
        return Color(rgb=rgba[:3])  # Drop alpha

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

        bars.move_to(ORIGIN)
        self.add(bars)

        center_line = Line(LEFT * 10, RIGHT * 10, color=WHITE, stroke_opacity=0.2)
        self.add(center_line)

        # Smoothing: interpolate between frames
        smoothed_fft_frames = []
        for i in range(len(fft_frames) - 1):
            current = fft_frames[i]
            next_frame = fft_frames[i + 1]
            # Interpolate N steps between frames
            for alpha in np.linspace(
                0, 1, self.interpolation_frames
            ):  # change to control smoothness
                interp = (1 - alpha) * current + alpha * next_frame
                smoothed_fft_frames.append(interp)

        # Animate the interpolated frames
        for fft in smoothed_fft_frames[:: self.downsample]:  # downsample for speed
            # Normalize heights
            norm = np.clip(fft / np.max(fft), 0, 1)
            for bar, h in zip(bars, norm):
                bar.scale_to_fit_height(h)
                bar.set_fill(self.color_for_amplitude(h))
                bar.move_to([bar.get_x(), bar.get_y(), 0]).align_to(ORIGIN, DOWN)
            self.wait(1 / self.frames_per_second)  # 30 FPS
