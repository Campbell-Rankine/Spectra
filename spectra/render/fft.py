import numpy as np
from manim import *
from scipy.fft import rfft, rfftfreq
import os
import logging
from typing import Any, Optional
from scipy.interpolate import interp1d

from spectra.io.audio import AudioIO


def compute_fft_frames(samples, rate, frame_size=2048, hop_size=1024):
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

    def __init__(
        self,
        path: str,
        max_height: Optional[int] = 6,
        frames_per_second: Optional[int] = 30,
        smoothing: Optional[float] = 0.2,
        opacity: Optional[float] = 0.85,
        **kw,
    ):
        super().__init__(**kw)
        self.io = AudioIO(path)
        self.max_h = max_height
        self.frames_per_second = frames_per_second
        self.smoothing = smoothing
        self.opacity = opacity
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
        return color_gradient([RED, BLUE], int(val))

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

        # Create bars for initial frame
        bars = VGroup()
        for i, tracker in enumerate(trackers):
            bar = always_redraw(
                lambda t=tracker, i=i: Rectangle(width=0.02, height=t.get_value())
                .set_fill(self.color_for_amplitude(t.get_value() / self.max_h), 0.9)
                .set_stroke(width=0)
                .align_to(ORIGIN, DOWN)
            )
            bars.add(bar)

        bars.arrange(RIGHT, buff=0.01).move_to(ORIGIN)

        self.add(bars)

        # Animate Frames
        for fft in fft_frames[::2]:  # downsample for speed
            # Normalize heights
            norm = np.clip(fft / np.max(fft), 0, 1)
            for bar, h in zip(bars, norm):
                bar.scale_to_fit_height(h * self.max_h)
                bar.move_to([bar.get_x(), bar.get_y(), 0]).align_to(ORIGIN, DOWN)
            self.wait(1 / self.frames_per_second)  # 30 FPS
