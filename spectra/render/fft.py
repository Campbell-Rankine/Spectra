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
        frames_per_second: Optional[int] = 60,
        smoothing: Optional[float] = 0.2,
        **kw,
    ):
        super().__init__(**kw)
        self.io = AudioIO(path)
        self.max_h = max_height
        self.frames_per_second = frames_per_second
        self.smoothing = smoothing
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
        return color_gradient([BLUE, GREEN, RED], int(val))

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
        min_freq = 20
        max_freq = rate / 2
        num_log_bins = 80

        # create log space
        log_freqs = np.logspace(
            np.log10(min_freq), np.log10(max_freq), num=num_log_bins
        )

        def fft_to_log_bins(fft, freqs, log_freqs):
            interp = interp1d(freqs, fft, bounds_error=False, fill_value=0.0)
            return interp(log_freqs)

        fft_frames = [fft_to_log_bins(fft, freqs, log_freqs) for fft in fft_frames]

        # --- Frequency Bins ---
        trackers = [ValueTracker(0.1) for _ in range(num_log_bins)]

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
        for fft in fft_frames[::2]:
            norm = np.clip(fft / np.max(fft), 0, 1)
            for tracker, h in zip(trackers, norm):
                target = np.clip(h * self.max_h, 0.1, self.max_h)
                smoothed = (
                    1 - self.smoothing
                ) * target + self.smoothing * tracker.get_value()
                tracker.set_value(smoothed)
            self.wait(1 / self.frames_per_second)
