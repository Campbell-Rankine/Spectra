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
        opacity: Optional[float] = 0.85,
        spacing: Optional[float] = 0.025,
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

    def construct(self):
        samples, rate = self.io.read()
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        fft_frames, freqs = compute_fft_frames(samples, rate)
        num_bins = len(freqs)

        center = ORIGIN
        radius = self.max_h
        trackers = [ValueTracker(0.1) for _ in range(num_bins)]
        bars = VGroup()

        angle_step = TAU / num_bins

        for i, tracker in enumerate(trackers):
            angle = i * angle_step

            def make_bar(i=i, t=tracker):
                def create_bar():
                    h = t.get_value()
                    bar = Rectangle(width=0.05, height=h)
                    bar.set_fill(self.color_for_amplitude(h / self.max_h), self.opacity)
                    bar.set_stroke(width=0)
                    bar.move_to(
                        center + radius * np.array([np.cos(angle), np.sin(angle), 0])
                    )
                    bar.rotate(angle, about_point=center)
                    bar.shift(
                        bar.get_height()
                        / 2
                        * np.array([np.cos(angle), np.sin(angle), 0])
                    )
                    return bar

                return always_redraw(create_bar)

            bars.add(make_bar())

        self.add(bars)

        for fft in fft_frames[::2]:
            norm = np.clip(fft / np.max(fft), 0, 1)
            for tracker, val in zip(trackers, norm):
                tracker.set_value(val * self.max_h)
            self.wait(1 / self.frames_per_second)
