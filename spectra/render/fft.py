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
        **kw,
    ):
        super().__init__(**kw)
        self.io = AudioIO(path)
        self.max_h = max_height
        self.frames_per_second = frames_per_second
        self.smoothing = smoothing
        self.opacity = opacity
        self.spacing = spacing
        self.colormap = cm.get_cmap("managua")
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
                bar.scale_to_fit_height(h)
                bar.set_fill(self.color_for_amplitude(h))
                bar.move_to([bar.get_x(), bar.get_y(), 0]).align_to(ORIGIN, DOWN)
            self.wait(1 / self.frames_per_second)  # 30 FPS


class FFTFileStereoVisualizer(Scene):
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
        self.io = AudioIO(path, mono=False)
        self.max_h = max_height
        self.frames_per_second = frames_per_second
        self.smoothing = smoothing
        self.opacity = opacity
        self.spacing = spacing
        self.colormap_l = cm.get_cmap("plasma")
        self.colormap_r = cm.get_cmap("ocean")
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

    def left_color_for_amplitude(self, val):
        rgba = list(self.colormap_l(np.clip(val, 0, 1)))  # Returns (r, g, b, a)
        rgba[-1] = 1.0
        return color.rgb_to_color(rgb=tuple(rgba))  # Drop alpha

    def right_color_for_amplitude(self, val):
        rgba = list(self.colormap_r(np.clip(val, 0, 1)))  # Returns (r, g, b, a)
        rgba[-1] = 1.0
        return color.rgb_to_color(rgb=tuple(rgba))  # Drop alpha

    def merge_frames(self, left, right):
        result = []
        for i in range(len(left)):
            result.append(left[i])
            result.append(right[i])
        return np.array(result, dtype=np.float32)

    def construct(self):
        # Load and process audio
        self.animate_counter += 1
        samples, rate = self.io.read()
        left_samples = samples[:, 0]
        right_samples = samples[:, 1]
        assert left_samples.ndim == 1 and right_samples.ndim == 1

        # calculate fft
        left_fft_frames, freqs_left = compute_fft_frames(left_samples, rate)
        right_fft_frames, freqs_right = compute_fft_frames(right_samples, rate)
        self.log(
            f"Samples shape: {samples.shape}, Sample rate: {rate}, Number of Animations: {self.animate_counter}",
            "info",
        )
        fft_frames = self.merge_frames(left_fft_frames, right_fft_frames)
        freqs = self.merge_frames(freqs_left, freqs_right)

        # convert audio samples to log space for better alignment with human hearing
        num_bins = len(freqs)

        # --- Frequency Bins ---
        trackers = [ValueTracker(0.1) for _ in range(num_bins)]
        self.max_amp = np.max(fft_frames)

        # Create bars for initial frame
        bars = VGroup()
        for i, tracker in enumerate(trackers):
            if i % 2 == 0:
                bar = always_redraw(
                    lambda t=tracker, i=i: Rectangle(width=0.01, height=t.get_value())
                    .set_fill(
                        self.left_color_for_amplitude(t.get_value() / self.max_h),
                        self.opacity,
                    )
                    .set_stroke(width=0)
                    .align_to(ORIGIN, DOWN)
                )
            else:
                bar = always_redraw(
                    lambda t=tracker, i=i: Rectangle(width=0.01, height=t.get_value())
                    .set_fill(
                        self.right_color_for_amplitude(t.get_value() / self.max_h),
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
            for idx, (bar, h) in enumerate(zip(bars, norm)):
                bar.scale_to_fit_height(h)
                if idx % 2 == 0:
                    bar.set_fill(self.left_color_for_amplitude(h))
                else:
                    bar.set_fill(self.right_color_for_amplitude(h))
                bar.move_to([bar.get_x(), bar.get_y(), 0]).align_to(ORIGIN, DOWN)
            self.wait(1 / self.frames_per_second)  # 30 FPS
