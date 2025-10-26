import numpy as np
from manim import *
from scipy.fft import rfft, rfftfreq
import os
import logging
from typing import Any, Optional
from scipy.interpolate import interp1d
from matplotlib import cm
from manim import color
from tqdm import tqdm

from spectra.io.audio import AudioIO
from spectra.cmaps import spectra, spectra_warm
from spectra.fft import compute_fft_frames, Real1DLogFFT
from spectra.services.render.manim_.transforms import sqrt_transform, exp_transform
from spectra.services.render.manim_.base import _BaseAudioVisualizer

class FFT_Histogram(_BaseAudioVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Extend the scene class by writing the construct function
    def construct(self):
        # calculate number of frames to animate and create databar
        self.num_animations = self.fps * (len(self.samples)/self.sample_rate)

        # compute fft frames
        self.log(
            f"Samples shape: {self.samples.shape}, Sample rate: {self.sample_rate}, Number to Animate: {self.num_animations}",
            "info",
        )

        # Get bins
        bin_indices = self.get_bin_indices(self.frequencies, self.log_base)

        # create manim mobjects
        bars = VGroup()
        curr_frame = ValueTracker(0)

        # === Function to draw FFT bars ===
        def get_fft_bars():
            frame_idx = int(curr_frame.get_value())
            clip_amount = len(self.fft_frames) - 1
            
            frame_idx = np.clip(frame_idx, 0, len(self.fft_frames) - 1)
            mags = self.fft_frames[frame_idx]
            counts, bins = np.histogram(mags, bin_indices)

            bars = VGroup()
            for i, (count, bin) in enumerate(zip(mags, bins)):
                exp_height = exp_transform(count, self.log_base)
                sqrt_height = sqrt_transform(count)

                bar = Rectangle(
                    width=self.bar_width,
                    height=max(sqrt_height, self.min_height),
                    fill_color=self.color_for_amplitude(count),
                    fill_opacity=self.opacity,
                    stroke_width=0,
                ).move_to(((((i-1)*self.bar_width)+self.translate_x), self.translate_y, self.translate_z))
                bars.add(bar)
            self.animation_counter += 1
            return bars
        
        # === Dynamic FFT Bars ===
        bars = always_redraw(get_fft_bars)
        self.add(bars)

        # === Animate over frames ===
        self.play(curr_frame.animate.set_value(len(self.fft_frames)-1), run_time=self.samples.shape[0]/self.sample_rate, rate_func=linear)
        self.wait()


