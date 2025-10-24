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
from spectra.services.render.manim_.base import _BaseAudioVisualizer

class FFT_Histogram(_BaseAudioVisualizer):
    # Extend the scene class by writing the construct function
    def construct(self):
        # load audio

        # calculate number of frames to animate and create databar
        self.num_animations = self.fps * (len(self.samples)/self.sample_rate)

        # compute fft frames
        self.log(
            f"Samples shape: {self.samples.shape}, Sample rate: {self.sample_rate}, Number to Animate: {self.num_animations}",
            "info",
        )

        # Get bins
        bin_indices = self.get_bin_indices(self.frequencies, self.log_base)

        # map histogram
        current_frame = self.fft_frames[self.animation_counter]
        counts, bins = np.histogram(current_frame, bin_indices)
        counts = (self.max_amplitude * counts) / np.max(counts) # normalize counts

        bars = VGroup()

        trackers = [ValueTracker(self.min_height) for _ in range(self.num_bins)]

        # Plot only the right side and mirror to the left
        for i in range(len(bins) - 1):
            # calculate rectangle coord vars
            x_left = bins[i]
            x_right = bins[i + 1]
            bin_height = max(counts[i], self.min_height)
            trackers[i] = bin_height
            bin_colour = self.color_for_amplitude(bin_height)
            tracker = trackers[i]

            # create upward facing bar
            bar = always_redraw(
                lambda t=tracker, i=i: Rectangle(
                    width=(x_right - x_left), 
                    height=t.get_value(),
                    fill_color=bin_colour,
                    fill_opacity=self.opacity,
                    stroke_width=0
                ).move_to((x_right-x_left)/2 * RIGHT)
            )
            # ).next_to(self.axes.c2p((x_left + x_right) / 2, 0), UP, buff=0)

            # Mirror the bar across y=0 (x -> -x)
            bar_left = bar.copy().flip(UP).flip(LEFT)
            bar_left.set_fill(self.color_for_amplitude(self.max_amplitude/bin_height)) # colour inversion
            bars.add(bar, bar_left)
        self.add(bars)
        # animate frames
        seconds_rendered = 0
        for idx, frame in enumerate(self.fft_frames):
            # get frame data
            current_frame = self.fft_frames[self.animation_counter]
            counts, bins = np.histogram(current_frame, bin_indices)
            counts = (self.max_amplitude * counts) / np.max(counts) # normalize counts
            # Log number of seconds rendered
            if self.animation_counter % int(1/self.fps) == 0:
                seconds_rendered += 1
                self.log(f"Rendered {seconds_rendered}s of audio")
            
            # iterate over bars and update values
            for i, tracker in enumerate(trackers):
                # update trackers
                height = max(counts[i], self.min_height)
                tracker = height
            for i, (bar, height) in enumerate(zip(bars, counts)):


            # increment frame counter
            self.animation_counter += 1
            self.wait(1 / self.fps)



