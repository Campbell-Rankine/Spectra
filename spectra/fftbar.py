from manim import *

config.flush_cache = True
config.media_dir = "/root/spectra/output"

from manim import *
import numpy as np

class FFTBarChart(Scene):
    def construct(self):
        # Generate a time-domain signal: sum of two sine waves
        N = 256
        t = np.linspace(0, 1, N)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)

        # Compute FFT and magnitude
        fft_vals = np.fft.fft(signal)
        fft_mags = np.abs(fft_vals)[:N // 2]  # Only positive frequencies

        # Normalize and select frequencies for display
        freqs = np.fft.fftfreq(N, d=t[1] - t[0])[:N // 2]
        fft_mags /= np.max(fft_mags)  # normalize for bar height

        # Convert to BarChart input
        labels = [f"{int(f)}" for f in freqs[1:30]]
        values = fft_mags[1:30]  # skip the DC component

        # Create a BarChart
        chart = BarChart(
            values,
            bar_names=labels,
            y_range=[0, 1, 0.2],
            bar_width=0.3,
            x_length=10,
            y_length=4,
            axis_config={"include_tip": False},
        )
        chart.to_edge(DOWN)

        # Animate it
        self.play(Create(chart))
        self.wait(2)

        # Optional: animate bar growth
        for bar in chart.bars:
            self.play(bar.animate.set_height(bar.height), run_time=0.05)

        self.wait()

from manim import *
import numpy as np
import librosa

class AudioFFTVisualizer(Scene):
    def construct(self):
        # Load audio (mono, 22050 Hz default)
        audio_path = "audio.wav"
        y, sr = librosa.load(audio_path, sr=None)

        # Parameters
        frame_size = 1024
        hop_size = 256
        n_frames = (len(y) - frame_size) // hop_size
        n_bins = 40  # number of FFT bins to visualize

        # Prepare axes and bar chart
        bars = VGroup()
        max_height = 3.0
        spacing = 0.25
        for i in range(n_bins):
            bar = Rectangle(
                width=0.2, height=0.1, fill_color=BLUE, fill_opacity=0.8, stroke_width=0
            )
            bar.move_to(LEFT * (n_bins/2 - i) * spacing + DOWN * 2)
            bars.add(bar)

        self.add(bars)

        # Precompute all FFT frames
        fft_frames = []
        for i in range(n_frames):
            frame = y[i * hop_size : i * hop_size + frame_size]
            fft = np.abs(np.fft.fft(frame))[:n_bins]
            fft = fft / np.max(fft)  # normalize
            fft_frames.append(fft)

        # Animate the FFT bars
        for fft in fft_frames[:100]:  # limit to 100 frames for speed
            new_bars = VGroup()
            for i, magnitude in enumerate(fft):
                height = max_height * magnitude
                bar = bars[i]
                new_bar = Rectangle(
                    width=bar.width,
                    height=height,
                    fill_color=BLUE,
                    fill_opacity=0.8,
                    stroke_width=0,
                )
                new_bar.move_to(bar.get_center() + UP * (height - bar.height) / 2)
                new_bars.add(new_bar)
            self.play(
                *[bars[i].animate.become(new_bars[i]) for i in range(n_bins)],
                run_time=0.05,
                rate_func=linear,
            )

        self.wait()