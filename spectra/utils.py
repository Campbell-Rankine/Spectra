import os
import multiprocessing
import numpy
from moviepy import AudioFileClip, VideoFileClip


class ArrayCoreMap:
    def __init__(self, num_cpus: int):
        self.num_cpus = num_cpus
        # TODO: Write a function to map subsets of the frames to each cpu core


class AudioAttachment:
    def __init__(self, video_file: str, audio_file: str):
        assert os.path.exists(video_file) and os.path.exists(audio_file)
        self.video = VideoFileClip(video_file)
        self.audio = AudioFileClip(audio_file)
        self.output_path = video_file

    def attach(self):
        final_clip = self.video.set_audio(self.audio)

        # Export the final video with the new audio
        final_clip.write_videofile(self.output_path)
