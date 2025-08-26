import os
import torchaudio
import numpy as np

import gradio as gr
from spectra.io.audio import AudioIO


def cache_stem_request(path_to_audio: str, output_path: str) -> tuple[int, np.ndarray]:
    song_name = path_to_audio.split("/")[-1].split(".")[0]
    audios = []
    files = []
    if os.path.exists(f"{output_path}/{song_name}"):
        for file in os.listdir(f"{output_path}/{song_name}"):
            print(file)
            wav = gr.Audio(f"{output_path}/{song_name}/{file}")
            audios.append(wav)
            files.append(f"{output_path}/{song_name}/{file}")
    return audios, files
