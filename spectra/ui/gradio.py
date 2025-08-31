import gradio as gr
import os
from pydub import AudioSegment  # for combining stems


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎶 Audio Stem Splitter")
        with gr.Row():
            url_input = gr.Textbox(label="Paste YouTube URL")
        with gr.Row():
            download_btn = gr.Button("Download & Load")
        with gr.Row():
            audio_output = gr.Audio(label="Downloaded Audio", type="filepath")
        with gr.Row():
            info_output = gr.Textbox(label="Audio Info")

        # Fixed slots for 4 stems
        with gr.Row():
            with gr.Column():
                vocals_audio = gr.Audio(label="Vocals", type="filepath")
                drums_audio = gr.Audio(label="Drums", type="filepath")
            with gr.Column():
                bass_audio = gr.Audio(label="Bass", type="filepath")
                other_audio = gr.Audio(label="Other", type="filepath")

        stem_selector = gr.CheckboxGroup(
            ["vocals", "drums", "bass", "other"], label="Select stems to combine"
        )
        combine_button = gr.Button("Combine Selected")
        combined_output = gr.File(label="Download Combined Stems")

        # ---- Audio URL input ----
        def process_url(url: str):
            from spectra.interfaces.youtube import SpectraYTDLPInterface, SpectraYTubeInterface
            yt_interface = SpectraYTubeInterface(output_dir="./output") # TODO: Cache these downloads
            audio_file, waveform, sample_rate = yt_interface.download_audio(url)
            splitter = StemSplitter(load_on_init=True)

            labels, audios, files = splitter(file, output_path="./output")
            print(files, audios)
            del splitter

            # Map results back into fixed slots
            mapping = {
                "vocals": vocals_audio,
                "drums": drums_audio,
                "bass": bass_audio,
                "other": other_audio,
            }

            audio_vals = [None, None, None, None]

            for lbl, audio, _ in zip(labels, audios, files):
                if lbl in mapping:
                    idx = ["vocals", "drums", "bass", "other"].index(lbl)
                    audio_vals[idx] = audio

            return audio_vals + [audio_file]

        download_btn.click(fn=process_url, inputs=url_input, outputs=[
                vocals_audio,
                drums_audio,
                bass_audio,
                other_audio,
                audio_output,
            ],)

    return demo
