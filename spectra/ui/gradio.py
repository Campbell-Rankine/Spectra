import gradio as gr
import os
from pydub import AudioSegment  # for combining stems


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎶 Audio Stem Splitter")

        audio_input = gr.File(label="Upload Audio", file_types=[".wav", ".mp3"])

        # Fixed slots for 4 stems
        with gr.Row():
            vocals_audio = gr.Audio(label="Vocals", type="filepath")
            vocals_file = gr.File(label="Download Vocals")

        with gr.Row():
            drums_audio = gr.Audio(label="Drums", type="filepath")
            drums_file = gr.File(label="Download Drums")

        with gr.Row():
            bass_audio = gr.Audio(label="Bass", type="filepath")
            bass_file = gr.File(label="Download Bass")

        with gr.Row():
            other_audio = gr.Audio(label="Other", type="filepath")
            other_file = gr.File(label="Download Other")

        stem_selector = gr.CheckboxGroup(
            ["vocals", "drums", "bass", "other"], label="Select stems to combine"
        )
        combine_button = gr.Button("Combine Selected")
        combined_output = gr.File(label="Download Combined Stems")

        # ---- Stem splitting ----
        def on_submit(file):
            from spectra.services.audio.stems import StemSplitter

            splitter = StemSplitter(load_on_init=True)

            labels, audios, files = splitter(file, output_path="./output")
            print(files, audios)
            del splitter

            # Map results back into fixed slots
            mapping = {
                "vocals": (vocals_audio, vocals_file),
                "drums": (drums_audio, drums_file),
                "bass": (bass_audio, bass_file),
                "other": (other_audio, other_file),
            }

            audio_vals = [None, None, None, None]
            file_vals = [None, None, None, None]

            for lbl, audio, f in zip(labels, audios, files):
                if lbl in mapping:
                    idx = ["vocals", "drums", "bass", "other"].index(lbl)
                    audio_vals[idx] = audio
                    file_vals[idx] = f

            return audio_vals + file_vals

        audio_input.change(
            fn=on_submit,
            inputs=audio_input,
            outputs=[
                vocals_audio,
                drums_audio,
                bass_audio,
                other_audio,
                vocals_file,
                drums_file,
                bass_file,
                other_file,
            ],
        )

        # ---- Stem combining ----
        def combine_stems(file, selected):
            if not selected:
                return None

            combined = None
            for stem_name in selected:
                path = f"./output/{stem_name}.wav"
                if os.path.exists(path):
                    seg = AudioSegment.from_file(path)
                    if combined is None:
                        combined = seg
                    else:
                        combined = combined.overlay(seg)

            output_path = "./output/combined.wav"
            if combined:
                combined.export(output_path, format="wav")
                return output_path
            return None

        combine_button.click(
            combine_stems, inputs=[audio_input, stem_selector], outputs=combined_output
        )

    return demo
