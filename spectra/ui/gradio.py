import gradio as gr
from spectra.utils.audio import process_audio, combine_stems


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎶 Audio Stem Splitter")

        audio_input = gr.File(label="Upload Audio", file_types=[".wav", ".mp3"])

        with gr.Row():
            stem_gallery = gr.Dataset(
                components=["label", "audio", "file"], label="Stems"
            )

        stem_selector = gr.CheckboxGroup(
            ["vocals", "drums", "bass", "other"], label="Select stems to combine"
        )
        combine_button = gr.Button("Combine Selected")
        combined_output = gr.File(label="Download Combined Stems")

        def on_submit(file):
            outputs = process_audio(file)
            labels, audios, files = zip(*outputs)
            return {"label": labels, "audio": audios, "file": files}

        audio_input.change(on_submit, inputs=audio_input, outputs=stem_gallery)
        combine_button.click(
            combine_stems, inputs=[audio_input, stem_selector], outputs=combined_output
        )

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch()
