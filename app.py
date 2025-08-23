import gradio as gr
import soundfile as sf
import numpy as np
import io
import zipfile

from spectra.ui.gradio import ui

if __name__ == "__main__":
    app = ui()
    app.launch()
