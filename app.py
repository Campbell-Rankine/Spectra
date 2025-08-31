import gradio as gr
import soundfile as sf
import numpy as np
import io
import zipfile
import argparse

from spectra.ui.gradio import ui
from spectra.ui.gradio_local import ui as local_ui

def parse_args():
    parser = argparse.ArgumentParser(description="Run Spectra UI")
    parser.add_argument(
        "--local",
        dest="local",
        default=False,
        type=bool,
        help="Use local file upload interface instead of YouTube URL input",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # TODO: Fix youtube downloader
    # if args.local:
    #     app = local_ui()
    # else:
    #     app = ui()
    app = local_ui()
    app.launch()
