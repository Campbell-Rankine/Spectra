import os
import numpy
from timeit import default_timer as timer

from manim import config, tempconfig
from client import init_logger
from spectra.io.audio import AudioIO
from spectra.utils import AudioAttachment
from spectra.render.fft import FFTFileVisualizer

# only to be used on docker
path_ = "/root/spectra/test_data/jigsaw-3-parts.wav"
path2 = "./test_data/jigsaw-3-parts.wav"

if __name__ == "__main__":
    start = timer()
    logger = init_logger()

    if not os.path.exists(path_):
        path = path2
    else:
        path = path_

    output_file = f"{path.split('/')[-1].split('.')[0]}-render"
    output_path = f"./output/{output_file}"
    logger.info(f"Rendering file at: {path}, Output location: {output_file}")

    with tempconfig(
        {
            "quality": "medium_quality",
            "output_file": output_file,
            "format": "mp4",
            "media_dir": "./output",  # parent folder for videos/images
            "video_dir": "./output/video",  # subfolder for rendered videos
            "images_dir": "./output/frames",
        }
    ):
        # construct scene object
        scene = FFTFileVisualizer(path=path, max_height=4)
        scene.register("logger", logger)

        # render
        scene.render()

    # attach audio to video
    editor = AudioAttachment(output_path + ".mp4", path)
    editor.attach()

    end = timer()
    logger.info(f"Rendering Scene Took {round(end-start, 2)} seconds")

    del editor
