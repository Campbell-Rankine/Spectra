import sys
import os
import numpy as np
from timeit import default_timer as timer

from manim import config, tempconfig
from server.client import init_logger
from spectra.services.render.manim_.fft_hist import FFT_Histogram

# only to be used on docker
song_name = "The Chain - Fleetwood Mac (balanced).wav"
path_ = f"/root/spectra/test_data/{song_name}"
path2 = f"./test_data/{song_name}"

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
        scene = FFT_Histogram(
            path=path,
            num_bins=1024,
            log_scale=True,
            log_base=np.e,
        )
        scene.register("logger", logger)

        # render
        scene.render()

    # attach audio to video

    end = timer()
    logger.info(f"Rendering Scene Took {round(end-start, 2)} seconds")

    os.remove("./output/video/partial_movie_files")
    sys.exit(0)
