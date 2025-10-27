import sys
import os
import numpy as np
from timeit import default_timer as timer
import argparse

from manim import config, tempconfig
from server.client import init_logger
from spectra.io.audio import AudioIO
from spectra.services.manim.fft_hist import FFT_Histogram

# only to be used on docker
def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        dest="directory",
        default="./test_data",
        type=str,
        help="Directory path to load song from",
    )

    parser.add_argument(
        "-s",
        "--song", 
        type=str, 
        default="The Chain - Fleetwood Mac (balanced).wav",
        dest="song",
        help="Song name/destination"
    )

    parser.add_argument(
        "-q",
        "--quality", 
        type=str, 
        choices=["low", "medium", "high"],
        default="medium",
        dest="quality",
        help="Video / render quality"
    )

    parser.add_argument(
        "-o",
        "--opacity", 
        type=float, 
        default=0.7,
        dest="opacity",
        help="bar opacity"
    )

    parser.add_argument(
        "-x",
        "--translate-x", 
        type=float, 
        default=-7,
        dest="translate_x",
        help="Bar x coordinate translation. Applied to all"
    )

    parser.add_argument(
        "-y",
        "--translate-y", 
        type=float, 
        default=0,
        dest="translate_y",
        help="Bar y coordinate translation. Applied to all"
    )

    parser.add_argument(
        "-z",
        "--translate-z", 
        type=float, 
        default=1,
        dest="translate_z",
        help="Bar z coordinate translation. Applied to all"
    )

    args = parser.parse_args()
    return args
    

def low_quality(path: str):
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
            "quality": "low_quality",
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
            num_bins=256,
            log_scale=True,
            log_base=np.e,
            translate_x=-7,
            translate_y=-1,
            translate_z=1,
            bar_width=0.05,
            height_clipping=2.0
        )
        scene.register("logger", logger)

        # render
        scene.render()

    # attach audio to video
    # TODO

    end = timer()
    logger.info(f"Rendering Scene Took {round(end-start, 2)} seconds")
    sys.exit(0)

def medium_quality(path: str, opacity: float, translate_x, translate_y, translate_z):
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
            num_bins=512,
            log_scale=True,
            log_base=2,
            translate_x=translate_x,
            translate_y=translate_y,
            translate_z=translate_z,
            bar_width=0.05,
            height_clipping=2.75,
            opacity=opacity,
        )
        scene.register("logger", logger)

        # render
        scene.render()

    # attach audio to video
    end = timer()
    logger.info(f"Rendering Scene Took {round(end-start, 2)} seconds")
    sys.exit(0)

def high_quality(path: str):
    start = timer()
    logger = init_logger()

    output_file = f"{path.split('/')[-1].split('.')[0]}-render"
    output_path = f"./output/{output_file}"
    logger.info(f"Rendering file at: {path}, Output location: {output_file}")

    with tempconfig(
        {
            "quality": "high_quality",
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
            translate_x=-7,
            translate_y=-1,
            translate_z=1,
            bar_width=0.025,
            height_clipping=1
        )
        scene.register("logger", logger)

        # render
        scene.render()

    # attach audio to video
    end = timer()
    logger.info(f"Rendering Scene Took {round(end-start, 2)} seconds")
    sys.exit(0)

if __name__ == "__main__":
    args = parse_cli()

    song_name = args.song

    path_ = f"/root/spectra/test_data/{song_name}"
    path2 = f"{args.directory}/{song_name}"

    if not os.path.exists(path_):
        path = path2
    else:
        path = path_

    match args.quality:
        case "low":
            low_quality(path)

        case "medium":
            medium_quality(path, args.opacity, args.translate_x, args.translate_y, args.translate_z)

        case "high":
            high_quality(path)