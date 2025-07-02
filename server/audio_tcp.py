import pyaudio
import socket
import argparse
import time
import logging
from timeit import default_timer as timer


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    ### - Global Params - ###
    parser.add_argument(
        "-c",
        "--chunk-size",
        dest="chunk_size",
        metavar="chunk size",
        type=int,
        default=2048,
        help="Streaming chunk size",
    )

    parser.add_argument(
        "-r",
        "--sample-rate",
        dest="rate",
        metavar="rate",
        type=int,
        default=44100,
        help="Streaming sample rate",
    )


def init_logger():
    logging.basicConfig(
        filename="./stream.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger: logging.Logger = logging.getLogger("stream")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    # boot server
    start = timer()
    logger = init_logger()

    # load initial data so we can accept a keyboard interrupt at any time
    conn = None
    stream = None

    print("Available input devices:")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            logger.info(
                f"Index {i}: {info['name']} ({info['maxInputChannels']} channels)"
            )

    device_index = None
    try:
        device_index = int(input("Please select an input device index: ..."))
    except KeyboardInterrupt:
        logger.info(f"Stopped!")

    assert not device_index is None
    logger.info(f"Using device={p.get_device_info_by_index(i)['name']}")

    counter = 0
    try:
        while True:
            logger.info(f"Running stream for {counter} seconds")
            counter += 1
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info(f"Stopped!")
    finally:
        end = timer()
        logger.info(
            f"Shutting TCP connection down, stream lasted {round(end-start, 2)} seconds"
        )
        if conn is None or stream is None:
            pass
        else:
            conn.close()
            stream.stop_stream()
            stream.close()
            p.terminate()
