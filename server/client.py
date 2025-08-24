from spectra.io.stream import TCPAudioStream

import logging
import time
import pyaudio


def init_logger():
    logging.basicConfig(
        filename="./output/client.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger: logging.Logger = logging.getLogger("optimisation")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    logger = init_logger()

    stream = TCPAudioStream()

    while True:
        try:
            chunk = stream.get_audio_chunk()
            logger.info(f"Found Audio - Array shape = {chunk.shape}")
        except KeyboardInterrupt:
            break

    logger.info(f"Stopping socket connection.")
    del stream
