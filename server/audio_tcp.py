import pyaudio
import socket
import argparse
import time
import logging
from timeit import default_timer as timer

from buffer import BufferedAudioRecorder


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
        help="Streaming chunk size (default = 2048)",
    )

    parser.add_argument(
        "-r",
        "--sample-rate",
        dest="rate",
        metavar="rate",
        type=int,
        default=44100,
        help="Streaming sample rate (default = 44100hz)",
    )

    parser.add_argument(
        "-a",
        "--address",
        dest="address",
        metavar="address",
        type=str,
        default="0.0.0.0",
        help="Streaming http address (default = 0.0.0.0)",
    )

    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        metavar="port",
        type=int,
        default=12345,
        help="Streaming port (default = 12345)",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        dest="timeout",
        metavar="timeout",
        type=int,
        default=180,
        help="Streaming socket connect timeout (default = 3mins)",
    )

    args = parser.parse_args()
    return args


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
    args = parse_arg()
    logger = init_logger()

    # load initial data so we can accept a keyboard interrupt at any time
    conn = None
    stream = None

    # select audio devices
    logger.info("[SERVER] Available input devices:")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            logger.info(
                f"[SERVER] Index {i}: {info['name']} ({info['maxInputChannels']} channels)"
            )
    device_index = None
    device_index = int(
        input("[SERVER] Please select an input device index: ...\nDevice: ")
    )

    # tcp connection
    assert not device_index is None
    logger.info(f"[SERVER] Using device={p.get_device_info_by_index(i)['name']}")

    # open buffered recording object
    stream = BufferedAudioRecorder(
        rate=args.rate, chunk_size=args.chunk_size, device_index=device_index
    )  # Replace with your line-in index
    stream.open()

    # open socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((args.address, args.port))
    sock.listen(1)

    try:
        print(f"[SERVER] Waiting for connection on port {args.port}...")
        conn, addr = sock.accept()
        print(f"[SERVER] Connected to {addr}")
    except KeyboardInterrupt:
        logger.info(f"Stopping")

    counter = 0
    try:
        while True:
            if stream.live_buffer:
                chunk = stream.get_latest_chunk()
                conn.sendall(chunk)
                counter += 1
    except Exception as e:
        logger.info(
            f"[SERVER] Stopped! Streamed {counter} audio chunks! Exception: {e}"
        )
        conn.close()
        stream.close()
        p.terminate()
    finally:
        end = timer()
        logger.info(
            f"[SERVER] Shutting TCP connection down, stream lasted {round(end-start, 2)} seconds"
        )
        if conn is None or stream is None:
            pass
        else:
            conn.close()
            stream.close()
            p.terminate()
