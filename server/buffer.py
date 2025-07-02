import pyaudio
import threading
import collections
import numpy as np


class BufferedAudioRecorder:
    def __init__(
        self,
        rate=44100,
        chunk_size=1024,
        max_buffer_secs=180,
        device_index=None,
        logger=None,
    ):
        self.logger = logger
        self.rate = rate
        self.chunk = chunk_size
        self.format = pyaudio.paInt16
        self.channels = 1
        self.device_index = device_index

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False

        self.raw_frames = []
        self.live_buffer = collections.deque(
            maxlen=(rate // chunk_size) * max_buffer_secs
        )

        self.thread = None

    def log(self, msg: str, level="info"):
        if not self.logger is None:
            match level:
                case "info":
                    self.logger.info(msg)
                case "warn":
                    self.logger.warn(msg)
                case "critical":
                    self.logger.critical(msg)
        else:
            print(msg)

    def open(self):
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
        )
        self.running = True
        self.thread = threading.Thread(target=self._buffer_audio)
        self.thread.start()

    def _buffer_audio(self):
        while self.running:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.raw_frames.append(data)
                self.live_buffer.append(data)
            except Exception as e:
                print("Stream read error:", e)
                break

    def get_latest_chunk(self):
        return self.live_buffer[-1] if self.live_buffer else b"\x00" * self.chunk * 2

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
