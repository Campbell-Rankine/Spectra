import pyaudio
import numpy as np
import threading
import collections
import time
import wave
from typing import Optional, List, Any, Dict, Callable


class _AudioStream:
    backend: str = "pyaudio"

    def __init__(
        self,
        rate: Optional[int] = 44100,
        chunk_size: Optional[int] = 1024,
        stream_fmt: Optional[Any] = pyaudio.paInt16,
        num_channels: Optional[int] = 1,
    ):
        self.rate = rate
        self.channels = num_channels
        self.stream_fmt = stream_fmt
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []

    def open(self, input_device_index=None):
        self.stream = self.audio.open(
            format=self.stream_fmt,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=input_device_index,
        )

    def read(self, index: Optional[int] = -1, **kw):
        # get read attrs
        chunk_size = kw.get("chunk_size", self.chunk_size)
        rate = kw.get("rate", self.rate)
        dtype_ = kw.get("dtype", np.int16)

        # init return object
        chunk = None
        if index == -1:
            chunk = self.stream.read(chunk_size)
            self.frames.append(chunk)
        elif index >= 0:
            # handle invalid index case (return streamed chunk)
            if len(self.frames) < index:
                chunk = self.stream.read(chunk_size)
                self.frames.append(chunk)
            else:
                chunk = self.frames[index]

        assert not chunk is None
        return chunk

    def save_wav(self, filename="output.wav"):
        wf = wave.open(filename, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.stream_fmt))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(self.frames))
        wf.close()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


class BufferedLiveAudio(_AudioStream):
    def __init__(
        self,
        rate: Optional[int] = 44100,
        chunk_size: Optional[int] = 1024,
        stream_fmt: Optional[Any] = pyaudio.paInt16,
        num_channels: Optional[int] = 1,
        max_num_seconds: Optional[int] = 120,
    ):
        super().__init__(rate, chunk_size, stream_fmt, num_channels)
        self.running = False
        self._buffer = collections.deque(
            maxlen=(self.rate // self.chunk_size) * max_num_seconds
        )
        self.thread = None

    def open(self, input_device_index=None):
        self.stream = self.audio.open(
            format=self.stream_fmt,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=self.chunk_size,
        )
        self.running = True
        self.thread = threading.Thread(target=self._buffer)

    def _buffer_audio(self):
        while self.running:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
                self._buffer.append(data)
            except Exception as e:
                print("Stream read error:", e)
                break

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.save_wav("/root/spectra/output/recording.wav")


import socket
import numpy as np


class TCPAudioStream:
    def __init__(self, host="0.0.0.0", port=12345, chunk_size=2048):
        self.chunk = chunk_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        print("[CLIENT] Connected to audio server")

    def get_audio_chunk(self):
        data = self.sock.recv(self.chunk * 2)
        return np.frombuffer(data, dtype=np.int16)
