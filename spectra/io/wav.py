import numpy as np
from scipy.io import wavfile

class WavReader:
    backend: str = "scipy"