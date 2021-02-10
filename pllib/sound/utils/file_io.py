import numpy as np
import librosa
import soundfile


class SoundIO:
    def __init__(self, sampling_rate: int):
        self.sr = sampling_rate

    def load(self, path: str):
        wave, _ = librosa.load(path, sr=self.sr)
        return wave

    def save(self, path: str, wave: np.ndarray):
        soundfile.write(path, wave, samplerate=self.sr)
