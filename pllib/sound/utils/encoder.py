import numpy as np
import librosa
import soundfile
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio


class SoundSpecTransmitter:
    def __init__(self, n_fft: int = 400, hop_length: int = 160, to_db: bool = False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.to_db = to_db

    def encode(self, wave: np.ndarray) -> np.ndarray:
        spec = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.hop_length)
        if self.to_db:
            spec = librosa.amplitude_to_db(spec)
        self.mean = np.mean(spec)
        self.std = np.std(spec)
        return (spec - self.mean) / self.std

    def decode(self, spec: np.ndarray) -> np.ndarray:
        spec = spec * self.std + self.mean
        if self.to_db:
            spec = librosa.db_to_amplitude(spec)
        wave = librosa.griffinlim(spec, hop_length=self.hop_length)
        return wave
