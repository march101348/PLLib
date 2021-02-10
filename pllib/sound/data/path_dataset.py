import librosa
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchaudio
import pytorch_lightning as pl

from pllib.sound.utils import SoundSpecTransmitter, SoundIO


class AudioTransformDataset(Dataset):
    def __init__(self, filepath: str, span: int, transmitter: SoundSpecTransmitter,
                 sound_io: SoundIO, transforms=None):
        super().__init__()
        wave = sound_io.load(filepath)
        spectrogram = transmitter.encode(wave)
        self.spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        self.span = span
        self.transforms = transforms

    def __len__(self):
        return self.spectrogram.size()[-1] // self.span

    def __getitem__(self, idx):
        ret = self.spectrogram[:, idx*self.span:(idx+1)*self.span]
        return torch.reshape(ret, (1, *ret.size()))


class PairedAudioDataset(Dataset):
    def __init__(self, filepath1: str, filepath2: str, span: int, transmitter: SoundSpecTransmitter,
                 sound_io: SoundIO, transforms=None):
        super(PairedAudioDataset, self).__init__()
        self.dataset1 = AudioTransformDataset(filepath1, span=span,
                                              transmitter=transmitter, sound_io=sound_io, transforms=transforms)
        self.dataset2 = AudioTransformDataset(filepath2, span=span,
                                              transmitter=transmitter, sound_io=sound_io, transforms=transforms)

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        return self.dataset1[idx % len(self.dataset1)], self.dataset2[idx % len(self.dataset2)]


class PairedAudioDataModule(pl.LightningDataModule):
    def __init__(self, filepath1: str, filepath2: str, span: int, transmitter: SoundSpecTransmitter,
                 sound_io: SoundIO, batch_size: int, transforms=None):
        super(PairedAudioDataModule, self).__init__()
        self.dataset = PairedAudioDataset(filepath1, filepath2, span=span,
                                          transmitter=transmitter, sound_io=sound_io, transforms=transforms)
        self.batch_size = batch_size

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
