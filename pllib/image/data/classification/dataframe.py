import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class DataFrameDataset(Dataset):
    def __init__(self, filenames: pd.DataFrame, root_dir: str, transform=None) -> None:
        self._frame = filenames
        self._root_dir = root_dir
        self._transform = transform

    def __len__(self):
        return len(self._frame)

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        img_name = os.path.join(self._root_dir, self._frame.iloc[index, 0])
        image = Image.open(img_name)
        image = np.array(image)
        label = self._frame.iloc[index, 1]

        if self._transform:
            image = self._transform(image=image)

        return image['image'], label


class DataFrameDataModule(pl.LightningDataModule):
    def __init__(self, train_filenames: pd.DataFrame, root_dir: str, test_filenames: pd.DataFrame = None,
                 batch_size: int = 32, train_transform=None, test_transform=None) -> None:
        super().__init__()
        self.train_frame = train_filenames
        self.test_frame = test_filenames
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = DataFrameDataset(self.train_frame, self.root_dir, self.train_transform)
            if self.test_frame is not None:
                self.test_dataset = DataFrameDataset(self.test_frame, self.root_dir, self.test_transform)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.test_frame is not None:
            return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return super().val_dataloader(*args, **kwargs)
