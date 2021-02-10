from dotenv import load_dotenv
from multiprocessing import freeze_support
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.optim as optim
import ttach as tta

from pllib.image.loss import MixedLabelLoss
from pllib.image.net.classification import TimmNet
from pllib.image.data.classification import DataFrameDataModule
from pllib.image.augmentations import snapmix


SEED = 1234


def main():
    load_dotenv('cassava.env')
    seed_everything(SEED)

    root_path = os.getenv('ROOT_PATH')
    train_csv_path = root_path + 'train.csv'
    train_root_path = root_path + 'train_images'

    num_classes = int(os.getenv('NUM_CLASSES', 5))
    num_epoch = int(os.getenv('NUM_EPOCH', 10))
    num_folds = int(os.getenv('NUM_FOLDS', 5))
    batch_size = int(os.getenv('BATCH_SIZE'), 16)
    grad_acc = int(os.getenv('GRAD_ACC', 8))

    resize = os.getenv('RESIZE', 224)

    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.ShiftScaleRotate(p=1.0),
        A.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.0, p=1.0, always_apply=False),
        A.RandomResizedCrop(resize, resize, p=1.0, always_apply=True),
        normalize,
        ToTensorV2(p=1.0),
    ], p=1.0)
    test_transform = A.Compose([
        A.Resize(int(resize * 1.5), int(resize * 1.5)),
        normalize,
        ToTensorV2(p=1.0),
    ], p=1.0)
    tta_transform = tta.Compose([
        tta.FiveCrops(resize, resize),
    ])

    criterion = MixedLabelLoss(nn.CrossEntropyLoss(reduction='none'))
    augmentations = [snapmix, ]

    df = pd.read_csv(train_csv_path)
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED).split(df['image_id'], df['label'])
    for _fold, (train, test) in enumerate(folds):
        train = df.iloc[train]
        test = df.iloc[test]
        scheduler = optim.lr_scheduler.CosineAnnealingLR

        model = TimmNet('efficientnet_b3a', num_classes, criterion, learning_rate=1e-3, scheduler=scheduler,
                        n_epoch=num_epoch, eta_min=1e-6, augmentations=augmentations, tta_transform=tta_transform)
        dm = DataFrameDataModule(train, train_root_path, test, batch_size=batch_size,
                                 train_transform=train_transform, test_transform=test_transform)

        mlf_logger = MLFlowLogger(
            experiment_name='cassava',
            tracking_uri='file:./cassava'
        )
        trainer = Trainer(gpus=-1, precision=32, deterministic=True, accumulate_grad_batches=grad_acc,
                          profiler='simple', val_check_interval=1.0, logger=mlf_logger, max_epochs=num_epoch)
        trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    freeze_support()
    main()
