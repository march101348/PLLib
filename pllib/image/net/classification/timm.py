from typing import List, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from ttach.wrappers import ClassificationTTAWrapper


class TimmNet(pl.LightningModule):
    def __init__(self, net_name: str, n_classes: int, criterion: nn.Module, learning_rate=1e-3, scheduler=None,
                 pretrained: bool = True, n_epoch: int = 8, eta_min=1e-6, augmentations=None, tta_transform=None):
        super().__init__()
        backbone = timm.create_model(net_name, pretrained=pretrained)
        n_features = backbone.classifier.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_features, n_classes)
        self.criterion = criterion
        self.scheduler = scheduler
        self.augmentations = augmentations
        self.tta_transform = tta_transform
        self.save_hyperparameters('learning_rate', 'n_epoch', 'eta_min')

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, labels = batch
        if self.augmentations:
            for augmentation in self.augmentations:
                x, labels = augmentation(x, labels, self)
        outputs = self(x)
        loss = self.criterion(outputs, labels)
        return loss

    def training_step_end(self, output):
        self.log('loss', output, logger=True, prog_bar=True)
        output = {'loss': output}
        return output

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss = 0.
        for output in outputs:
            loss += output['loss']
        self.log('loss-ep', loss / len(outputs), logger=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        if self.tta_transform:
            outputs = ClassificationTTAWrapper(self, self.tta_transform)(x)
        else:
            outputs = self(x)
        _, predicted = torch.max(outputs.data, 1)
        if len(labels.size()) == 2:
            _, labels = torch.max(labels.data, 1)
        return labels.size(0), (predicted == labels).sum().detach()

    def validation_step_end(self, output):
        count, correct = output
        return {'count': count, 'correct': correct}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        count = 0
        correct = 0
        for output in outputs:
            cnt, cor = output['count'], output['correct']
            count += cnt
            correct += cor
        self.log('acc-ep', correct / count, logger=True, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if not self.scheduler:
            return optimizer
        scheduler = self.scheduler(optimizer, T_max=self.hparams.n_epoch, eta_min=self.hparams.eta_min)
        return [optimizer], [scheduler]
