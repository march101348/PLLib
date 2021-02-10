import torch
from torch import nn


class MixedLabelLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, targets):
        ya, yb, lam_a, lam_b = targets
        loss_a = self.criterion(outputs, ya)
        loss_b = self.criterion(outputs, yb)
        loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        return loss
