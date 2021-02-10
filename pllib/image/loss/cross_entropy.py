import torch
from torch.nn.modules.module import Module


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        target = target.reshape(input.shape)
        loss = torch.sum(-target * logsoftmax(input), dim=1)
        loss = torch.mean(loss)
        return loss
