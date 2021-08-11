import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class SegmentationLosses:
    def __init__(
        self,
        dataset=None,
        mode='ce',
        ignore_index=255,
    ):            
        self.ignore_index = ignore_index
        self.mode = mode
    
    def set_device(self, cuda):
        self.cuda = cuda
    
    def build_loss(self):
        """Choices: ['ce' or 'focal']"""
        if self.mode == "ce":
            return self.CrossEntropyLoss
        elif self.mode == "focal":
            return self.FocalLoss
        elif self.mode == "nll":
            return self.NLLLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction='mean',
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, _, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction='mean',
        )
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        return loss

    def NLLLoss(self, logit, target):
        criterion = nn.NLLLoss(
            ignore_index=self.ignore_index,
            reduction='mean',
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        return loss
