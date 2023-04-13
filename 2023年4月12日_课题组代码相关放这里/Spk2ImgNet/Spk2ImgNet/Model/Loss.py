import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Loss():

    def __init__(self):

        self.l1Loss = nn.L1Loss()

    def CoarseLoss(self, coarseImg, gtImgs):
        B, D, H, W = gtImgs.size()
        gtImgs = gtImgs.view(B, D, 1, H, W)
        gtImgs = gtImgs.repeat(1, 1, 4, 1, 1)

        loss = self.l1Loss(coarseImg, gtImgs)

        return loss

    def ReconLoss(self, preImg, gtImg):

        loss = self.l1Loss(preImg, gtImg)

        return loss
