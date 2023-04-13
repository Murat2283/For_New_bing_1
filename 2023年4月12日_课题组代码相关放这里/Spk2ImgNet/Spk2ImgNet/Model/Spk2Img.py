import torch
import torch.nn as nn
import torchvision.ops as ops
from collections import OrderedDict
from Model.SALI import SALI
from Model.FE import FE
from Model.MAIR import MAIR

class Spk2Img(nn.Module):

    def __init__(self, spikeRadius, frameRadius, frameStride):
        super(Spk2Img, self).__init__()
        self.spikeRadius = spikeRadius
        self.frameRadius = frameRadius
        self.frameStride = frameStride

        self.SALI = SALI(self.spikeRadius, self.frameRadius, self.frameRadius)
        self.FE = FE()
        self.MAIR = MAIR()


    def forward(self, spikes): #B, 1, D, H, W

        coarseImg, I_h = self.SALI(spikes) # B, D, 4, H, W; B, D, 4, H, W
        features = self.FE(I_h) #B, D, C, H, W
        preImg = self.MAIR(features) #B, H, W
        return coarseImg, preImg


if __name__ == "__main__":

    x = torch.ones((4,5,4,256,256))
    fe = FE()
    out = fe(x)