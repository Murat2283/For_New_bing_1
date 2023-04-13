import torch
import torch.nn as nn
from collections import OrderedDict


class FE(nn.Module):

    def __init__(self):
        super(FE, self).__init__()

        self.ResidualModule(15)
        # for name in self.modules():
        #     print(name)
    def forward(self, I_h): #B, D, 4, H, W

        B, D, C, H, W = I_h.size()
        I_h = I_h.view(B * D, C, H, W)

        x = self.Cov2D_1(I_h)
        y = x.clone()
        for residual, relu in zip(self.residualModuleList, self.residualReluList):
            x1 = residual(y)
            x1 = y + x1
            x1 = relu(x1)
            y = x1

        x3 = x + y
        BD, C, H, W = x3.size()
        x3 = x3.view(B, D, C, H, W)
        return x3


    def ResidualModule(self, num=15):

        self.Cov2D_1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))

        self.residualModuleList = nn.ModuleList()
        self.residualReluList = nn.ModuleList()
        for i in range(num):
            residual, residual_relu = self.ResiualBlock()
            self.residualModuleList.append(residual)
            self.residualReluList.append(residual_relu)


    def ResiualBlock(self):

        residualBlock = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))

        relu = nn.Sequential(OrderedDict([
            ('relu1', nn.ReLU(inplace=True)),
        ]))

        return residualBlock, relu


if __name__ == "__main__":

    x = torch.ones((4,5,4,256,256))
    fe = FE()
    out = fe(x)