import torch
import torch.nn as nn
import torchvision.ops as ops
from collections import OrderedDict

class SALI(nn.Module):

    def __init__(self, spikeRadius, frameRadius, frameStride):
        super(SALI, self).__init__()
        self.spikeRadius = spikeRadius
        self.frameRadius = frameRadius
        self.frameStride = frameStride

        self.timeSize = 2 * self.spikeRadius + 1
        self.k_sizes = [self.timeSize - 2 * i for i in range(3, -1, -1)]
        self.strides = [self.timeSize - 2 * i for i in range(3, -1, -1)]
        self.mask = [i for i in range(3, -1, -1)]

        self.MultiScale()
        self.AttentionBlock()

    def forward(self, spikes): #B, 1, D, H, W

        spikesList = [spikes[:, :, i*7:i*7+self.timeSize]for i in range(5)]
        spikes0 = [sp[:, :, self.mask[0]:self.timeSize - self.mask[0]] for sp in spikesList]
        spikes1 = [sp[:, :, self.mask[1]:self.timeSize - self.mask[1]] for sp in spikesList]
        spikes2 = [sp[:, :, self.mask[2]:self.timeSize - self.mask[2]] for sp in spikesList]
        spikes3 = [sp[:, :, self.mask[3]:self.timeSize - self.mask[3]] for sp in spikesList]

        spikes0 = torch.cat(spikes0, dim=2) #B, 1, D, H, W
        spikes1 = torch.cat(spikes1, dim=2)
        spikes2 = torch.cat(spikes2, dim=2)
        spikes3 = torch.cat(spikes3, dim=2)

        coarseImgs0 = self.Conv3D_mask0(spikes0)   #B, C, D, H, W
        coarseImgs1 = self.Conv3D_mask1(spikes1)
        coarseImgs2 = self.Conv3D_mask2(spikes2)
        coarseImgs3 = self.Conv3D_mask3(spikes3)

        B, C, D, H, W = coarseImgs0.size()
        coarseImgs0 = coarseImgs0.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        coarseImgs1 = coarseImgs1.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        coarseImgs2 = coarseImgs2.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        coarseImgs3 = coarseImgs3.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        coarseImgs0 = self.Conv2D_mask0(coarseImgs0).view(B, D, 1, H, W) #B, D, 1, H, W
        coarseImgs1 = self.Conv2D_mask1(coarseImgs1).view(B, D, 1, H, W)
        coarseImgs2 = self.Conv2D_mask2(coarseImgs2).view(B, D, 1, H, W)
        coarseImgs3 = self.Conv2D_mask3(coarseImgs3).view(B, D, 1, H, W)

        coarseImgs = torch.cat([coarseImgs0, coarseImgs1, coarseImgs2, coarseImgs3], dim=2)
        coarseImgs = coarseImgs.view(B * D, 4, H, W) # B * D, 4, H, W

        # attFeatures = self.Conv2D_att1(coarseImgs) #B * D, C, H, W
        # attFeatures =  self.Conv2D_att2(attFeatures)
        # attFeatures = attFeatures.permute(0, 2, 3, 1).contiguous().view(B * D * H * W, 4)  # B * D * H * W, 4
        # attWeights = self.softMax(attFeatures)
        # attWeights = attWeights.view(B * D, H, W, 4).permute(0, 3, 1, 2).contiguous() # B * D, 4, H, W
        attFeatures = self.Conv2D_att1(coarseImgs)  # B * D, C, H, W
        attWeights = self.Conv2D_att2(attFeatures)

        I_h = torch.multiply(coarseImgs, attWeights)

        coarseImgs = coarseImgs.view(B, D, 4, H, W) # B, D, 4, H, W
        I_h = I_h.view(B, D, 4, H, W) # B, D, 4, H, W

        return coarseImgs, I_h


    def MultiScale(self):

        self.Conv3D_mask0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1,
                                64,
                                kernel_size=(self.k_sizes[0], 3, 3),
                                stride=(self.strides[0], 1, 1),
                                padding=(0, 1, 1),
                                dilation=(1, 1, 1),
                                bias=False)),
            # ('relu1', nn.ReLU(inplace=True)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_mask0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 1, kernel_size=(3,3),stride=1,padding=1, bias=False)),
            # ('norm1', nn.InstanceNorm2d(64)),
            # ('relu1', nn.ReLU(inplace=True)),
            # ('conv2', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))

        self.Conv3D_mask1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1,
                                64,
                                kernel_size=(self.k_sizes[1], 3, 3),
                                stride=(self.strides[1], 1, 1),
                                padding=(0, 1, 1),
                                dilation=(1, 1, 1),
                                bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_mask1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            # ('norm1', nn.InstanceNorm2d(64)),
            # ('relu1', nn.ReLU(inplace=True)),
            # ('conv2', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))

        self.Conv3D_mask2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1,
                                64,
                                kernel_size=(self.k_sizes[2], 3, 3),
                                stride=(self.strides[2], 1, 1),
                                padding=(0, 1, 1),
                                dilation=(1, 1, 1),
                                bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_mask2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            # ('norm1', nn.InstanceNorm2d(64)),
            # ('relu1', nn.ReLU(inplace=True)),
            # ('conv2', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))

        self.Conv3D_mask3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1,
                                64,
                                kernel_size=(self.k_sizes[3], 3, 3),
                                stride=(self.strides[3], 1, 1),
                                padding=(0, 1, 1),
                                dilation=(1, 1, 1),
                                bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_mask3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            # ('norm1', nn.InstanceNorm2d(64)),
            # ('relu1', nn.ReLU(inplace=True)),
            # ('conv2', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))

    def AttentionBlock(self):

        self.Conv2D_att1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_att2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 4, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        # self.softMax = nn.Softmax(dim=1)

