import os
import torch
# from torchvision import transforms
from torch.utils import data
import numpy as np
from PIL import Image
import cv2
import random


from DataProcess.RedsDataset import RedsDataset
from DataProcess.NfsDataset import NfsDataset
from DataProcess.EventsDataset import EventsDataset
from DataProcess.LoadSpike import LoadSpike, load_spike_raw

class Dataset(data.Dataset):

    def __init__(self, pathList, dataType, spikeRadius, frameRadius, frameStride):

        self.pathList = pathList
        self.dataType = dataType
        self.spikeRadius = spikeRadius
        self.frameRadius = frameRadius
        self.frameStride = frameStride

        if self.dataType == "train" or self.dataType == "valid_real" or \
           self.dataType == "valid" or self.dataType == "test":

            self.GetItem = self.GetFromTraOrVal
        else:
            self.GetItem = self.GetFromTest

        #Random Rotation
        if self.dataType == "train":
            self.choice = [0, 1, 2, 3]
        else:
            self.choice = [0]



    def __getitem__(self, index):

        spSeq, gtFrames, edge = self.GetItem(index)

        return spSeq, gtFrames, edge

    def __len__(self):

        return len(self.pathList)

    def GetFromTraOrVal(self, index):

        path = self.pathList[index]
        spSeq, gtFrames = LoadSpike(path)
        # print(path, spSeq.shape, gtFrames.shape)
        spLen, _, _ = spSeq.shape
        gtLen, _, _ = gtFrames.shape
        spCenter = spLen // 2
        gtCenter = gtLen // 2

        # cLen = 2 * self.spikeRadius + 2 * self.frameRadius * self.frameStride + 1
        spLeft, spRight = (spCenter - self.spikeRadius - self.frameRadius * self.frameStride,
                           spCenter + self.spikeRadius + self.frameRadius * self.frameStride + 1)
        spSeq = spSeq[spLeft:spRight]

        gtLeft, gtRight = gtCenter - self.frameRadius * self.frameStride, gtCenter + self.frameRadius * self.frameStride + 1
        gtFrames = gtFrames[gtLeft:gtRight:self.frameStride]

        new_gtLen, _, _ = gtFrames.shape
        new_gtCenter = new_gtLen // 2
        centerImg = gtFrames[new_gtCenter]
        edge = cv2.Canny(centerImg, 96, 176).astype(np.uint8)
        edge[edge == 255.] = 1.
        edge[edge == 0.] = 0.

        # spSeq = np.pad(spSeq, ((0, 0), (3, 3), (0, 0)), mode='constant')
        spSeq = spSeq.astype(float) * 2 - 1
        # spSeq = spSeq.astype(float)

        # gtFrames = np.pad(gtFrames, ((0, 0), (3, 3), (0, 0)), mode='constant')
        gtFrames = gtFrames.astype(float) / 255. * 2.0 - 1.


        spSeq = torch.FloatTensor(spSeq)
        gtFrames = torch.FloatTensor(gtFrames)
        edge = torch.FloatTensor(edge)

        choice = random.choice(self.choice)
        spSeq = torch.rot90(spSeq, choice, dims=(1,2))
        gtFrames =torch.rot90(gtFrames, choice, dims=(1,2))
        edge = torch.rot90(edge, choice, dims=(0,1))

        return spSeq, gtFrames, edge

    def GetFromTest(self, index):
        path = self.pathList[index]
        spikePath = path[0]
        imgPath = path[1]
        spSeq = load_spike_raw(spikePath)
        gtFrame = Image.open(imgPath)
        gtFrame = gtFrame.convert('L')
        gtFrame = np.array(gtFrame)

        spLen, _, _ = spSeq.shape
        spCenter = spLen // 2

        cLen = 2 * self.spikeRadius + 2 * self.frameRadius * self.frameStride + 1
        spLeft, spRight = (spCenter - self.spikeRadius - self.frameRadius * self.frameStride,
                           spCenter + self.spikeRadius + self.frameRadius * self.frameStride + 1)
        spSeq = spSeq[spLeft:spRight]

        # edge = cv2.Canny(gtFrame, 96, 176).astype(np.uint8)
        # edge[edge==255.] = 1.
        # edge[edge==0.] = 0.
        edge = np.zeros((256, 400))

        spSeq = np.pad(spSeq, ((0, 0), (3, 3), (0, 0)), mode='constant')
        spSeq = spSeq.astype(float) * 2 - 1
        # spSeq = spSeq.astype(float)

        gtFrame = np.pad(gtFrame, ((3, 3), (0, 0)), mode='constant')
        gtFrame = gtFrame.astype(float) / 255. * 2.0 - 1.



        spSeq = torch.FloatTensor(spSeq)
        gtFrame = torch.FloatTensor(gtFrame).unsqueeze(axis=0)
        edge = torch.FloatTensor(edge)

        return spSeq, gtFrame, edge



class DataContainer():

    def __init__(self, dataName='car', dataType='train', spikeRadius=16, frameRadius=16, frameStride=1, batchSize=128, numWorks=0):

        self.dataName = dataName
        self.dataType = dataType
        self.spikeRadius = spikeRadius
        self.frameRadius = frameRadius
        self.frameStride = frameStride
        self.batchSize = batchSize
        self.numWorks = numWorks

        self.__GetData()

    def __GetData(self):

        dataset = None

        if self.dataName == "reds":
            dataset = RedsDataset(type=self.dataType)
        if self.dataName == "nfs":
            dataset = NfsDataset(type=self.dataType)
        if self.dataName == "events":
            dataset = EventsDataset(type=self.dataType)

        self.pathList = dataset.GetData()

    def GetLoader(self):

        dataset = Dataset(self.pathList, self.dataType, self.spikeRadius, self.frameRadius, self.frameStride)
        # for pa in self.pathList:
        #     print(pa)
        dataLoader = None
        if self.dataType == "train":
            dataLoader = data.DataLoader(dataset, batch_size=self.batchSize, shuffle=True,
                                         num_workers=self.numWorks, pin_memory=False)
        else:
            dataLoader = data.DataLoader(dataset, batch_size=self.batchSize, shuffle=False,
                                         num_workers=self.numWorks, pin_memory=False)

        return dataLoader

if __name__ == "__main__":

    dataContainer = DataContainer(dataName='reds', dataType='train', spikeRadius=16, frameRadius=2, frameStride=8, batchSize=6)

    dataLoader = dataContainer.GetLoader()

    for spSeq, gtFrames, _ in dataLoader:
        pass
        # for pa in path:
        #     print(pa)


