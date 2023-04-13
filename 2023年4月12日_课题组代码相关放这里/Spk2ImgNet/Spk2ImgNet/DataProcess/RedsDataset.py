import os
import numpy as np


class RedsDataset():

    def __init__(self, type='train'):

        self.type = type
        # self.rootPath = "/home/storage1/Dataset/REDS/Spikes_for_Spk2Img"
        self.rootPath = "/home/share1/yangchen/Dataset/REDS/Spikes_for_Spk2Img"

    def GetData(self):
        if self.type == "train":
            return self.__GetTrainData()
        if self.type == "valid_real":
            return self.__GetValidRealData()
        if self.type == "test_real":
            return self.__GetTestRealData()

        if self.type == "valid":
            return self.__GetValidData()
        if self.type == "test":
            return self.__GetTestData()


    def __GetTrainData(self):

        pathList = []

        root = os.path.join(self.rootPath, 'train')
        fileNames = os.listdir(root)
        fileNames.sort()
        for name in fileNames:
            path = os.path.join(root, name)
            pathList.append(path)

        return pathList

    def __GetValidRealData(self):

        pathList = []

        root = os.path.join(self.rootPath, 'valid_real')
        fileNames = os.listdir(root)
        fileNames.sort()
        for name in fileNames:
            path = os.path.join(root, name)
            pathList.append(path)

        return pathList

    def __GetTestRealData(self):

        root = os.path.join(self.rootPath, 'test_real')
        self.inputPath = os.path.join(root, 'input')
        self.gtPath = os.path.join(root, 'gt')
        pathList = []

        spikeNames = os.listdir(self.inputPath)
        spikeNames.sort()
        imgNames = os.listdir(self.gtPath)
        imgNames.sort()

        for sna, ina in zip(spikeNames, imgNames):
            spikePath = os.path.join(self.inputPath, sna)
            imgPath = os.path.join(self.gtPath, ina)
            pathList.append((spikePath, imgPath))

        return pathList

    def __GetValidData(self):

        pathList = []

        root = os.path.join(self.rootPath, 'valid')
        fileNames = os.listdir(root)
        fileNames.sort()
        for name in fileNames:
            path = os.path.join(root, name)
            pathList.append(path)

        return pathList

    def __GetTestData(self):

        pathList = []

        root = os.path.join(self.rootPath, 'test')
        fileNames = os.listdir(root)
        fileNames.sort()
        for name in fileNames:
            path = os.path.join(root, name)
            pathList.append(path)

        return pathList