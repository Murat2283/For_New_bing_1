import os
import numpy as np


class EventsDataset():

    def __init__(self, type='train'):

        self.type = type
        self.rootPath = "/home/storage1/Dataset/Events/Spikes_for_256x256_v3"

    def GetData(self):

        if self.type == "train":
            return self.__GetTrainData()
        if self.type == "valid":
            return self.__GetValidData()
        if self.type == "test":
            return self.__GetTestData()

    def __GetTrainData(self):

        pathList = []

        root = os.path.join(self.rootPath, 'train')
        fileNames = os.listdir(root)
        for name in fileNames:
            path = os.path.join(root, name)
            pathList.append(path)

        return pathList

    def __GetValidData(self):

        pathList = []

        root = os.path.join(self.rootPath, 'valid')
        fileNames = os.listdir(root)

        for name in fileNames:
            path = os.path.join(root, name)
            pathList.append(path)

        return pathList

    def __GetTestData(self):

        pathList = []

        root = os.path.join(self.rootPath, 'test')
        fileNames = os.listdir(root)

        for name in fileNames:
            path = os.path.join(root, name)
            pathList.append(path)

        return pathList
