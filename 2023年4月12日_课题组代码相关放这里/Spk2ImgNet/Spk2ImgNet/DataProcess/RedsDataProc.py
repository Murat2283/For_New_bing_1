import os
import random
import shutil
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
from PIL import Image
import cv2
from DataProcess.GeneSpike import Vi2Sp, Img2Sp_new, DumpSpike
from LoadSpike import LoadSpike
import copy
import torch.multiprocessing as mp
# from ImgsInterPolation import Interpolate
# import tqdm


def BreadthWalk(path):
    root = path
    dirSeq = [""]
    while len(dirSeq) != 0:
        midFix = dirSeq[0]
        dirList = []
        fileList = []
        eleList = os.listdir(os.path.join(root,midFix))
        eleList.sort()
        for ele in eleList:
            path = os.path.join(root, midFix, ele)
            if os.path.isdir(path):
                dirList.append(ele)
                dirSeq.append(os.path.join(midFix, ele))
            else:
                fileList.append(ele)
        dirSeq.pop(0)
        yield midFix, dirList, fileList

def DeepWalk(path):
    root = path
    eleList = os.listdir(root)
    fileList = []
    dirList = []
    for ele in eleList:
        path = os.path.join(root, ele)
        if not os.path.isdir(path):
            fileList.append(ele)
        else:
            dirList.append(ele)
    yield root, dirList, fileList
    for dir in dirList:
        path = os.path.join(root, dir)
        yield from DeepWalk(path)

class RedsDataProc():

    def __init__(self):
        pass

    def Sample(self, srcPath, tarPath, stride=11, total=5):
        if os.path.exists(tarPath):
            shutil.rmtree(tarPath)
        os.mkdir(tarPath)

        for midFix, dirs, files in BreadthWalk(srcPath):
            if len(files) != 0:
                files.sort()
                length = len(files)
                indexList = list(range(length - stride + 1))
                for i in range(1,total+1):

                    newDirPath = tarPath
                    splitedMidFix = midFix.split('/')
                    for preFix in splitedMidFix[0:-1]:
                        newDirPath = os.path.join(newDirPath, preFix)
                        if not os.path.exists(newDirPath):
                            os.mkdir(newDirPath)
                    newDir = splitedMidFix[-1] + "_part_%s" % (i)
                    print(newDir)
                    newDirPath = os.path.join(newDirPath, newDir)
                    if not os.path.exists(newDirPath):
                        os.mkdir(newDirPath)

                    random.shuffle(indexList)
                    start = indexList[0]
                    indexList.pop(0)
                    childFiles = files[start : start + stride]
                    for file in childFiles:
                        filePath = os.path.join(srcPath, midFix, file)
                        im = Image.open(filePath)
                        newFilePath = os.path.join(newDirPath, file)
                        im.save(newFilePath)



    # def Interpolation(self, srcPath, tarPath):
    #     if os.path.exists(tarPath):
    #         shutil.rmtree(tarPath)
    #     os.mkdir(tarPath)
    #
    #     for midFix, dirs, files in BreadthWalk(srcPath):
    #         if len(dirs) != 0:
    #             for dir in dirs:
    #                 inputPath = os.path.join(srcPath, midFix, dir)
    #                 outputPath = os.path.join(tarPath,midFix)
    #                 # if outputPath != tarPath:
    #                 #     if os.path.exists(outputPath):
    #                 #         shutil.rmtree(outputPath)
    #                 #     os.mkdir(outputPath)
    #                 if not os.path.exists(outputPath):
    #                     os.mkdir(outputPath)
    #                 outputPath = os.path.join(outputPath, dir)
    #                 # if os.path.exists(outputPath):
    #                 #     shutil.rmtree(outputPath)
    #                 os.mkdir(outputPath)
    #                 print('Images in %s dir are interpolating!' %(dir))
    #                 Interpolate(inputPath, outputPath, batch_size=9, sf=400)


    def GenVideo(self, srcPath, tarPath):

        # if not os.path.exists(tarPath):
        #     os.mkdir(tarPath)
        # videoPath = os.path.join(tarPath, 'videos')
        videoPath = tarPath
        if os.path.exists(videoPath):
            shutil.rmtree(videoPath)
        os.mkdir(videoPath)

        for midFix, dirs, files in BreadthWalk(srcPath):
            if len(files) > 0:
                videoName = midFix.split('/')[-1] + ".avi"
                outPath = os.path.join(videoPath, videoName)

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(outPath, fourcc, 15.0, (400, 250))
                for file in files:
                    imgPath = os.path.join(srcPath, midFix, file)
                    im = Image.open(imgPath).convert('L')
                    im = np.array(im)
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                    out.write(im)
                out.release()


    def GenSpikeFromImgs(self, imgsPath, spikePath, height, width):

        # viPath = os.path.join(videoPath, 'videos')
        # spPath = os.path.join(spikePath, 'spikes')
        imgsPath = imgsPath
        spPath = spikePath

        if os.path.exists(spPath):
            shutil.rmtree(spPath)
        os.mkdir(spPath)

        for midFix, dirs, files in BreadthWalk(imgsPath):
            if len(files) > 0:
                print('Number %s is processing!' % (midFix))
                spikeName = midFix
                outPath = os.path.join(spPath, spikeName)
                # outPath = os.path.join(spPath, spikeName + '-{:0>4d}')
                imgList = []
                files.sort(key=lambda x: int(x.split('.')[0]))
                for file in files:
                    imgPath = os.path.join(imgsPath, midFix, file)
                    im = Image.open(imgPath).convert('L')
                    im = np.array(im)
                    imgList.append(im)
                Img2Sp_new(imgList, (width, height), outPath)

    def SpikeSample(self, srcPath, tarPath, sampleNum=4, radius=16):

        if os.path.exists(tarPath):
            shutil.rmtree(tarPath)
        os.mkdir(tarPath)

        index = [num for num in range(40, 400, 40)]
        sampleIndex = []
        for num in index:
            if num >= 2 * radius and num < 400 - 2 * radius:
                sampleIndex.append(num)
        files = os.listdir(srcPath)
        for file in files:
            spikeName = file.split('.')[0]
            spPath = os.path.join(srcPath, file)
            spFrames, gtFrames = LoadSpike(spPath)
            currentIndex = copy.deepcopy(sampleIndex)
            for i in range(sampleNum):
                random.shuffle(currentIndex)

                center = currentIndex[0]
                samSpFrames = spFrames[center - 2 * radius: center + 2 * radius + 1]
                samGtFrames = gtFrames[center - 2 * radius: center + 2 * radius + 1]
                savePath = os.path.join(tarPath, spikeName + '_id_%s' % (i))
                DumpSpike(savePath, samSpFrames, samGtFrames)

                currentIndex.pop(0)




if __name__ == "__main__":

    process = RedsDataProc()
    width, height = 256, 256
    radius = 16

    # print('Image Samle!')
    # srcPath = "/home/storage1/Dataset/REDS/train"
    # tarPath = "/home/storage1/Dataset/REDS/Sampled"
    # process.Sample(srcPath, tarPath)

    # print('Generate spike from images!')
    # srcPath = "/home/storage2/Dataset/REDS/Crop_256x256"
    # tarPath = "/home/storage2/Dataset/REDS/Spike_256x256"
    # process.GenSpikeFromImgs(srcPath, tarPath, height, width)


    print('Sample Spikes from Generated Spikes!')
    srcPath = "/home/storage2/Dataset/REDS/Spike_256x256"
    tarPath = "/home/storage2/Dataset/REDS/Spike_Sample_256x256"
    process.SpikeSample(srcPath, tarPath, sampleNum = 4, radius = radius)


