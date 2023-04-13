import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch import optim
import numpy as np
from DataProcess import DataLoader as dl
from Model.Loss import Loss
from Model.Spk2Img import Spk2Img
from PIL import Image
from Metrics.Metrics import Metrics
import cv2
import shutil

def SaveModel(epoch, model, optimizer, saveRoot, best=False):
    saveDict = {
        'pre_epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    }
    fileName = "Training_Epoch_%s.pth" %(epoch if not best else 'best')
    savePath = os.path.join(saveRoot, fileName)
    torch.save(saveDict, savePath)

def LoadModel(checkPath, model, optimizer=None):
    stateDict = torch.load(checkPath)
    pre_epoch = stateDict['pre_epoch']
    model.load_state_dict(stateDict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(stateDict['optimizer_state_dict'])

    return pre_epoch, stateDict['model_state_dict'], stateDict['optimizer_state_dict']

def load_spike_raw(path: str, width=400, height=250) -> np.ndarray:
    '''
    Load bit-compact raw spike data into an ndarray of shape
        (`frame number`, `height`, `width`).
    '''
    with open(path, 'rb') as f:
        fbytes = f.read()
    fnum = (len(fbytes) * 8) // (width * height)  # number of frames
    frames = np.frombuffer(fbytes, dtype=np.uint8)
    frames = np.array([frames & (1 << i) for i in range(8)])
    frames = frames.astype(np.bool).astype(np.uint8)
    frames = frames.transpose(1, 0).reshape(fnum, height, width)
    frames = np.flip(frames, 1)
    return frames

def GenerateVideo(outputPath,frameList, height=250, width=400):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputPath, fourcc, 15.0, (width, height))
    frameLen = len(frameList)
    for i in range(0,frameLen):
        frame = frameList[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()


if __name__ == "__main__":

    restore_root = "/home/yangchen/Desktop/restored_img/spk"
    if not os.path.exists(restore_root):
        os.mkdir(restore_root)
    # spikeRadius = 16  # 6
    # frameRadius = 2  # 2
    # frameStride = 8  # 7
    spikeRadius = 6
    frameRadius = 2
    frameStride = 7
    block_len = 2 *spikeRadius + 2 * frameRadius *frameStride +1

    useFLow = False
    batchSize = 4

    reuse = True

    reuseNum = 'best'  #number or 'best'i
    checkPath = "CheckPoints/Training_Epoch_%s_41.pth" %(reuseNum)
    # checkPath = os.path.join(rootPath, checkPath)
    # train-350kmh.dat #ballon.dat viaduct-bridge.dat forest.da car-100kmh.dat
    # dataPath = "/home/storage1/Dataset/SpikeImageData/RealData/car-100kmh.dat"
    # train-350kmh.dat ballon.dat viaduct-bridge.dat forest.dat car-100kmh.dat rotation1.dat rotation2.dat railway.dat
    nameList = ['train-350kmh', 'ballon', 'viaduct-bridge', 'forest', 'car-100kmh', 'rotation1',
                'rotation2', 'railway']
    s = 32
    frameLen = 2 * spikeRadius + 2 * frameRadius * frameStride + 1
    halfLen = frameLen // 2
    for name in nameList:
        dataPath = "/home/storage1/Dataset/SpikeImageData/RealData/" + name + ".dat"
        spikes = load_spike_raw(dataPath)

        # dataPath = "/home/storage1/Dataset/REDS/Spike_256x256_v12/234.npz"
        # frameLen = 2 * spikeRadius + 2 * frameRadius * frameStride + 1
        # halfLen = frameLen // 2
        # from DataProcess.LoadSpike import LoadSpike
        # spikes,_ = LoadSpike(dataPath)

        totalLen = spikes.shape[0]
        center = totalLen // 2
        # metrics = Metrics()
        model = Spk2Img(spikeRadius, frameRadius, frameStride).cuda()

        if reuse:
            _, modelDict, _ = LoadModel(checkPath, model)

        model.eval()
        with torch.no_grad():
            num = 0
            pres = []
            batchFlag = 1
            inputs = np.zeros((batchSize, block_len, 256, 400))  # 65
            for i in range(32, totalLen - 32, s):
                batchFlag = 1
                spike = spikes[i - halfLen: i + halfLen + 1]
                # print(spike.shape)
                spike = np.pad(spike, ((0, 0), (3, 3), (0, 0)), mode='constant')
                spike = spike.astype(float) * 2 - 1
                # spike = spike.astype(float)
                # print(num, spike.shape)
                inputs[num % batchSize] = spike

                num += 1

                if num % batchSize == 0:
                    # inputs = np.array(inputs)
                    inputs = torch.FloatTensor(inputs)

                    B, D, H, W = inputs.size()
                    inputs = inputs.view(B, 1, D, H, W)
                    inputs = inputs.cuda()

                    _, preImg = model(inputs)

                    preImg = preImg.clamp(min=-1., max=1.)
                    preImg = preImg.detach().cpu().numpy()
                    preImg = (preImg + 1.) / 2. * 255.
                    preImg = np.clip(preImg, 0., 255.)
                    preImg = preImg.astype(np.uint8)
                    preImg = preImg[:, 3:-3]

                    inputs = np.zeros((batchSize, block_len, 256, 400))  # 65

                    pres.append(preImg)
                    batchFlag = 0

            if batchFlag == 1:
                # inputs = np.array(inputs)
                imgNum = num % batchSize
                inputs = inputs[0:imgNum]
                inputs = torch.FloatTensor(inputs)

                B, D, H, W = inputs.size()
                inputs = inputs.view(B, 1, D, H, W)
                inputs = inputs.cuda()

                _, preImg = model(inputs)

                preImg = preImg.clamp(min=-1., max=1.)
                preImg = preImg.detach().cpu().numpy()
                preImg = (preImg + 1.) / 2. * 255.
                preImg = np.clip(preImg, 0., 255.)
                preImg = preImg.astype(np.uint8)
                preImg = preImg[:, 3:-3]

                inputs = np.zeros((batchSize, block_len, 256, 400))  # 65

                pres.append(preImg)

            preImgs = np.concatenate(pres, axis=0)

            dir = restore_root + "/spk_" + name + "_str_" + str(s) + "/"
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
            from PIL import Image

            count = 0
            for img in preImgs:
                count += 1
                img = Image.fromarray(img)
                img.save(dir + '%s.jpg' % (count))

            # GenerateVideo('/home/yangchen/Desktop/real_eval_spk2img.avi', preImgs, 250, 400)

            # num = 0
