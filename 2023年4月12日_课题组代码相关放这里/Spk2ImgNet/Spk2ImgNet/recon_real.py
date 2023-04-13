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

if __name__ == "__main__":

    dataName = "reds"
    spikeRadius = 6
    frameRadius = 2
    frameStride = 7
    useFLow = False
    batchSize = 8

    reuse = True
    reuseNum = 'best'  #number or 'best'
    checkPath = "CheckPoints/%s/Training_Epoch_%s.pth" %('UseFlow' if useFLow else 'NoFlow',reuseNum)
    # checkPath = os.path.join(rootPath, checkPath)

    dataPath = "/home/storage1/Dataset/SpikeImageData/RealData/train-350kmh.dat"
    frameLen = 2 * spikeRadius + 2 * frameRadius * frameStride + 1
    halfLen = frameLen // 2
    spikes = load_spike_raw(dataPath)
    totalLen = spikes.shape[0]
    center = 1000#totalLen // 2

    metrics = Metrics()
    model = Spk2Img(spikeRadius, frameRadius, frameStride).cuda()

    if reuse:
        _, modelDict, _ = LoadModel(checkPath, model)

    model.eval()
    with torch.no_grad():
        num = 0
        pres = []
        gts = []
        spikes = spikes[center - halfLen: center + halfLen + 1]
        spikes = np.pad(spikes, ((0, 0), (3, 3), (0, 0)), mode='constant')
        spikes = spikes.astype(float) * 2 - 1
        spikes = torch.FloatTensor(spikes)
        D, H, W = spikes.size()
        spikes = spikes.view(1, 1, D, H, W)
        spikes = spikes.cuda()
        _, preImg = model(spikes)

        preImg = preImg.clamp(min=-1., max=1.)
        preImg = preImg.detach().cpu().numpy()

        preImg = (preImg + 1.) / 2. * 255.
        preImg = np.clip(preImg, 0., 255.)
        preImg = preImg.astype(np.uint8)
        preImg = preImg[:, 3:-3][0]
        img = Image.fromarray(preImg)
        img.save('/home/yangchen/Desktop/real_eval/1000.jpg')

        niqe = metrics.Cal_Single_NIQE(preImg)
        print('NIQE: %s' % (niqe))