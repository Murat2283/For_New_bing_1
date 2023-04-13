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

def SaveModel(epoch, model, optimizer, saveRoot, best=False):
    saveDict = {
        'pre_epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    }
    fileName = "Training_Epoch_%s.pth" %(str(1) if not best else 'best')
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

    dataName = "reds"  # "reds"
    spikeRadius = 16#6
    frameRadius = 2#2 
    frameStride = 8#7  
    batchSize = 64  # 4#2
    epoch = 200
    start_epoch = 0
    lr = 2e-4#2e-3
    saveRoot = "CheckPoints"
    decay_recon = 0.
    decay_flow = 0.
    perIter = 20

    reuse = True
    reuseNum = 'best'  # number or 'best'
    checkPath = "CheckPoints/Training_Epoch_%s.pth" % (reuseNum)
    # checkPath = os.path.join(rootPath, checkPath)


    validContainer = dl.DataContainer(dataName=dataName, dataType='test', spikeRadius=spikeRadius,
                                      frameRadius=frameRadius, frameStride=frameStride, batchSize=4)
    validData = validContainer.GetLoader()

    metrics = Metrics()
    loss = Loss()
    model = Spk2Img(spikeRadius, frameRadius, frameStride).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=False)

    # lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=0, min_lr=1e-10, verbose=True)
    if reuse:
        preEpoch, modelDict, optDict = LoadModel(checkPath, model, optimizer)
        # model.load_state_dict(modelDict)
        # optimizer.load_state_dict(optDict)
        start_epoch = preEpoch + 1
        for para in optimizer.param_groups:
            para['lr'] = lr

    model.eval()

    with torch.no_grad():
        num = 0
        pres = []
        gts = []
        for i, (spikes, gtImgs, _) in enumerate(validData):
            B, D, H, W = spikes.size()
            spikes = spikes.view(B, 1, D, H, W)
            spikes = spikes.cuda()
            _, D_frame, _, _ = gtImgs.size()
            gtImgs = gtImgs.cuda()
            _, preImg= model(spikes)
            center = D_frame // 2
            gtImg = gtImgs[:, center]

            preImg = preImg.clamp(min=-1., max=1.)
            preImg = preImg.detach().cpu().numpy()
            gtImg = gtImg.clamp(min=-1., max=1.)
            gtImg = gtImg.detach().cpu().numpy()

            preImg = (preImg + 1.) / 2. * 255.
            preImg = preImg.astype(np.uint8)
            # preImg = preImg[:, 3:-3]

            gtImg = (gtImg + 1.) / 2. * 255.
            gtImg = gtImg.astype(np.uint8)
            # gtImg = gtImg[:, 3:-3]

            pres.append(preImg)
            gts.append(gtImg)
        pres = np.concatenate(pres, axis=0)
        gts = np.concatenate(gts, axis=0)

        psnr = metrics.Cal_PSNR(pres, gts)
        ssim = metrics.Cal_SSIM(pres, gts)

        print('*********************************************************')
        # print('PSNR: %s, SSIM: %s, Best_PSNR: %s, Best_SSIM: %s'
        #       % (psnr, ssim))
        print('PSNR: %s, SSIM: %s'
              % (psnr, ssim))