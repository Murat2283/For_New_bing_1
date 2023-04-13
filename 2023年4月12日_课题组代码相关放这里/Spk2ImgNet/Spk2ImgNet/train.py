import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
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

def eval(model, validData, epoch, optimizer, metrics):

    model.eval()
    print('Eval Epoch: %s' %(epoch))

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
        best_psnr, best_ssim, _ = metrics.GetBestMetrics()

        if psnr >= best_psnr and ssim >= best_ssim:
            metrics.Update(psnr, ssim)
            SaveModel(epoch, model, optimizer, saveRoot, best=True)
            with open('eval_best_log.txt', 'w') as f:
                f.write('epoch: %s; psnr: %s, ssim: %s\n' %(epoch, psnr, ssim))
            B, H, W = pres.shape
            divide_line = np.zeros((H,4)).astype(np.uint8)
            for pre, gt in zip(pres, gts):
                num += 1
                concatImg = np.concatenate([pre, divide_line, gt], axis=1)
                concatImg = Image.fromarray(concatImg)
                concatImg.save('EvalResults/valid_%s.jpg' % (num))

        with open('eval_log.txt', 'a') as f:
            f.write('epoch: %s; psnr: %s, ssim: %s\n' %(epoch, psnr, ssim))
        print('*********************************************************')
        best_psnr, best_ssim, _ = metrics.GetBestMetrics()
        print('Eval Epoch: %s, PSNR: %s, SSIM: %s, Best_PSNR: %s, Best_SSIM: %s'
              %(epoch, psnr, ssim, best_psnr, best_ssim))

    model.train()

def Train(trainData, validData, model, loss, optimizer, epoch, start_epoch, metrics, saveRoot, perIter):
    avg_coarse_loss = 0.
    avg_recon_loss = 0.
    avg_total_loss = 0.
    for i in range(start_epoch, epoch):
        for iter, (spikes, gtImgs, _) in enumerate(trainData):
            # spikes = spikes[:, :, 0:250, 0:250]
            # print(spikes.size())
            B, D, H, W = spikes.size()
            spikes = spikes.view(B, 1, D, H, W)
            spikes = spikes.cuda()
            _, D_frame, _, _ = gtImgs.size()
            gtImgs = gtImgs.cuda()
            coarseImgs, preImg = model(spikes)
            center = D_frame // 2
            gtImg = gtImgs[:,center]

            coarseLoss = loss.CoarseLoss(coarseImgs, gtImgs)
            reconLoss = loss.ReconLoss(preImg, gtImg)
            totalLoss = 1. * coarseLoss + reconLoss
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            avg_coarse_loss += coarseLoss.detach().cpu()
            avg_recon_loss += reconLoss.detach().cpu()
            avg_total_loss += totalLoss.detach().cpu()
            if (iter + 1) % perIter == 0:
                avg_coarse_loss = avg_coarse_loss / perIter
                avg_recon_loss = avg_recon_loss / perIter
                avg_total_loss = avg_total_loss / perIter
                print('=============================================================')
                print('Epoch: %s, Iter: %s' % (i, iter + 1))
                print('CoarseLoss: %s; ReconLoss: %s; TotalLoss: %s' % (
                    avg_coarse_loss.item(), avg_recon_loss.item(), avg_total_loss.item()))
                avg_coarse_loss = 0.
                avg_recon_loss = 0.
                avg_total_loss = 0.

        if (i + 1) % 1 == 0:
            SaveModel(i, model, optimizer, saveRoot)
            eval(model, validData, i, optimizer, metrics)

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

    reuse = False
    reuseNum = -1  # number or 'best'
    checkPath = "CheckPoints/Training_Epoch_%s.pth" % (reuseNum)
    # checkPath = os.path.join(rootPath, checkPath)

    trainContainer = dl.DataContainer(dataName=dataName, dataType='train', spikeRadius=spikeRadius,
                                      frameRadius=frameRadius, frameStride=frameStride, batchSize=batchSize)
    trainData = trainContainer.GetLoader()

    validContainer = dl.DataContainer(dataName=dataName, dataType='valid', spikeRadius=spikeRadius,
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

    model.train()


    Train(trainData, validData, model, loss, optimizer, epoch, start_epoch,
                    metrics, saveRoot, perIter)
