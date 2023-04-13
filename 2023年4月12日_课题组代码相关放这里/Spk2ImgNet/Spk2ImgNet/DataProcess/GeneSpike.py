import numpy as np
import cv2

def DumpSpike(path: str, spSeq: np.ndarray, gt: np.ndarray):
    '''
    Store a spike sequence with it's tag to `.npz` file.
    '''
    length = spSeq.shape[0]
    spSeq = spSeq.astype(np.bool)
    spSeq = np.array([spSeq[i] << (i & 7) for i in range(length)])
    spSeq = np.array([np.sum(spSeq[i: min(i+8, length)], axis=0)
                    for i in range(0, length, 8)]).astype(np.uint8)
    np.savez(path, spSeq=spSeq, gt=gt, length=np.array(length))

def Vi2Sp(path: str, size: tuple, radius: int, save_path: str):

    window = 2 * 2 * radius

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('width : {}\nheight: {}\nfps   : {}\nframes: {}'
          .format(width, height, fps, frame_num))

    cratio = min(width/size[0], height/size[1])
    cleft = round((width - size[0]*cratio) / 2)
    cright = width - cleft
    ctop = round((height - size[1]*cratio) / 2)
    cbottom = height - ctop

    adder = np.random.random((size[1], size[0])) * 255
    win_num = 0
    win_size = 0
    xframes = []  # (window, height, width)
    yframes = []   # (2*radius+1, height, width)
    counter = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[ctop:cbottom, cleft:cright, :]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

        adder += frame
        saturated = adder >= 255  # saturated is a bool matrix
        xframes.append(saturated)
        adder -= saturated * 255

        if win_size >= radius and win_size <= 3 * radius:  # choose the middle frame as y
            yframes.append(frame)
        win_size += 1
        if win_size == window+1:  # end this window
            xframes = np.array(xframes).astype(np.bool)
            yframes = np.array(yframes).astype(np.uint8)

            DumpSpike(save_path.format(counter), xframes, yframes)
            counter += 1
            xframes = []
            yframes = []
            win_size = 0
            win_num += 1
            print('\rtransforming: {}/{}'.format(win_num, frame_num // window),
                  end='', flush=True)

    # print()
    cap.release()
    cv2.destroyAllWindows()

def Img2Sp(imgList: list, size: tuple, radius: int, save_path: str):

    window = 2 * 2 * radius

    width = size[0]
    height = size[1]
    frame_num = len(imgList)

    adder = np.random.random((height, width)) * 255
    win_num = 0
    win_size = 0
    xframes = []  # (window, height, width)
    yframes = []   # (2*radius+1, height, width)
    counter = 0
    for frame in imgList:

        adder += frame
        saturated = adder >= 255  # saturated is a bool matrix
        xframes.append(saturated)
        adder -= saturated * 255

        if win_size >= radius and win_size <= 3 * radius:  # choose the middle frame as y
            yframes.append(frame)
        win_size += 1
        if win_size == window+1:  # end this window
            xframes = np.array(xframes).astype(np.bool)
            yframes = np.array(yframes).astype(np.uint8)

            DumpSpike(save_path.format(counter), xframes, yframes)
            counter += 1
            xframes = []
            yframes = []
            win_size = 0
            win_num += 1
            print('\rtransforming: {}/{}'.format(win_num, frame_num // window),
                  end='', flush=True)


def Img2Sp_new(imgList: list, size: tuple, save_path: str):

    width = size[0]
    height = size[1]

    adder = np.random.random((height, width)) * 255
    xframes = []  # (window, height, width)
    yframes = []   # (2*radius+1, height, width)
    counter = 0
    for index, frame in enumerate(imgList):
        counter += 1
        if index % 10 == 0:
            yframes.append(frame)
        adder += (frame * 0.06)
        if counter == 10:
            saturated = adder >= 250  # saturated is a bool matrix
            mask = 1 - saturated.astype(np.float)
            xframes.append(saturated)
            adder -= saturated * 255
            counter = 0
            # adder *= mask
    xframes = np.array(xframes).astype(np.bool)
    yframes = np.array(yframes).astype(np.uint8)
    DumpSpike(save_path, xframes, yframes)

# def Img2Sp_new(imgList: list, size: tuple, save_path: str):
#
#
#     width = size[0]
#     height = size[1]
#
#
#     adder = np.random.random((height, width)) * 255
#
#     xframes = []  # (window, height, width)
#     yframes = []   # (2*radius+1, height, width)
#     counter = 0
#     for frame in imgList:
#         if counter % 10 == 0:
#             yframes.append(frame)
#         counter += 1
#         adder += (frame * 0.06)
#         if counter % 10 == 0:
#             saturated = adder >= 250  # saturated is a bool matrix
#             mask = 1 - saturated.astype(np.float)
#             xframes.append(saturated)
#             adder -= saturated * 255
#             # adder *= mask
#
#     xframes = np.array(xframes).astype(np.bool)
#     yframes = np.array(yframes).astype(np.uint8)
#     DumpSpike(save_path, xframes, yframes)