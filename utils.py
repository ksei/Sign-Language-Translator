import os

import cv2


def getGestureNo(cat):
    list = []
    for root, dirs, files in os.walk('./Train Data'):
        for newname in files:
            filename = os.path.join(newname)
            if (filename[0] == cat):
                newstring = filename[1:filename.find('.')]
                list.append(int(newstring))

        if (len(list) == 0):
            return 0
        return max(list)


def calc_HOG (gesture_frame):
    winSize = (224, 256)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 2
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (1, 1)
    hist = hog.compute(gesture_frame, winStride);
    return hist;

