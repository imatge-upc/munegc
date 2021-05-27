# --*-- coding:utf-8 --*--
import math
import cv2
import os
import numpy as np

from utils.rgbd_util import *
from utils.getCameraParam import *

'''
C: Camera matrix
D: Depth image, the unit of each element in it is "meter"
RD: Raw depth image, the unit of each element in it is "meter"
'''


def getHHA(C, D, RD):
    missingMask = (RD == 0)
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C)

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1, np.maximum(-1, np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180

    pc[:, :, 2] = np.maximum(pc[:, :, 2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:, :, 2] = 31000/pc[:, :, 2]
    I[:, :, 1] = h
    I[:, :, 0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I > 255] = 255
    HHA = I.astype(np.uint8)
    return HHA
