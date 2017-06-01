import numpy as np
import tensorflow as tf

RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_BGR = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)

def np_rgb2yuv(im):
    im = im[...,::-1]
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET

def np_yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_BGR, ([2], [1]))[...,::-1]#BGR->RGB
