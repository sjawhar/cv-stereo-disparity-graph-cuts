import cv2
import numpy as np
from scipy import ndimage

def to_gray(image):
    if len(image.shape) == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

def convolve(image, kernel):
    return ndimage.filters.convolve(image, kernel, mode='constant', cval=0)
