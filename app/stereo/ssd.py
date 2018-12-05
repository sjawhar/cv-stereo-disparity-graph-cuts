import numpy as np
from .utils import to_gray, convolve

def disparity(image_left, image_right, kernel=7, search_depth=30):
    gray_left = to_gray(image_left)
    gray_right = to_gray(image_right)
    kernel = np.ones((kernel, kernel), dtype=np.float32)

    min_ssd = np.full(gray_left.shape, float('inf'), dtype=np.float32)
    labels = np.zeros(gray_left.shape, dtype=np.int)
    for offset in range(search_depth):
        shifted = gray_right if offset == 0 else gray_right[:, :-offset]
        raw_ssd = np.square(gray_left[:, offset:] - shifted)
        ssd = convolve(raw_ssd, kernel)
        label_min = ssd < min_ssd[:, offset:]
        min_ssd[:, offset:][label_min] = ssd[label_min]
        labels[:, offset:][label_min] = offset

    return labels
