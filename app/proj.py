import cv2
import os
from disparity import disparity_ssd

IMAGES_DIR = 'input-images'

left = cv2.imread(os.path.join(IMAGES_DIR, 'Adirondack-perfect', 'im0.png'))
right = cv2.imread(os.path.join(IMAGES_DIR, 'Adirondack-perfect', 'im1.png'))

cv2.imshow('tmp', disparity_ssd(left, right, max_search=200).astype(np.uint8))
cv2.waitKey(0)
