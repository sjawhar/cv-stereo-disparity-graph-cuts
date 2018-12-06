import numpy as np
import re
import sys
import cv2

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.

Taken from https://gist.github.com/chpatrick/8935738
'''
def load_pfm(filename, resize=1):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  with open(filename, 'rb') as file:
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
      color = True
    elif header == 'Pf':
      color = False
    else:
      raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s*(\d+)\s*$', file.readline().decode('utf-8'))
    if not dim_match:
      raise Exception('Malformed PFM header.')
    width, height = map(int, dim_match.groups())

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
      endian = '<'
      scale = -scale
    else:
      endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')

  shape = (height, width, 3) if color else (height, width)
  data = np.reshape(data, shape)[::-1]

  if resize != 1:
    data = cv2.resize(data, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA).astype(float) * resize

  is_inf = data == np.inf
  data[is_inf] = -1
  return data

