import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from torch import nn
from torch.nn.modules.utils import _pair

# print(_pair(3))

box = torch.tensor([[0.5, 1.4, 4.3, 3.6]], dtype=torch.float)
# print(box.shape)
#
# image = torch.rand(1, 1, 6, 6)
# print(image.numpy())
#
# pooled_regions = torchvision.ops.roi_align(image, [box], output_size=(2, 2))
# # check the size
# print(pooled_regions.shape)
# print(pooled_regions)

cor = [0.5, 1.4, 4.3, 3.6]
w4 = (cor[3] - cor[1]) / 8
h4 = (cor[2] - cor[0]) / 8
x1 = cor[1] + w4
y1 = cor[0] + h4
x2 = cor[1] + 3 * w4
y2 = cor[0] + h4
x3 = cor[1] + w4
y3 = cor[0] + 3 * h4
x4 = cor[1] + 3 * w4
y4 = cor[0] + 3 * h4

imgInfo = [[0.32060778, 0.11620915, 0.71926695, 0.36150396, 0.6166717, 0.45884472],
           [0.4763655, 0.43691587, 0.11007494, 0.94817156, 0.7158479, 0.33027387],
           [0.03077334, 0.57008356, 0.31272697, 0.30899173, 0.2531662, 0.8905018],
           [0.9859025, 0.33705324, 0.1618337, 0.8115549, 0.48484266, 0.31700397],
           [0.24059433, 0.50169885, 0.6884048, 0.9396151, 0.10167116, 0.9164184],
           [0.9951332, 0.08902311, 0.3455944, 0.58422947, 0.3528126, 0.34536755]]


def binaryInsert(x, y, imgInfo):
    x2 = int(np.ceil(x))
    x1 = int(np.floor(x))
    y2 = int(np.ceil(y))
    y1 = int(np.floor(y))
    return imgInfo[x1][y1] * (x2 - x) * (y2 - y) + \
           imgInfo[x2][y1] * (x - x1) * (y2 - y) + \
           imgInfo[x1][y2] * (x2 - x) * (y - y1) + \
           imgInfo[x2][y2] * (x - x1) * (y - y1)


'''
[[0.3984, 0.4091],
 [0.3365, 0.5874]]
'''

ax1 = binaryInsert(x1, y1, imgInfo)
ax2 = binaryInsert(x2, y2, imgInfo)
ax3 = binaryInsert(x3, y3, imgInfo)
ax4 = binaryInsert(x4, y4, imgInfo)
print(np.average([ax1, ax2, ax3, ax4]))
