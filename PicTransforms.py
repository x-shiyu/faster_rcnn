# -*- coding: utf-8 -*-
import torchvision
import cv2.cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt

transforms = torchvision.transforms
F = torch.nn.functional
nn = torch.nn


class GeneralTrans(object):

    def __call__(self, data):
        img, pos = data
        img = self._maxToOneNormal(img)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_normal = transforms.Normalize(mean=mean, std=std)
        return (trans_normal(img), pos)

    def _maxToOneNormal(self, img):
        img = img.type(torch.float)
        max = img.max()
        min = img.min()
        return (img - min) / (max - min)


class PaddingResize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        img, pos = data
        pos = np.array(pos)
        width = img.shape[1]
        height = img.shape[0]
        radio_w = width / self.output_size
        radio_h = height / self.output_size
        target_radio = max([radio_w, radio_h])
        target_w = np.ceil(width / target_radio)
        target_h = np.ceil(height / target_radio)
        # 计算目标框的缩放
        pos_x_min = np.ceil(pos[:, 0] / target_radio).astype(np.int)
        pos_y_min = np.ceil(pos[:, 1] / target_radio).astype(np.int)
        pos_x_max = np.ceil(pos[:, 2] / target_radio).astype(np.int)
        pos_y_max = np.ceil(pos[:, 3] / target_radio).astype(np.int)
        img = cv.resize(img, (int(target_w), int(target_h)), interpolation=cv.INTER_LINEAR)
        # 超出部分
        w_over = (self.output_size - img.shape[1]) % 2
        h_over = (self.output_size - img.shape[0]) % 2
        padding_w = (self.output_size - img.shape[1]) // 2
        padding_h = (self.output_size - img.shape[0]) // 2

        padding_all = [0, 0, 0, 0]
        pos_target = [pos_y_min, pos_x_min, pos_y_max, pos_x_max]

        # 判断高和宽哪个边小于输出大小就补全0
        if target_w < self.output_size:
            padding_all[0] = int(padding_w)
            padding_all[1] = int(padding_w + w_over)
            pos_target[0] = (pos_target[0] + int(padding_h)).astype(np.int)
            pos_target[2] = (pos_target[2] + int(padding_h)).astype(np.int)

        if target_h < self.output_size:
            padding_all[2] = int(padding_h)
            padding_all[3] = int(padding_h + h_over)
            pos_target[1] = (pos_target[1] + int(padding_w)).astype(np.int)
            pos_target[3] = (pos_target[3] + int(padding_w)).astype(np.int)

        padding_trans = nn.ZeroPad2d(tuple(padding_all))
        img = img.astype('int16')
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).contiguous()
        img = padding_trans(img)
        return img, np.array(pos_target).T

# trans = transforms.Compose([
#     PaddingResize(448),
#     GeneralTrans()
# ])
# pos = [625, 380, 724, 477]
# img = cv.imread('./images/balloon/train/6810773040_3d81036d05_k.jpg')
# img, res_pos = trans((img, pos))
# img = img.permute(1, 2, 0).contiguous().numpy()
# img = cv.rectangle(img, (res_pos[0], res_pos[1]), (res_pos[2], res_pos[3]), (0, 0, 0), 3)
# print(res_pos)
# plt.imshow(img)
# plt.show()
