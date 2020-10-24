# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision
import math
nms = torchvision.ops.nms
F = torch.functional
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loc2bbox(src_bbox, loc):
    """
    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """
    # src_bbox是初始化的anchor，偏移后的anchor
    # loc是经过卷积和激活计算后得出的anchor位置
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)
        
    # 将两个的tensor的type变成一致的
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    # y_min x_min y_max x_max
    src_height = src_bbox[:, 2] - src_bbox[:, 0] #anchor的高
    src_width = src_bbox[:, 3] - src_bbox[:, 1]  #anchor的宽
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height #anchor的中心点的y
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width  #anchor的中心点的x


    # 预测的修正参数（这个卷积出来的值是用来给初始化的anchor做修正的，用这个值来修正初始化的anchor以最大程度满足ground-truth）
    # loc: [9*h*w,4],4表示预测的修正值(dy,dx,dh,dw)
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # 对于anchor的中心点加上（高度*预测值）的偏移
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    # 加上exp是防止dh和dw或出现负值
    # 这里对高宽进行缩放
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    # 预测的y_min,x_min,y_max,x_max
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    '''
        Args:
            src_bbox (array): An image coordinate array whose shape is
                :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
                These coordinates are
                :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            dst_bbox (array): An image coordinate array whose shape is
                :math:`(R, 4)`.
                These coordinates are
                :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.
    
        Returns:
            array:
            Bounding box offsets and scales from :obj:`src_bbox` \
            to :obj:`dst_bbox`. \
            This has shape :math:`(R, 4)`.
            The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    '''

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc



#生成基础anchors按照ratios和scales（后续的图像上的anchors都是在这个生成的anchors基础上进行一些扩大和位移）
'''
ratios表示高宽比
anchor_scales表示缩放比
base_size表示基础大小
'''
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                     anchor_scales=[8, 16, 32]):
    py = base_size / 2.
    px = base_size / 2.
    
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            # 基础大小*缩放比*比率==>基础高度
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            # 基础大小*缩放比*(1/比率)==>基础宽度
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            # 中心点减去一半的高度就是ymin
            anchor_base[index, 0] = py - h / 2.
            # 中心点减去一半的宽度就是xmin
            anchor_base[index, 1] = px - w / 2.
            # 中心点加上一半的高度就是ymax
            anchor_base[index, 2] = py + h / 2.
            # 中心点加上一半的宽度就是xmax
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


class ProposalCreator:
    def __init__(
            self,
            parent_model,
            nms_thresh=0.7,  #nms的阈值
            n_train_pre_nms=12000,  #训练阶段传进来的bbx最大个数
            n_train_post_nms=2000,  #训练阶段nms完的bbx最大个数
            n_test_pre_nms=6000,  #测试阶段传进来的bbx最大个数
            n_test_post_nms=300,  #测试阶段nms完的bbx最大个数
            min_size=16  #bbx最小的一条边大小
    ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    '''
        loc:[w*h*anchor,4]，表示经过卷积操作后的坐标值，其中w*h*anchor行，每一行是4个值(dy,dx,dh,dw)
        score:[w*h*anchor],一维数据，表示每个anchor的得分
        anchor:[w*h*anchor,4]，表示偏移后的anchor值
    '''
    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        # 对偏移后的anchor框进行dy,dx,dw,dh修正得到roi
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        # 将预测的x，y最大值和最小值规范到0到图片的高宽
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0,
                                         img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0,
                                         img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        # 将预测的anchor中高度或者宽度小于设定的阈值的数据删除，同时也要删除其对应的score
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        # 将得分从大到小排序然后截取前n_pre_nms个（只保留n_pre_nms个anchor是为了加速训练），同时也要删除其对应的score
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep = nms(
            torch.from_numpy(roi).to(device),
            torch.from_numpy(score).to(device), self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        # 截断操作，使得所有的参数都小于2，先标准化（化为标准正太也就是0，1），然后对2取余数，再乘上标准差，最后加上平均数
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        # 直接参数化为标准正太
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def getAnchorLayer(k_0,w,h,min_size):
    return str(int(np.ceil(k_0-math.log2(math.sqrt(w*h)/min_size))))



'''
    pos:[ymin,xmin,ymax,xmax]
    Returns:(x_center,y_center,width,height)
'''


def posToPosWH(pos):
    width = int(np.round(pos[3] - pos[1]))
    height = int(np.round(pos[2] - pos[0]))
    x_center = int(np.round(pos[1] + width / 2))
    y_center = int(np.round(pos[0] + height / 2))
    return x_center, y_center, width, height
