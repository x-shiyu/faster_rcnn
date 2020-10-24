# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils import generate_anchor_base,ProposalCreator,normal_init

nn = torch.nn
F = nn.functional

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 生成网格点，其中网格点的间距为feat_stride例如：
    '''
        a=[1,2,3,4]
        b=[1,2,3,4]
        shift_a = [[1,2,3,4]
                  [1,2,3,4],
                  [1,2,3,4],
                  [1,2,3,4]]
        shift_b = [[1,1,1,1],
                   [2,2,2,2],
                   [3,3,3,3],
                   [4,4,4,4]]
        这样就能组合出所有点的坐标了
    '''
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # shift为 (w*h)*4维度
    # 生成ymin xmin ymax xmax的点例如:
    '''
        ravel将数据打平成下面的格式:
        [[1234123412341234],
        [1111222233334444],
        [1234123412341234],
        [1111222233334444]].T
        
    '''
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0] #9
    K = shift.shape[0] #w*h
    # 1*9*4 + K*1*4 => (K*9)*4
    # 每个像素点给 9个anchor，加起来就是(w*h)*9个

    #anchor_base:9*4
    # shift :(w*h)*4
    # 对生成的基础anchor加行偏移值
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
   
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor



class RegionProposalNetwork(nn.Module):
   
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # 生成anchor表(9,4) 9行表示9个框，每列为4个基准坐标(y_min,x_min,y_max,x_max)
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        # 创建建议框
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0] #anchor个数：现在是9
        # 大小不变的3*3卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 1*1卷积的到9个框的得分，分为正负
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 1*1卷积的到9个框的坐标
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # 把参数变为正太分布（均值0，标准差为0.01）
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
       
        # batch,channel,height,width
        n, _, hh, ww = x.shape
        # anchor (w*h*9)*4 
        #anchor_base是在0，0的基础上做的，如果最后一层的图像大小是输入图像的1/n，那么需要加上这个缩放
        # _enumerate_shifted_anchor返回的是在基础anchor上进行偏移后的anchor
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        # anchor数量，这里是9，这里 anchor.shape[0]是9*(w*h)
        # anchor.size = [9*w*h,4]
        n_anchor = anchor.shape[0] // (hh * ww)
        # 先进行1 * 1卷积
        h = F.leaky_relu(self.conv1(x))

        '''
        这一层的作用的是融合特征图的所有通道数得到其对应的anchor值，例如对于
        输入的featuremap如果是[1024,w,h]，那么对于这一步的操作就是将这1024的channel进行融合
        成anchor*4个channel来表示预测的值，这里的预测值不是ymin,xmin,ymax,xmax，预测的是偏移和缩放值
        对于基础anchor要进行偏移和缩放才能和目标框相匹配，所以，这里预测的就是这个偏移和缩放
        '''
        # (batch,9*4,w,h)
        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
         # (batch,w*h*9,4)这里的4表示的是预测的偏移值，不是ymin,xmin,ymax,xmax
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        #  (batch,9*2,w,h)
        # 通过卷积操作融合特征图来得到特征图上每个点的positive或negative
        rpn_scores = self.score(h)

        #  (batch,w,h,9*2)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
         #  (batch,w,h,9,2) 2表示softmax后的positive和negative
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # (batch,w,h,9,1) 取第二个数为正例（也可以取第一个数字）
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        # (N, H W A, 4)
        for i in range(n):
            # 每一张图的预选框
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            # 这个是来标注roi属于那个batch的，因为经过RPN后每张图生成的roi大小是不一样的，所以无法直接stack只能用一个数组来标注
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


