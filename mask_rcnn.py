import torch
import torchvision
import cv2.cv2 as cv
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from RPNModel import RegionProposalNetwork
from PicTransforms import PaddingResize, GeneralTrans
from utils import getAnchorLayer, posToPosWH
from MaskFPN import MaskFPN
from ROIAlign import ROIAlign
from torchvision.ops import roi_align

nn = torch.nn
transforms = torchvision.transforms
DataLoader = torch.utils.data.DataLoader
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

modelCom = {}
epoch = 500
INPUT_SIZE = 448
batch_size = 2

for name, module in model.named_children():
    if name != 'roi_heads':
        modelCom[name] = module
    else:
        for c_name, c_module in module.named_children():
            modelCom[c_name] = c_module
for name, module in modelCom['backbone'].named_children():
    modelCom[name] = module


class PicDataset(object):
    def __init__(self, img_infos, transforms=None):
        self.img_infos = img_infos
        self.transforms = transforms
        self.train_data = self.getTrainData(self.img_infos)

    def __getitem__(self, idx):
        img = cv.imread(self.train_data[idx]['filename'])
        pos = self.train_data[idx]['pos']
        if self.transforms:
            img = self.transforms((img, pos))
        return img[0], {"pos": img[1]}

    def __len__(self):
        return len(self.train_data)

    # 解析训练数据
    def getTrainData(self, infos):
        dataArr = []
        imgDirPath = os.path.abspath('./images/balloon/val')
        for idx, item in enumerate(infos.values()):
            posArr = []
            for posItem in item['regions'].values():
                shape_attr = posItem['shape_attributes']
                xmin = np.min(shape_attr['all_points_x'])
                xmax = np.max(shape_attr['all_points_x'])
                ymin = np.min(shape_attr['all_points_y'])
                ymax = np.max(shape_attr['all_points_y'])
                posArr.append([ymin, xmin, ymax, xmax])
            dataArr.append({
                "filename": os.path.join(imgDirPath, item['filename']),
                "pos": posArr
            })
        return dataArr


class FasterRcnn(nn.Module):
    def __init__(self):
        super(FasterRcnn, self).__init__()

        self.backbone = modelCom['body']
        self.rpn = RegionProposalNetwork(256, 256)
        self.fpn = MaskFPN()
        self.roi_heads = ROIAlign(7)
        # self.head_fc = modelCom['box_head']
        # self.cls = nn.linear(1024,2)

    def _getAlignLevel(self, k_0, w, h, input_size):
        return np.floor(k_0 + np.log2(np.sqrt(w * h) / input_size))

    def forward(self, x):
        # 主干网提取特征
        x = self.backbone(x)
        # fpn生成多个尺寸的特征
        fpn_out = self.fpn(x)
        rpn_out = []
        for item in fpn_out:
            # rpn输出：rpn_locs, rpn_scores, rois, roi_indices, anchor
            self.rpn.feat_stride = int(INPUT_SIZE / item.shape[2])
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(item, [INPUT_SIZE, INPUT_SIZE])
            # rpn_index = self._getAlignLevel(0, item.shape[2], item.shape[3], 14)
            # roi_res = roi_align(fpn_out[int(rpn_index)], [torch.from_numpy(rois)], 2)
            rpn_out.append([rois, roi_indices, rpn_scores])
        return rpn_out


data_path = os.path.abspath('./images/balloon/val/via_region_data.json')

with open(data_path) as f:
    dataInfo = json.load(f)

pic_trans = transforms.Compose([
    PaddingResize(INPUT_SIZE),
    GeneralTrans()
])

dataset = PicDataset(img_infos=dataInfo, transforms=pic_trans)

dataLoader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

fast_rcnn = FasterRcnn()

loss_rpn_cls = nn.BCELoss()
loss_rpn_reg = nn.SmoothL1Loss()
# print(modelCom['backbone'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for i in range(epoch):
    for index, item in enumerate(dataLoader):
        imgs, label_pos = item
        label_pos = torch.stack(label_pos).T
        x_center, y_center, width, height = posToPosWH(label_pos[0])
        rect = plt.Rectangle((x_center, y_center), width, height, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        plt.imshow(imgs[0].permute(1, 2, 0))
        plt.show()
        # backbone为res+fpn,输出为(0,1,2,pool)
        # rpn_locs, rpn_scores, rois, roi_indices, anchor
        out = fast_rcnn(imgs)
        break
