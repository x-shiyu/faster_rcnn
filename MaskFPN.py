import torch.nn as nn

F = nn.functional


class MaskFPN(nn.Module):
    def __init__(self):
        super(MaskFPN, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.down_channel = nn.ModuleList([
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        ])
        self.size_upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3_list = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        # x为resnet输出的4层特征图，然后进过这里进行融合
        last_layer_pool = self.maxpool(x['3'])
        fpn_out = []
        layers = [last_layer_pool]
        for key, value in x.items():
            layers.insert(1, value)
        for idx, action in enumerate(self.down_channel):
            # 先进行1 * 1卷积降维
            res = action(layers[idx])
            # 再进行上采样融合
            concat_res = self.size_upsample(res) + layers[idx + 1]
            # 最后进行3 * 3卷积消除混叠效应
            fpn_out.append(self.conv3_list[idx](concat_res))
        return fpn_out
