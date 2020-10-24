import numpy as np
import math


class ROIAlign():
    def __init__(self, out_size):
        self.out_size = out_size

    def _getAnchorLayer(k_0, w, h, min_size):
        return str(int(np.ceil(k_0 + math.log2(math.sqrt(w * h) / min_size))))

    def __call__(self, feture_maps, proposals):
        # layer_num = self._getAnchorLayer(0, )
        pass