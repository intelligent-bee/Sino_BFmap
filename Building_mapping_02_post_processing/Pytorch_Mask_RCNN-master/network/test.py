import datetime
import glob
import itertools
import json
import logging
import math
import os
import random
import re
import time
import sys
from collections import OrderedDict

import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torchvision
from torch.autograd import Variable

# from lib.nms_wrapper import nms
# from lib.roi_align.roi_align import CropAndResize, RoIAlign
from roi_align import RoIAlign,CropAndResize

img1 = torch.arange(1, 26).reshape(1, 5, 5)     # values 1~25
img2 = torch.arange(26, 51).reshape(1, 5, 5)    # values 26~50

# 拼接为 [2, 1, 5, 5] 输入图像
images = torch.stack([img1, img2], dim=0).float()

rois = torch.tensor([
    [0, 1.5, 1.5, 4.5, 4.5],  # 从第一张图裁剪中心区域
    [1, 0, 0, 3, 3],  # 从第二张图裁剪左上角区域
], dtype=torch.float)

rois_1_box= torch.tensor([
    [0.3, 0.3, 0.9, 0.9],
    [0, 0, 0.6, 0.6],   # 从第一张图裁剪中心区域
], dtype=torch.float)
rois_i_index= torch.tensor([0, 1], dtype=torch.int)

roi_align1=CropAndResize(3,3)
roi_align2=torchvision.ops.RoIAlign(
            output_size=(3,3), spatial_scale=1.0, sampling_ratio=2)
print(roi_align1(images, rois_1_box, rois_i_index))
print(roi_align2(images, rois))