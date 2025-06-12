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
# from roi_align import RoIAlign,CropAndResize
from tasks.bbox.generate_anchors import generate_pyramid_anchors
from tasks.merge_task import build_detection_targets

def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable
    
# ROIAlign function
def log2_graph(x):
    """Implementatin of Log2. pytorch doesn't have a native implemenation."""
    return torch.div(torch.log(x), math.log(2.))
    

    
       
def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, :, 0] * height
    center_x += deltas[:, :, 1] * width
    height *= torch.exp(deltas[:, :, 2])
    width *= torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = [y1, x1, y2, x2]
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = window
    y1, x1, y2, x2 = boxes
    # Clip

    y1 = torch.max(torch.min(y1, wy2), wy1)
    x1 = torch.max(torch.min(x1, wx2), wx1)
    y2 = torch.max(torch.min(y2, wy2), wy1)
    x2 = torch.max(torch.min(x2, wx2), wx1)

    clipped = torch.stack([x1, y1, x2, y2], dim=2)
    return clipped

#Backbone of the model
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=True)  # change
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class resnet_graph(nn.Module):
    def __init__(self, block, layers, stage5=False):
        self.inplanes = 64
        super(resnet_graph, self).__init__()
        self.stage5 = stage5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        if self.stage5:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion,  eps=0.001),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # C1 has 64 channels
        C1 = self.maxpool(x)
        # C2 has 64x4 channels
        C2 = self.layer1(C1)
        # C3 has 128x4 channels
        C3 = self.layer2(C2)
        # C4 has 256x4 channels
        C4 = self.layer3(C3)
        # C5 has 512x4 channels
        if self.stage5:
            C5 = self.layer4(C4)
        else:
            C5 = None
        return C1, C2, C3, C4, C5


############################################################
#  Proposal Layer
############################################################
class rpn_graph(nn.Module):
    def __init__(self, input_dims, anchors_per_location,
                 anchor_stride):

        super(rpn_graph, self).__init__()
        # Setup layers
        self.rpn_conv_shared = nn.Conv2d(
            input_dims, 512, kernel_size=3, stride=anchor_stride, padding=1)
        self.rpn_class_raw = nn.Conv2d(
            512, 2 * anchors_per_location, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(
            512, 4 * anchors_per_location, kernel_size=1)
    #[1,256,16,16]*[
    def forward(self, x):
        shared = F.relu(self.rpn_conv_shared(x), True)
        x = self.rpn_class_raw(shared)
        rpn_class_logits = x.permute(
            0, 2, 3, 1).contiguous().view(x.size(0), -1, 2)          
        rpn_probs = F.softmax(rpn_class_logits, dim=-1)
        x = self.rpn_bbox_pred(shared)

        rpn_bbox = x.permute(0, 2, 3, 1).contiguous().view(
            x.size(0), -1, 4)  # reshape to (N, 4)

        return rpn_class_logits, rpn_probs, rpn_bbox


############################################################
#  Bbox Layer
############################################################
class fpn_classifier_graph(nn.Module):
    def __init__(self, num_classes, config):

        super(fpn_classifier_graph, self).__init__()
        self.num_classes = num_classes
        self.config = config
        # Setup layers
        self.mrcnn_class_conv1 = nn.Conv2d(
            256, 1024, kernel_size=self.config.POOL_SIZE, stride=1, padding=0)
        self.mrcnn_class_bn1 = nn.BatchNorm2d(1024, eps=0.001)

#        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.mrcnn_class_conv2 = nn.Conv2d(
            1024, 1024, kernel_size=1, stride=1, padding=0)
        self.mrcnn_class_bn2 = nn.BatchNorm2d(1024, eps=0.001)

        # Classifier head
        self.mrcnn_class_logits = nn.Linear(1024, self.num_classes)
        self.mrcnn_bbox_fc = nn.Linear(1024, self.num_classes * 4)

    def forward(self, x, rpn_rois):
        start = time.time()
        x = ROIAlign(x, rpn_rois, self.config, self.config.POOL_SIZE)

        spend = time.time()-start
        print('first roalign', spend)
        roi_number = x.size()[1]

        x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                   256, self.config.POOL_SIZE,
                   self.config.POOL_SIZE)

        x = self.mrcnn_class_conv1(x)
        x = self.mrcnn_class_bn1(x)
        x = F.relu(x, inplace=True)
#        x = self.dropout(x)
        x = self.mrcnn_class_conv2(x)
        x = self.mrcnn_class_bn2(x)
        x = F.relu(x, inplace=True)

        shared = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        # Classifier head
        mrcnn_class_logits = self.mrcnn_class_logits(shared)
        mrcnn_probs = F.softmax(mrcnn_class_logits, dim=-1)

        x = self.mrcnn_bbox_fc(shared)
        mrcnn_bbox = x.view(x.size()[0], self.num_classes, 4)

        mrcnn_class_logits = mrcnn_class_logits.view(self.config.IMAGES_PER_GPU,
                                                     roi_number,
                                                     mrcnn_class_logits.size()[-1])
        mrcnn_probs = mrcnn_probs.view(self.config.IMAGES_PER_GPU,
                                       roi_number,
                                       mrcnn_probs.size()[-1])
        # BBox head
        # [batch, boxes, num_classes , (dy, dx, log(dh), log(dw))]
        mrcnn_bbox = mrcnn_bbox.view(self.config.IMAGES_PER_GPU,
                                     roi_number,
                                     self.config.NUM_CLASSES,
                                     4)

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


############################################################
#  Mask Layer
############################################################
class build_fpn_mask_graph(nn.Module):

    def __init__(self, num_classes, config):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from diffent layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_shape: [height, width, depth]
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results

        Returns: Masks [batch, roi_count, height, width, num_classes]
        """
        # ROI Pooling
        # Shape: [batch, boxes, pool_height, pool_width, channels]
        super(build_fpn_mask_graph, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        # Setup layers
        self.mrcnn_mask_conv1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn1 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_conv2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn2 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_conv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn3 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_conv4 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn4 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_deconv = nn.ConvTranspose2d(
            256, 256, kernel_size=2, stride=2)

        self.mrcnn_mask = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1)

    def forward(self, x, rpn_bbox,box_index):
        '''
        box_index
        bbox [N,maxB,4]
        sample
        boxes = torch.Tensor([[1, 0, 5, 4],
                             [1, 0, 5, 4],
                             [0.5, 3.5, 4, 7]])

        box_index = torch.tensor([1,0,0], dtype=torch.int) # index of bbox in batch

        x:P2, P3, P4, P5
        '''
        rpn_bbox=rpn_bbox.detach()
        x_1 = rpn_bbox[:, 0]
        y_1 = rpn_bbox[:, 1]
        x_2 = rpn_bbox[:, 2]
        y_2 = rpn_bbox[:, 3]
        roi_level = log2_graph(
          torch.mul(torch.sqrt((y_2 - y_1) * (x_2 - x_1)), 1.0))
        roi_level = torch.clamp(torch.clamp(
          torch.add(torch.round(roi_level), 4), min=0), max=3)
        pooled=[]
        pooled_indices =[]
        original_indices = torch.arange(rpn_bbox.size(0))
        for level in range(4):
            
            level_feature=x[level]
            ixx = torch.eq(roi_level, level)
            ix=ixx.unsqueeze(1).expand(-1,4)
            level_boxes = torch.masked_select(rpn_bbox, ix)
            level_boxes=level_boxes.view(-1, 4)
            level_indices=torch.masked_select(box_index, ixx)
            orig_idx = torch.masked_select(original_indices, ixx)
            if level_boxes.size(0) > 0:
                pooled_feature=self.roi_align(level_feature,torch.cat([level_indices.unsqueeze(1),level_boxes*level_feature.size()[3]], dim=1))
                # print(torch.cat([level_indices.unsqueeze(1),level_boxes], dim=1))
                # print('pooled_feature',pooled_feature.size())
                # print('pooled_feature1',pooled_feature1.size())
                # print(pooled_feature1[0,0,0,0])
                # print(pooled_feature[0,0,0,0])
                pooled.append(pooled_feature)
                pooled_indices.append(orig_idx)
        pooled = torch.cat(pooled, dim=0)
        pooled_indices=torch.cat(pooled_indices, dim=0)
        sorted_indices = torch.argsort(pooled_indices)
        pooled = pooled[sorted_indices]
        pooled = pooled.cuda()
        # pooled B*N,C,H,W


        return pooled


############################################################
#  Main Class of MASK-RCNN
############################################################
class MaskRCNN(nn.Module):
    """
    Encapsulates the Mask RCNN model functionality.
    
    """
    def __init__(self, config, mode='inference'):
        super(MaskRCNN, self).__init__()
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.mode = mode
        self.resnet_graph = resnet_graph(
            Bottleneck, [3, 4, 6, 3], stage5=True)#resnet50

        # feature pyramid layers:
        self.fpn_c5p5 = nn.Conv2d(
            512 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c4p4 = nn.Conv2d(
            256 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c3p3 = nn.Conv2d(
            128 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c2p2 = nn.Conv2d(
            64 * 4, 256, kernel_size=1, stride=1,  padding=0)

        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.scale_ratios = [4, 8, 16, 32]
        self.fpn_p6 = nn.MaxPool2d(
            kernel_size=1, stride=2, padding=0, ceil_mode=False)

        self.anchors = generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                self.config.RPN_ANCHOR_RATIOS,
                                                self.config.BACKBONE_SHAPES,
                                                self.config.BACKBONE_STRIDES,
                                                self.config.RPN_ANCHOR_STRIDE)
        self.anchors = self.anchors.astype(np.float32)

        # RPN Model
        self.rpn = rpn_graph(256, len(self.config.RPN_ANCHOR_RATIOS),
                             self.config.RPN_ANCHOR_STRIDE)

        self.rpn_mask = build_fpn_mask_graph(config.NUM_CLASSES, config)
        self.rpn_class = fpn_classifier_graph(config.NUM_CLASSES, config)

        self.proposal_count = self.config.POST_NMS_ROIS_TRAINING if self.mode == "training"\
            else self.config.POST_NMS_ROIS_INFERENCE
        self.fc=nn.Linear(12544,1024)
        if not self.config.USE_GEO_INFO:
            self.fc1=nn.Linear(1024,1024)
            self.relu1=nn.ReLU()
            self.relu2=nn.ReLU()
            self.outpred=nn.Linear(1024,7)
        self.loc_emb=FCNet()
        self.tang_net=TangNet(num_classes=config.NUM_CLASSES)
        self._initialize_weights()

        pretrained_resnet50 = models.resnet50(pretrained=True)
        model_dict = self.resnet_graph.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_resnet50.state_dict().items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet_graph.load_state_dict(model_dict)

    def forward(self, x, bbox,bbox_indices,geo_info=None):

        start = time.time()
        saved_for_loss = []
        
        C1, C2, C3, C4, C5 = self.resnet_graph(x)
        
        resnet_time = time.time()
       
        # print('resnet spend', resnet_time-start)
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + F.upsample(P5,
                                            scale_factor=2, mode='bilinear')
        P3 = self.fpn_c3p3(C3) + F.upsample(P4,
                                            scale_factor=2, mode='bilinear')
        P2 = self.fpn_c2p2(C2) + F.upsample(P3,
                                            scale_factor=2, mode='bilinear')

        # Attach 3x3 conv to all P layers to get the final feature maps.
        # P2 is 256, P3 is 128, P4 is 64, P5 is 32
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = self.fpn_p6(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.

        self.mrcnn_feature_maps = [P2, P3, P4, P5]
        net_feat = self.rpn_mask(self.mrcnn_feature_maps,bbox,bbox_indices)
        net_feat = self.fc(net_feat.view(net_feat.size(0),-1))
        if self.config.USE_GEO_INFO:
            net_feat=torch.sigmoid(net_feat)
            geo_info=self.loc_emb(geo_info)
            pred=self.tang_net(geo_info,net_feat)
        else:
            net_feat = self.relu1(net_feat)
            net_feat = self.fc1(net_feat)
            net_feat = self.relu2(net_feat)
            pred=self.outpred(net_feat)
        return pred

    #bbox refinment including deltas apply, clip to border, NMS, etc.
    def proposal_layer(self,rpn_bbox):

        # Box deltas [batch, num_rois, 4]
        deltas_mul = Variable(torch.from_numpy(np.reshape(
            self.config.RPN_BBOX_STD_DEV, [1, 1, 4]).astype(np.float32))).cuda()
        deltas = rpn_bbox * deltas_mul

        pre_nms_limit = min(6000, self.anchors.shape[0])




        deltas = torch.gather(deltas, 1, ix)

        _anchors = []
        for i in range(self.config.IMAGES_PER_GPU):
            anchors = Variable(torch.from_numpy(
                self.anchors.astype(np.float32))).cuda()
            _anchors.append(anchors)
        anchors = torch.stack(_anchors, 0) 
    
        pre_nms_anchors = torch.gather(anchors, 1, ix)
        refined_anchors = apply_box_deltas_graph(pre_nms_anchors, deltas)

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]  (1024,1024)
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        window = Variable(torch.from_numpy(window)).cuda()

        refined_anchors_clipped = clip_boxes_graph(refined_anchors, window)

        refined_proposals = []
        for i in range(self.config.IMAGES_PER_GPU):
            indices = nms(
                torch.cat([refined_anchors_clipped.data[i], scores.data[i]], 1), 0.7)
            indices = indices[:self.proposal_count]
            indices = torch.stack([indices, indices, indices, indices], dim=1)
            indices = Variable(indices).cuda()
            proposals = torch.gather(refined_anchors_clipped[i], 0, indices)
            padding = self.proposal_count - proposals.size()[0]
            proposals = torch.cat(
                [proposals, Variable(torch.zeros([padding, 4])).cuda()], 0)
            refined_proposals.append(proposals)

        rpn_rois = torch.stack(refined_proposals, 0)

        return rpn_rois
            
        
    @staticmethod
    def build_loss(saved_for_loss, ground_truths, config):
        #create dict to save loss for visualization
        saved_for_log = OrderedDict()
        #unpack saved log
        predict_rpn_class_logits, predict_rpn_class,\
        predict_rpn_bbox, predict_rpn_rois,\
        predict_mrcnn_class_logits, predict_mrcnn_class,\
        predict_mrcnn_bbox, predict_mrcnn_masks_logits = saved_for_loss

        batch_rpn_match, batch_rpn_bbox, \
        batch_gt_class_ids, batch_gt_boxes,\
        batch_gt_masks, active_class_ids = ground_truths
        

        rpn_rois = predict_rpn_rois.cpu().data.numpy() 
        rpn_rois = rpn_rois[:, :, [1, 0, 3, 2]]
        batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask = stage2_target(rpn_rois, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, config)

#        print(np.sum(batch_mrcnn_class_ids))
        batch_mrcnn_mask = batch_mrcnn_mask.transpose(0, 1, 4, 2, 3)
        batch_mrcnn_class_ids = to_variable(
            batch_mrcnn_class_ids).cuda()
        batch_mrcnn_bbox = to_variable(batch_mrcnn_bbox).cuda()
        batch_mrcnn_mask = to_variable(batch_mrcnn_mask).cuda()   
             
#        print(batch_mrcnn_class_ids)
        # RPN branch loss->classification
        rpn_cls_loss = rpn_class_loss(
            batch_rpn_match, predict_rpn_class_logits)
        
        # RPN branch loss->bbox            
        rpn_reg_loss = rpn_bbox_loss(
            batch_rpn_bbox, batch_rpn_match, predict_rpn_bbox, config)

        # bbox branch loss->bbox
        stage2_reg_loss = mrcnn_bbox_loss(
            batch_mrcnn_bbox, batch_mrcnn_class_ids, predict_mrcnn_bbox)

        # cls branch loss->classification
        stage2_cls_loss = mrcnn_class_loss(
            batch_mrcnn_class_ids, predict_mrcnn_class_logits, active_class_ids, config)
            
        # mask branch loss
        stage2_mask_loss = mrcnn_mask_loss(
            batch_mrcnn_mask, batch_mrcnn_class_ids, predict_mrcnn_masks_logits)                           

        total_loss = rpn_cls_loss + rpn_reg_loss + stage2_cls_loss + stage2_reg_loss + stage2_mask_loss
        saved_for_log['rpn_cls_loss'] = rpn_cls_loss.data[0]
        saved_for_log['rpn_reg_loss'] = rpn_reg_loss.data[0]
        saved_for_log['stage2_cls_loss'] = stage2_cls_loss.data[0]
        saved_for_log['stage2_reg_loss'] = stage2_reg_loss.data[0]
        saved_for_log['stage2_mask_loss'] = stage2_mask_loss.data[0]
        saved_for_log['total_loss'] = total_loss.data[0]

        return total_loss, saved_for_log
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def stage2_target(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
     
    batch_rois = []
    batch_mrcnn_class_ids = []
    batch_mrcnn_bbox = []
    batch_mrcnn_mask = []
                                
    for i in range(config.IMAGES_PER_GPU):
        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
        build_detection_targets(
        rpn_rois[i], gt_class_ids[i], gt_boxes[i], gt_masks[i], config)
    
        batch_rois.append(rois)
        batch_mrcnn_class_ids.append(mrcnn_class_ids)
        batch_mrcnn_bbox.append(mrcnn_bbox)
        batch_mrcnn_mask.append(mrcnn_mask)
        
    batch_rois = np.array(batch_rois)
    batch_mrcnn_class_ids = np.array(batch_mrcnn_class_ids)
    batch_mrcnn_bbox = np.array(batch_mrcnn_bbox)
    batch_mrcnn_mask = np.array(batch_mrcnn_mask)
    return batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask
                        
################
#Loss functions#
################
# region proposal network confidence loss
def rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.    
    anchor_class = torch.eq(rpn_match, 1)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.    
    indices = torch.ne(rpn_match, 0.)

    rpn_class_logits = torch.masked_select(rpn_class_logits, indices)
    anchor_class = torch.masked_select(anchor_class, indices)

    rpn_class_logits = rpn_class_logits.contiguous().view(-1, 2)

    anchor_class = anchor_class.contiguous().view(-1).type(torch.cuda.LongTensor)
    loss = F.cross_entropy(rpn_class_logits, anchor_class, weight=None)
    return loss

# region proposal bounding bbox loss
def rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox, config):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.   
    indices = torch.eq(rpn_match, 1) 
    rpn_bbox = torch.masked_select(rpn_bbox, indices)
    batch_counts = torch.sum(indices.float(), dim=1)
        
    outputs = []
    for i in range(config.IMAGES_PER_GPU):
#        print(batch_counts[i].cpu().data.numpy()[0])
        outputs.append(target_bbox[torch.cuda.LongTensor([i]), torch.arange(int(batch_counts[i].cpu().data.numpy()[0])).type(torch.cuda.LongTensor)])
    
    target_bbox = torch.cat(outputs, dim=0)
    
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox, size_average=True)
    return loss

# rcnn head confidence loss
def mrcnn_class_loss(target_class_ids, pred_class_logits, active_class_ids, config):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """

    # Find predictions of classes that are not in the dataset.
    pred_class_logits = pred_class_logits.contiguous().view(-1, config.NUM_CLASSES)

    target_class_ids = target_class_ids.contiguous().view(-1).type(torch.cuda.LongTensor)
    # Loss
    loss = F.cross_entropy(
        pred_class_logits, target_class_ids, weight=None, size_average=True)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
#    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
#    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss

# rcnn head bbox loss
def mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = target_class_ids.contiguous().view(-1)
    target_bbox = target_bbox.contiguous().view(-1, 4)
    pred_bbox = pred_bbox.contiguous().view(-1, pred_bbox.size()[2], 4)
#    print(target_class_ids)

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = torch.gt(target_class_ids , 0)
#    print(positive_roi_ix)
    positive_roi_class_ids = torch.masked_select(target_class_ids, positive_roi_ix)
    
    indices = target_class_ids
#    indices = torch.stack([positive_roi_ix, positive_roi_class_ids], dim=1)
#    print(indices)
    # Gather the deltas (predicted and true) that contribute to loss
#    target_bbox = torch.gather(target_bbox, positive_roi_ix)
#    pred_bbox = torch.gather(pred_bbox, indices)

    loss = F.smooth_l1_loss(pred_bbox, target_bbox, size_average=True)
    return loss

# rcnn head mask loss
def mrcnn_mask_loss(target_masks, target_class_ids, pred_masks_logits):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = target_class_ids.view(-1)

    loss = F.binary_cross_entropy_with_logits(pred_masks_logits, target_masks)
    return loss                            

################
#Geo prior#
################
# combine net feature and geo feature

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs=4, num_classes=7, num_filts=128, num_users=1):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

        self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
                                    nn.ReLU(inplace=True),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts))

    def forward(self, x, class_of_interest=None, return_feats=True):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class TangNet(nn.Module):
    def __init__(self, ip_loc_dim=128, feats_dim=1024, loc_dim=256, num_classes=8, use_loc=True):
        super(TangNet, self).__init__()
        self.use_loc  = use_loc
        self.fc_loc   = nn.Linear(ip_loc_dim, loc_dim)
        if self.use_loc:
            self.fc_class = nn.Linear(feats_dim+loc_dim, num_classes)
        else:
            self.fc_class = nn.Linear(feats_dim, num_classes)

    def forward(self, loc, net_feat):
        if self.use_loc:
            x = torch.sigmoid(self.fc_loc(loc))
            x = self.fc_class(torch.cat((x, net_feat), 1))
        else:
            x = self.fc_class(net_feat)
        return F.log_softmax(x, dim=1)