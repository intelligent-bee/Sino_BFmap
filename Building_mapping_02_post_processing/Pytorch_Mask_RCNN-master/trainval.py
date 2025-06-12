# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import  DataLoader

from preprocess.data_center import CocoDataset,det_mask_collate,NormalizeImage
# from network.mask_rcnn import MaskRCNN
from network.mask_backbone import MaskRCNN,FCNet
from network.focal_loss import FocalLoss
# from roi_align import RoIAlign
# from roi_align import CropAndResize
import math
import random
from tqdm import tqdm as tqdm
import pandas as pd
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
ROOT_DIR = parent_dir
def log2_graph(x):
    """Implementatin of Log2. pytorch doesn't have a native implemenation."""
    return torch.div(torch.log(x), math.log(2.))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def update_confusion_matrix(conf_matrix, preds, labels):
    """
    更新混淆矩阵。
    conf_matrix: 当前的混淆矩阵
    preds: 当前批次的预测结果
    labels: 当前批次的真实标签
    """
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix

def calculate_micro_f1(conf_matrix):
    """
    计算Micro-F1分数。
    conf_matrix: 累计的混淆矩阵
    """
    true_positive = conf_matrix.diag().sum().item()
    predicted_positive = conf_matrix.sum(0).sum().item()
    actual_positive = conf_matrix.sum(1).sum().item()

    precision = true_positive / predicted_positive if predicted_positive > 0 else 0
    recall = true_positive / actual_positive if actual_positive > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--city', dest='city',
                      help='city name',
                      default="Jiaxing", type=str)


  args = parser.parse_args()
  return args
from config import Config

class CocoTrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "coco_train"
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    USE_GEO_INFO = True

    TRAIN_EPOCH = 20
    NUM_CLASSES = 8
    BATCH_SIZE = 128


def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable
def calculate_precision_recall_f1(conf_matrix):
    """
    计算每个类别的精确度、召回率和F1分数。
    conf_matrix: 混淆矩阵，其中行代表真实类别，列代表预测类别。
    """
    # 精确度 = TP / (TP + FP) 对于每一列
    precision = torch.diag(conf_matrix) / conf_matrix.sum(0)
    # 召回率 = TP / (TP + FN) 对于每一行
    recall = torch.diag(conf_matrix) / conf_matrix.sum(1)
    # F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall)

    precision[precision != precision] = 0  # 将NaN替换为0
    recall[recall != recall] = 0           # 将NaN替换为0
    f1_scores[f1_scores != f1_scores] = 0  # 将NaN替换为0

    return precision, recall, f1_scores

def calculate_macro_f1(f1_scores):
    """
    计算Macro-F1分数。
    f1_scores: 每个类别的F1分数数组。
    """
    return f1_scores.mean()
def train(args,cfg,model,train_loader,val_loader,optimizer,criterion,scheduler,output_dir):
    best_acc=0.0
    best_epoch=0
    num_classes=cfg.NUM_CLASSES
    for epoch in tqdm(range(cfg.TRAIN_EPOCH)):
      model.train()
      running_loss = 0.0
      running_corrects = 0
      train_count=0.0

      for batch_data in tqdm(train_loader,miniters=10,maxinterval=100):
        image,num_boxes,padded_class, bbox, padded_geo_info = batch_data
        # print(image.size(0))
        # print('\n')

        actual_boxes = []
        actual_classes=[]
        actual_info=[]
        indices = []
        for i in range(bbox.size(0)):  #在这里还原，从maxbox到actualbox 原box的尺寸是[N,maxbox,4]->[N,actualbox,4]
            num = num_boxes[i].item()
            actual_boxes.append(bbox[i, :num, :])
            actual_classes.append(padded_class[i, :num, :])
            actual_info.append(padded_geo_info[i, :num, :])
            indices.extend([i] * num)
        actual_boxes = torch.cat(actual_boxes, dim=0)
        actual_classes = torch.cat(actual_classes, dim=0)
        # if image.size(0)==1:
        #   print(actual_classes.shape)
        actual_classes=torch.tensor(actual_classes,dtype=torch.long).squeeze(1)
        # if image.size(0)==1:
        #   print(actual_classes.shape)

        optimizer.zero_grad()


        indices = torch.tensor(indices, dtype=torch.int)

        if cfg.USE_GEO_INFO:
          actual_info = torch.cat(actual_info, dim=0)
          image,actual_boxes,indices,actual_info,actual_classes=image.cuda(),actual_boxes.cuda(),indices.cuda(),actual_info.cuda(),actual_classes.cuda()
          pred=model(image,actual_boxes,indices,actual_info)
        else:
          image,actual_boxes,indices,actual_classes=image.cuda(),actual_boxes.cuda(),indices.cuda(),actual_classes.cuda()
          pred=model(image,actual_boxes,indices)

        loss=criterion(pred,actual_classes)
        # print(loss,actual_classes.size(0))
        loss.backward()
        optimizer.step()

        running_loss += loss.cpu().item() * actual_classes.size(0)
        train_count+=actual_classes.cpu().size(0)
      print('Epoch:{} Loss: {:.4f}'.format(epoch, running_loss/train_count))
      # if epoch % 5 == 0 or epoch == train_epoch_num - 1:
      if True:
        print(f'validation in epoch {epoch}')
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0.0
        count=0.0
        class_correct = np.zeros(num_classes,dtype=np.float64)
        class_total = np.zeros(num_classes,dtype=np.float64)
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        with torch.no_grad():
          for batch_data in tqdm(val_loader, mininterval=1):
            # print(len(batch_data))
            image,num_boxes,padded_class, bbox, padded_geo_info = batch_data
            

            actual_boxes = []
            actual_classes=[]
            actual_info=[]
            indices = []
            for i in range(bbox.size(0)):
                num = num_boxes[i].item()
                actual_boxes.append(bbox[i, :num, :])
                actual_classes.append(padded_class[i, :num, :])
                actual_info.append(padded_geo_info[i, :num, :])
                indices.extend([i] * num)
            actual_boxes = torch.cat(actual_boxes, dim=0)
            actual_classes = torch.cat(actual_classes, dim=0)
            actual_classes=torch.tensor(actual_classes,dtype=torch.long).squeeze(1)


            actual_info = torch.cat(actual_info, dim=0)
            indices = torch.tensor(indices, dtype=torch.int)

            image,actual_boxes,indices,actual_info,actual_classes=image.cuda(),actual_boxes.cuda(),indices.cuda(),actual_info.cuda(),actual_classes.cuda()

            pred=model(image,actual_boxes,indices,actual_info)
            loss=criterion(pred,actual_classes)
            _, pred = torch.max(pred, 1)
            label=actual_classes.cpu().view(-1)
            conf_matrix = update_confusion_matrix(conf_matrix, pred, label)
            for i in range(num_classes):
              class_mask = (label == i)
              class_correct[i] += (pred[class_mask] == i).sum().item()
              class_total[i] += class_mask.sum().item()

            count+=actual_classes.cpu().size(0)
            val_running_loss += loss.item() * actual_classes.size(0)
            val_running_corrects += torch.sum(pred.cpu() == actual_classes.cpu())

        val_epoch_loss = val_running_loss / count
        val_epoch_acc = val_running_corrects.double() / count
        class_accuracy = class_correct / class_total
        output_str = ""
        output_str += 'Epoch: {}\t'.format(epoch)
        output_str += 'Val Loss: {:.4f} Acc: {:.4f}\n'.format(val_epoch_loss, val_epoch_acc)
        output_str += 'class_accuracy: {}\n'.format(class_accuracy)
        precision, recall, f1_scores = calculate_precision_recall_f1(conf_matrix)
        macro_f1 = calculate_macro_f1(f1_scores).item()
        output_str += "Macro-F1 Score: {}\t".format(macro_f1)
        output_str += 'Precision: {}\t'.format(precision)
        output_str += 'best epoch: {} best F1: {}\n\n'.format(best_epoch, best_acc)
        with open(os.path.join(output_dir,'running_logs.txt'), 'a') as f:
          f.write(output_str)
        if macro_f1 > best_acc:
            best_acc = macro_f1
            best_epoch=epoch
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            np.save(os.path.join(output_dir,'conf_matrix.npy'),conf_matrix.numpy())
      scheduler.step()
    with open(os.path.join(output_dir,'running_logs.txt'), 'a') as f:
      f.write(f'best epoch: {best_epoch} best F1: {best_acc}\n')
def test(args,cfg,model,test_loader,output_dir,save_dir):
  pretrain_model=os.path.join(output_dir, 'best_model.pth')
  model.load_state_dict(torch.load(pretrain_model))
  model.eval()

  batch_size=cfg.BATCH_SIZE
  num_classes = cfg.NUM_CLASSES
  pred_count={}

  with torch.no_grad():
    for batch_data in tqdm(test_loader,miniters=10,maxinterval=100):

      image,num_boxes,padded_class, bbox, padded_geo_info,img_ids = batch_data

      actual_boxes = []
      actual_classes=[]
      actual_info=[]
      indices = []
      for i in range(bbox.size(0)):
          num = num_boxes[i].item()
          actual_boxes.append(bbox[i, :num, :])
          actual_classes.append(padded_class[i, :num, :])
          actual_info.append(padded_geo_info[i, :num, :])
          indices.extend([i] * num)
      actual_boxes = torch.cat(actual_boxes, dim=0)
      actual_classes = torch.cat(actual_classes, dim=0)
      actual_classes=torch.tensor(actual_classes,dtype=torch.long).squeeze()


      actual_info = torch.cat(actual_info, dim=0)
      indices = torch.tensor(indices, dtype=torch.int)

      image,actual_boxes,indices,actual_info,actual_classes=image.cuda(),actual_boxes.cuda(),indices.cuda(),actual_info.cuda(),actual_classes.cuda()

      pred=model(image,actual_boxes,indices,actual_info)
      _, pred = torch.max(pred, 1)
      pred,actual_boxes,indices=pred.cpu(),actual_boxes.cpu(),indices.cpu()
      actual_boxes = torch.mul(actual_boxes,float(256))
      
      for i in range(len(img_ids)):
        box_data=[]
        mask=torch.eq(indices,i)
        img_pred=pred[mask].detach().numpy()
        img_boxes=actual_boxes[mask].detach().numpy()
        img_boxes=img_boxes.astype(np.uint16)
        img_id=img_ids[i]
        save_path=os.path.join(save_dir,img_id+'.csv')
        for idx,box in enumerate(img_boxes):
          xmin,ymin,xmax,ymax=box
          box_data.append([img_pred[idx],xmin,ymin,xmax,ymax])
          pred_count[img_pred[idx]]=pred_count.get(img_pred[idx],0)+1
        df = pd.DataFrame(box_data, columns=['class', 'x_min', 'y_min', 'x_max', 'y_max'])
        df.to_csv(save_path,index=False)
    with open(os.path.join(output_dir,'test_logs.txt'), 'w') as f:
      for i in range(num_classes):
        f.write(f'class {i} count: {pred_count.get(i,0)}\n')
     
def validate():
    pass
if __name__ == '__main__':

  args = parse_args()
  cfg = CocoTrainConfig()
  setup_seed(20)
  import warnings
  warnings.filterwarnings("ignore")
  
  city_name=args.city
  root=os.path.join(ROOT_DIR,'Building_mapping_02_post_processing')
  normalize = NormalizeImage((123.675, 116.28,103.53),(58.395,57.12,57.375))
  train_dataset = CocoDataset(config = cfg,transform=normalize,root=os.path.join(root,'data',city_name,'train_data'))
  val_dataset = CocoDataset(config = cfg,transform=normalize,root=os.path.join(root,'data',city_name,'val_data'))
  test_dataset = CocoDataset(config = cfg,transform=normalize,root=os.path.join(root,'data',city_name,'test_data'),data_test=True)
  print(city_name)
  print(len(train_dataset))
  train_loader = DataLoader(train_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True,num_workers=8)
  val_loader = DataLoader(val_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True,num_workers=8)
  test_loader = DataLoader(test_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True,num_workers=8)
  #model
  mask_rcnn = MaskRCNN(config=cfg)
  mask_rcnn = mask_rcnn.cuda()

  criterion = nn.CrossEntropyLoss()
  # class_num=np.array([22868,4442,742,1015,3491,7198,6825],dtype=np.float32)
  # class_num=class_num**-1
  # class_num=class_num/class_num.sum()
  # alpha=torch.tensor(class_num,dtype=torch.float32).cuda()
  # criterion=FocalLoss(gamma=1.5, alpha=alpha)
  
  optimizer=optim.Adam(mask_rcnn.parameters(),lr=1e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
  output_dir = os.path.join(root,'checkpoint')
  save_dir = os.path.join(root,'data',city_name,'pred')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  train(args,cfg,mask_rcnn,train_loader,val_loader,optimizer,criterion,scheduler,output_dir)
  test(args,cfg,mask_rcnn,test_loader,output_dir,save_dir)



  

  






