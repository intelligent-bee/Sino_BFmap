import argparse
import glob
from building_preprocess import *  #raster2shp1
from utils import *
from shp2tif import shp2tif,shp2tif_gdal
import os
from tqdm import tqdm as tqdm
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Building Post process Pipeline')
  parser.add_argument('--city', dest='city',
                      help='city name',
                      default="Jiaxing", type=str)
  parser.add_argument('--input_tif', dest='input_tif',
                      help='the result of segmentation',
                      default="/data/ashelee/Building_classification/model/Jiaxing_HRNet_7band_std_w19/Jiaxing_HRNet_7band_std_w19_epoch60_OL.tif", type=str)
  args = parser.parse_args()
  return args
if __name__ == '__main__':
    args=parse_args()  
    root=ROOT_DIR
    city_name=args.city
    output_folder=os.path.join(ROOT_DIR,'temp')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    footprint_path=os.path.join(ROOT_DIR,'dataset',str(city_name),f'BM_{city_name}.shp')
    raster_path=args.input_tif
    output_path=os.path.join(output_folder,city_name+'_origin.shp')
    # dataset_root=os.path.join(ROOT_DIR,'dataset')
    dataset_root=os.path.join(ROOT_DIR,'dataset')
    raster2shp1(raster_path,footprint_path,output_path)
    print(f'{city_name} preprocess done\n')


    data_output_root=os.path.join(ROOT_DIR,'Building_mapping_02_post_processing','data')
    if not os.path.exists(data_output_root):
        os.makedirs(data_output_root)

    split_image(os.path.join(dataset_root,city_name),outputroot=data_output_root,start=0,tile_size=(256, 256),ignore_list=[0])  
    split_image(os.path.join(dataset_root,city_name),outputroot=data_output_root,start=128,tile_size=(256, 256),ignore_list=[0])
    
    shp2tif(root=os.path.join(dataset_root,city_name),outputroot=data_output_root,footprint=footprint_path)  #切出原始的footprint
    split_image_test(os.path.join(dataset_root,city_name),outputroot=data_output_root,tile_size=(256, 256)) #切块作为最终测试
    print(f'Split finished\n')

