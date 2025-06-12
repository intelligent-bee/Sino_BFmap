import numpy as np
import argparse
from shp2tif import *
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
    output_path=os.path.join(output_folder,city_name+'_origin.shp')
    city_root=os.path.join(ROOT_DIR,'Building_mapping_02_post_processing','data',city_name)
    pred_box_path=os.path.join(city_root,'pred')
    img_path=os.path.join(city_root,'footprint')
    output_root=os.path.join(ROOT_DIR,'Building_mapping_02_post_processing','result')
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    origin_path=os.path.join(output_folder,f'{city_name}_origin.shp')
    output_pred_path=os.path.join(output_folder,f'{city_name}_pred.shp')
    output_padding_path=os.path.join(output_root,f'{city_name}_padding.shp')
    add_box2shp(img_path=img_path,pred_box_path=pred_box_path,footprint_path=footprint_path,output_path=output_pred_path)
    padding_shp(origin=origin_path,pred=output_pred_path,class_name='class',output=output_padding_path,display=False)

        
