import numpy as np
from scipy.ndimage import distance_transform_edt
from rasterio import features
import geopandas as gpd
import rasterio
import os
from tqdm import tqdm
from WBT.whitebox_tools import WhiteboxTools
import argparse
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR

wbt = WhiteboxTools()
wbt.set_working_dir(os.path.dirname(os.path.abspath(__file__)) + "/data/")

def shapefile_to_raster(shapefile_path, reference_tif_path, output_raster_path):
    # 读取参考 TIFF（用于获取元数据）
    with rasterio.open(reference_tif_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    # 读取 Shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # 若坐标系不同则重投影
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # 栅格化：所有要素填充为 1，背景为 0
    shapes = ((geom, 1) for geom in gdf.geometry)

    rasterized = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # 写入新 TIFF
    meta.update({
        "count": 1,
        "dtype": 'uint8'
    })

    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(rasterized, 1)

    print(f"Raster saved to: {output_raster_path}")

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Building Post process Pipeline')
  parser.add_argument('--city', dest='city',
                      help='city name',
                      default="Shanghai", type=str)
  parser.add_argument('--class_name', dest='class_name',
                        help='building function class',
                        default=1, type=int)
  args = parser.parse_args()
  return args

# 设置文件路径
if __name__ == '__main__':
    city_name = 'Jiaxing'
    root = ROOT_DIR
    input_folder = os.path.join(root, 'temp')
    output_folder = os.path.join(root, 'access_process','result')
    class_dict={1:'Residential',2:'Commercial',3:'Public_service',4:'Public_health',5:'Sport_and_art',6:'Educational',7:'Industrial',8:'Administration'}
    for class_name in [3,4,6]:
        class_name_en=class_dict[class_name]
        shapefile_path = os.path.join(input_folder,f'{class_name_en}.shp')
        source_raster_path = os.path.join(input_folder,f'{class_name_en}.tif')
        cost_raster_path = os.path.join(input_folder,'costmap_merged.tif')
        out_accum = os.path.join(output_folder,f'accum_{class_name_en}.tif')
        out_backlink = os.path.join(output_folder,f'backlink_{class_name_en}.tif')
        # 将 Shapefile 转换为栅格
        shapefile_to_raster(shapefile_path, cost_raster_path, source_raster_path)

        # 计算成本距离

        info=wbt.cost_distance(
            source_raster_path, 
            cost_raster_path, 
            out_accum, 
            out_backlink, 
        )
        print(info)