import numpy as np
from osgeo import gdal, ogr, osr
from scipy.ndimage import distance_transform_edt
import os
from tqdm import tqdm
from WBT.whitebox_tools import WhiteboxTools
import argparse

wbt = WhiteboxTools()
wbt.set_working_dir(os.path.dirname(os.path.abspath(__file__)) + "/data/")

def shapefile_to_raster(shapefile_path, reference_tif_path, output_raster_path):
    # 打开 Shapefile
    source_ds = ogr.Open(shapefile_path)
    source_layer = source_ds.GetLayer()

    # 获取参考 TIFF 文件的地理信息
    raster_ds = gdal.Open(reference_tif_path)
    geotransform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()
    x_res = raster_ds.RasterXSize
    y_res = raster_ds.RasterYSize

    # 创建一个新的栅格文件
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(output_raster_path, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)

    # 将 Shapefile 栅格化
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

    # 关闭数据集
    target_ds = None
    source_ds = None
    raster_ds = None

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
                        default=1, type=str)
  args = parser.parse_args()
  return args

# 设置文件路径
if __name__ == '__main__':
    args=parse_args()
    root='/home/ashelee/llx/OSM'
    city_name=args.city
    class_name_en=args.class_name
    class_dict={1:'Residential',2:'Commercial',3:'Public_service',4:'Public_health',5:'Sport_and_art',6:'Educational',7:'Industrial',8:'Administration'}
    # class_name_en='comprehensive_hospital'
    root1='/home/ashelee/llx/display/OSM_display'
    shapefile_path = os.path.join('/home/ashelee/llx/display/OSM_display',class_name_en,f'{city_name}.shp')
    if not os.path.exists(os.path.join(root1,'temp',city_name)):
        os.makedirs(os.path.join(root1,'temp',city_name))
    source_raster_path = os.path.join(root1,'temp',city_name,f'{city_name}_{class_name_en}.tif')
    cost_raster_path = os.path.join(root,city_name,'merged.tif')
    if not os.path.exists(os.path.join(root1,'result',city_name)):
        os.makedirs(os.path.join(root1,'result',city_name))
    out_accum = os.path.join(root1,'result',city_name,f'accum_{class_name_en}.tif')
    out_backlink = os.path.join(root1,'temp',city_name,f'backlink_{class_name_en}.tif')
    # 将 Shapefile 转换为栅格
    shapefile_to_raster(shapefile_path, cost_raster_path, source_raster_path)

    # 计算成本距离
    wbt.cost_distance(
        source_raster_path, 
        cost_raster_path, 
        out_accum, 
        out_backlink, 
    )