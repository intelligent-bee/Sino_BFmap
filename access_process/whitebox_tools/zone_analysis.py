import numpy as np
from osgeo import gdal, ogr, osr
from scipy.ndimage import distance_transform_edt
import os
from tqdm import tqdm
from WBT.whitebox_tools import WhiteboxTools
import argparse
from glob import glob

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
                        default=1, type=int)
  args = parser.parse_args()
  return args
# 读取建筑物轮廓矢量数据
if __name__ == "__main__":
    args=parse_args()
    city_name=args.city
    root='/home/ashelee/llx/Population'
    shp_root='/home/ashelee/llx/preprocess_shp'


    # 读取人口密度和建筑物高度栅格数据
    pop_path=os.path.join(root,'population_tif',f'{city_name}_population.tif')
    # nodata=pop_density_src.nodata
    # pop_density_raster = pop_density_src.read()[0]
    # pop_density_raster[pop_density_raster<=0]=0
    tif_root='/home/ashelee/Building_classification/All_City_GIU-BH'
    for level_file in os.listdir(tif_root):
        if city_name in os.listdir(os.path.join(tif_root,level_file)):
            bh_path = glob(os.path.join(tif_root,level_file,city_name, f'*{city_name}_BH_rep.tif'))[0]
            break
    feature_shp=os.path.join(shp_root,city_name, f'{city_name}_padding.shp')
    shapefile_to_raster(feature_shp, bh_path, f'/home/ashelee/llx/temp/{city_name}_shp.tif')
    
    wbt = WhiteboxTools()
    wbt.set_working_dir(os.path.dirname(os.path.abspath(__file__)) + "/data/")


    wbt.zonal_statistics(
        bh_path, 
        f'/home/ashelee/llx/temp/{city_name}_shp.tif', 
        output=f'/home/ashelee/llx/temp/{city_name}_bh.tif', 
        stat="mean", 
        out_table=f'/home/ashelee/llx/temp/{city_name}_bh.html', 
    )