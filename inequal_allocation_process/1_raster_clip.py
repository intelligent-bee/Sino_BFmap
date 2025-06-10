import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import CRS
from rasterio.warp import reproject, calculate_default_transform
import time
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR

def clip_and_resample_tif_by_shp(input_tif, shapes, output_tif, target_crs, target_resolution=1.0, resampling_method=Resampling.bilinear):
    # 加载shapefile，获取其地理范围

    with rasterio.open(input_tif) as src:
        # 创建一个窗口，覆盖shapefile的地理范围
        print(src.crs)
        print(shapes.crs)
        shp_bounds = shapes.total_bounds  # 获取shapefile的总地理范围
        window = from_bounds(*shp_bounds, transform=src.transform)
        transform = src.window_transform(window)
              
        # 计算输出栅格的形状（分辨率）
        out_width = int((shp_bounds[2] - shp_bounds[0]) / target_resolution)
        out_height = int((shp_bounds[3] - shp_bounds[1]) / target_resolution)
        
        new_transform = rasterio.transform.from_bounds(
            shp_bounds[0], shp_bounds[1], shp_bounds[2], shp_bounds[3], 
            out_width, out_height
        )

        # 创建一个新的空数组来存储重采样结果
        out_image = np.empty((src.count, out_height, out_width), dtype=src.dtypes[0])

        # 重采样栅格数据
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=out_image[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=resampling_method
            )
        
        # 更新元数据
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "transform": new_transform,
            "crs": target_crs,

        })
        
        # 保存裁剪和重采样后的栅格
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)

def reproject_tif(input,output,target_crs):

    # 读取原始 TIFF 文件
    with rasterio.open(input) as src:
        # 获取原始 CRS 和仿射变换信息
        src_crs = src.crs
        transform, width, height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height, *src.bounds)
        
        # 重投影的输出参数
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            "nodata":0
        })

        # 打开输出文件
        with rasterio.open(output, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear  # 可选重采样方法
                )

    print(f"Reprojected TIFF file saved to {output}")
if __name__ == "__main__":
    root = ROOT_DIR
    input_folder = os.path.join(root, 'input_data')
    city_name = 'Jiaxing'
    population_path = os.path.join(input_folder,'PopSE_China2020_100m_rep.tif')
    bh_path = os.path.join(input_folder,f'{city_name}_BH_rep.tif')
    shp_file = os.path.join(input_folder,'building_map', f'{city_name}.shp')
    shapes_file = gpd.read_file(shp_file)
    output = os.path.join(root, 'temp', f'{city_name}_population.tif')
    output_height=os.path.join(root, 'temp', f'{city_name}_height.tif')
    prj_file_path = os.path.join(input_folder,'Albers_China.prj')
    with open(prj_file_path, 'r') as prj_file:
        target_crs_wkt = prj_file.read()
        crs_from_prj = CRS.from_wkt(target_crs_wkt)
    shapes_file = shapes_file.to_crs(crs_from_prj)   
    bh_out=os.path.join(root,'temp',f'{city_name}_BH.tif')
    print('read finished\n')
    current=time.time()
    reproject_tif(bh_path,bh_out,crs_from_prj)
    print('reproject time:',time.time()-current)
    current=time.time()
    clip_and_resample_tif_by_shp(population_path, shapes_file, output,crs_from_prj,target_resolution=2.0)
    print('population clip time:',time.time()-current)
    current=time.time()
    clip_and_resample_tif_by_shp(bh_out, shapes_file, output_height,crs_from_prj,target_resolution=2.0)
    print('height clip time:',time.time()-current)
    