import geopandas as gpd
from shapely.geometry import Point
import os
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from shapely.geometry import box
from shapely.validation import make_valid
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import pyproj
from pyproj import CRS
import time
from multiprocessing import Pool, cpu_count
import json
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR

if __name__ == '__main__':

    root = ROOT_DIR
    input_folder = os.path.join(root, 'input_data')
    city_name = 'Jiaxing'
    population_path = os.path.join(input_folder,'PopSE_China2020_100m_rep.tif')
    bh_path = os.path.join(input_folder,f'{city_name}_BH_rep.tif')
    shp_file = os.path.join(input_folder,'building_map', f'{city_name}.shp')
    buildings = gpd.read_file(shp_file)
    prj_file_path = os.path.join(input_folder,'Albers_China.prj')
    with open(prj_file_path, 'r') as prj_file:
        target_crs_wkt = prj_file.read()
        crs_from_prj = CRS.from_wkt(target_crs_wkt)
    bh_out=os.path.join(root,'temp',f'{city_name}_BH.tif')


    raster_path = os.path.join(root, 'temp', f'{city_name}_population.tif')
    population_raster = rasterio.open(raster_path)
    height_raster_path=os.path.join(root, 'temp', f'{city_name}_height.tif')
    height_raster=rasterio.open(height_raster_path)
    raster_bounds=population_raster.bounds
    target_crs = population_raster.crs

    buildings = buildings.to_crs(crs_from_prj)



    buildings['idx'] = buildings.index
    buildings['Population'] = 0
    buildings['Height'] = 0

    # buildings=buildings[buildings['class']==1]

    def remove_z(geometry):
        geom_2d = mapping(geometry)
        if geom_2d['type'] == 'Polygon':
            geom_2d['coordinates'] = [
                [(value[0],value[1]) for value in ring] for ring in geom_2d['coordinates']
            ]
            
        # 处理 'MultiPolygon' 类型的几何对象
        elif geom_2d['type'] == 'MultiPolygon':
            geom_2d['coordinates'] = [
                [[(value[0],value[1]) for value in ring] if len(ring)>2 else ring for ring in polygon] 
                for polygon in geom_2d['coordinates']
            ]
        # 将字典形式转换回几何对象
        return shape(geom_2d)

    buildings['geometry'] = buildings['geometry'].apply(lambda geom: remove_z(geom) if geom.is_valid and not geom.is_empty else None)
    buildings = buildings[buildings['geometry'].notnull()]
    def calculate_population(idx_row_tuple,population_raster,height_raster):
        # population_raster = rasterio.open(raster_path)
        # raster_bounds=population_raster.bounds
        # target_crs = population_raster.crs
        idx, row = idx_row_tuple
        geometry = row['geometry']
        
        # 首先检查建筑物是否与栅格范围重叠
        if not geometry.intersects(box(*raster_bounds)):
            # 如果不重叠，直接返回人口数为 0
            return idx, 0,0
        
        # 将建筑物几何形状转换为GeoJSON格式，用于mask
        geojson_geom = [mapping(geometry)]
        
        # 裁剪栅格数据
        out_image, out_transform = mask(population_raster, geojson_geom, crop=True,nodata=population_raster.nodata)
        out_image = out_image[0]  # 取出单一波段的数据
        out_image=out_image[out_image!=0]
        
        # 检查裁剪结果是否为空或仅包含nodata值
        if out_image.size == 0 or np.all(out_image == population_raster.nodata):
            population_count = 0
        else:
            # 统计人口总数，忽略无效数据（无效数据通常为栅格中的 nodata 值）
            population_count = out_image[out_image != population_raster.nodata].flatten().mean()

        out_image, out_transform = mask(height_raster, geojson_geom, crop=True,nodata=height_raster.nodata)
        out_image = out_image[0]  # 取出单一波段的数据
        
        # 检查裁剪结果是否为空或仅包含nodata值
        if out_image.size == 0 or np.all(out_image == height_raster.nodata):
            height_count = 0
        else:
            # 统计人口总数，忽略无效数据（无效数据通常为栅格中的 nodata 值）
            height_count = out_image[out_image != height_raster.nodata].flatten().mean()
            # print(height_count)
        
        return idx, population_count,height_count


    total_start_time = time.time()
    batch_size = 5000
    all_updates_pop = {}
    all_updates_height = {}
    # 分批次处理
    for i in tqdm(range(0, len(buildings), batch_size)):
        buildings_batch = buildings.iloc[i:i + batch_size].copy()
        
        # with Pool(processes=1) as pool:
        #     results = pool.map(calculate_population, buildings_batch.iterrows())
        
        # # 将结果存储到GeoDataFrame
        # update_dict = {idx: population_count for idx, population_count in results}
        update_dict = {}
        update_dict_height = {}
        for idx, row in buildings_batch.iterrows():
            idx, population_count,height_count = calculate_population((idx, row), population_raster,height_raster)
            update_dict[idx] = population_count
            update_dict_height[idx] = height_count
        buildings.update(pd.DataFrame.from_dict(update_dict, orient='index', columns=['Population']))
        buildings.update(pd.DataFrame.from_dict(update_dict_height, orient='index', columns=['Height']))

    # 总处理结束时间
    total_end_time = time.time()

    current=time.time()
    buildings['Area'] = buildings['geometry'].area
    buildings_filtered = buildings[[ 'idx','class', 'Population','Height','Area']]

    buildings_filtered.to_csv(os.path.join(root,'inequal_allocation_process/result',f'{city_name}.csv'), index=False)
    population_raster.close()
    height_raster.close()