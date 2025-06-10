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

if __name__ == '__main__':
    city_name_list=['Jiaxing']
    for city_name in city_name_list:
        print(city_name)
        current=time.time()
        root=ROOT_DIR
        shp_root=os.path.join(root,'input_data','building_map')
        buildings = gpd.read_file(os.path.join(shp_root,f'{city_name}.shp'))
        prj_file_path = os.path.join(root,'input_data','Albers_China.prj')
        with open(prj_file_path, 'r') as prj_file:
            target_crs_wkt = prj_file.read()
            crs_from_prj = CRS.from_wkt(target_crs_wkt)


        buildings = buildings.to_crs(crs_from_prj) 
        buildings['idx'] = buildings.index
        buildings['Nearby_Buildings'] = None

        def remove_z(geometry):
            # 将几何数据转换为字典（GeoJSON风格）
            geom_2d = mapping(geometry)
            # 处理 'Polygon' 类型的几何对象
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
        print('load_time:',time.time()-current)
        sindex = buildings.sindex  

        def find_nearby_buildings(idx_row_tuple):
            idx, row = idx_row_tuple
            if row['buffer'] is None or row['buffer'].is_empty:
                return idx, {}

            possible_matches_index = list(sindex.intersection(row['buffer'].bounds))
            possible_matches = buildings.iloc[possible_matches_index]
            nearby_buildings = possible_matches[possible_matches.geometry.intersects(row['buffer']) & 
                                                (possible_matches.index != idx)]
            building_counts = nearby_buildings['class'].value_counts().to_dict()
            return idx, building_counts

        total_start_time = time.time()
        batch_size = 5000

        # Initialize a dictionary to accumulate all updates
        all_updates = {}

        for i in tqdm(range(0, len(buildings), batch_size), miniters=10, maxinterval=150):
            buildings_batch = buildings.iloc[i:i + batch_size].copy()
            buildings_batch['buffer'] = buildings_batch.geometry.buffer(1500)
            
            # Use the precomputed spatial index for the entire dataset
            with Pool(processes=8) as pool:
                results = pool.map(find_nearby_buildings, buildings_batch.iterrows())
            
            # Merge the updates from the current batch into the accumulated dictionary
            all_updates.update({idx: json.dumps(building_counts) for idx, building_counts in results})

        # Apply the accumulated updates to the DataFrame
        buildings.update(pd.DataFrame.from_dict(all_updates, orient='index', columns=['Nearby_Buildings']))

        total_end_time = time.time()

        print(f"Total processing time: {total_end_time - total_start_time} seconds")

        buildings_filtered = buildings[['Nearby_Buildings', 'class','idx']]
        buildings_filtered.to_csv(os.path.join(root,'availability_diversity_process/result',f'{city_name}_buildings.csv'), index=False)
        print('save_time:',time.time()-current)
