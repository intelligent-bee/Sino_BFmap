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
from scipy.interpolate import interp1d
import ast
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR,CITY_NAME


def get_result(input,output):
    df=pd.read_csv(input)
    df=df[df['class']==1]
    near_buildings=df['Nearby_Buildings'].values
    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        temp=ast.literal_eval(row['Nearby_Buildings'])        
        build_availability = sum(v for k, v in temp.items() if k != "7.0")
        
        type_sum = sum(v for k, v in temp.items() if k != "7.0" and k != "1.0")
        if type_sum !=0:
            proportions = {k: v / type_sum for k, v in temp.items() if k != "7.0" and k != "1.0" }
            entropy = -sum(p * np.log(p) for p in proportions.values())
        else:
            entropy = 0
        result_dict={'idx':row['idx'],'build_availability':build_availability,'entropy':entropy}
        results.append(result_dict)
    
    result_df=pd.DataFrame(results)
    build_availability_mean = result_df['build_availability'].mean()
    entropy_mean = result_df['entropy'].mean()
    entropy_mean_array = np.full(result_df.shape[0],entropy_mean)
    build_availability_mean_array = np.full(result_df.shape[0],build_availability_mean)
    entropy_mean_array[1:]=None
    build_availability_mean_array[1:]=None
    result_df['mean_entropy'] = entropy_mean_array
    result_df['mean_build_availability'] = build_availability_mean_array
    # 打印均值
    result_df.to_csv(output,index=False)
    return build_availability_mean,entropy_mean

if __name__ == '__main__':

    root=ROOT_DIR
    city_list=[CITY_NAME]
    for city_name in tqdm(city_list):
        city_result=os.path.join(root,'availability_diversity_process/result',f'{city_name}_buildings.csv')
        output=os.path.join(root,'availability_diversity_process/result',f'{city_name}_availability_diversity.csv')
        build_availability_mean,entropy_mean=get_result(city_result,output)
        city_stats={'mean_entropy':entropy_mean,'mean_build_availability':build_availability_mean}
        print(city_stats)




    
