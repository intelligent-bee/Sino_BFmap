import geopandas as gpd
import rasterio
from shapely.geometry import box
import os
from glob import glob
import argparse
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR,CITY_NAME

if __name__ == '__main__':

  city_name = CITY_NAME
  root=ROOT_DIR
  input_folder = os.path.join(root, 'input_data')
  output_folder = os.path.join(root, 'temp')
  gdf_path= os.path.join(input_folder, 'building_map', f'{city_name}.shp')
  class_dict={1:'Residential',2:'Commercial',3:'Public_service',4:'Public_health',5:'Sport_and_art',6:'Educational',7:'Industrial',8:'Administration'}
  for class_name in [3,4,6]:
    class_name_en=class_dict[class_name]
    shapefile_path = os.path.join(output_folder,f'{class_name_en}.shp')
    shapefile = gpd.read_file(gdf_path)
    shapefile = shapefile[shapefile['class'] == class_name]
    shapefile = shapefile[['class', 'geometry']]
    shapefile.to_file(shapefile_path)