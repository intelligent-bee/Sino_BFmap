import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from pyproj import CRS
import os
from glob import glob
import argparse
from utils_access import *
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR,CITY_NAME



if __name__ == '__main__':
    city_name = CITY_NAME
    root = ROOT_DIR
    input_folder = os.path.join(root, 'input_data')
    output_folder = os.path.join(root, 'temp')
    gdf_path= os.path.join(input_folder, 'building_map', f'{city_name}.shp')
    gdf_city = gpd.read_file(gdf_path)
    minx, miny, maxx, maxy = gdf_city.total_bounds
    bbox = box(minx, miny, maxx, maxy)
    shp_list=['gis_osm_railways_free_1.shp','gis_osm_roads_free_1.shp','gis_osm_waterways_free_1.shp']
    with open(os.path.join(input_folder,'osm_data','Albers_China.prj'), 'r') as prj_file:
      prj_text = prj_file.read()
      crs = CRS.from_wkt(prj_text)


    #clip the shapefiles to the bounding box of the city
    for shp_name in shp_list:
        shp_path=os.path.join(input_folder,'osm_data',shp_name)
        gdf = gpd.read_file(shp_path)
        bbox_gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=gdf.crs)
        cropped_gdf = gpd.clip(gdf, bbox_gdf)
        output_shapefile_path = os.path.join(output_folder,shp_name.split('.')[0]+'_cropped.shp')
        cropped_gdf.to_file(output_shapefile_path)
        print(f"Cropped shapefile saved to {output_shapefile_path}")

        # transform the CRS , define speed and rasterize
        gdf = cropped_gdf.to_crs(crs)
        shp_type=shp_name.split('_')[2]
        if shp_type=='railways':
            gdf['speed'] = 1.5
        elif shp_type=='roads':
            gdf['speed'] = gdf['fclass'].apply(lambda x: speed_transform_road(x))
        elif shp_type=='waterways':
            gdf['speed'] = 3
        resolution = 10
        minx, miny, maxx, maxy = gdf.total_bounds
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
        transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)
        raster = rasterize(
            ((geom, value) for geom, value in zip(gdf.geometry, gdf['speed'])),
            out_shape=(height, width),
            transform=transform,
            fill=60,  # 未覆盖的区域填充值
            dtype='float32'
        )

        output_raster_path = os.path.join(output_folder,shp_name.split('_')[2]+'_raster.tif')
        with rasterio.open(
            output_raster_path,
            'w',
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs=gdf.crs,
            transform=transform,
        ) as dst:
            dst.write(raster, 1)
        print(f"Raster saved to {output_raster_path}")

    
    
