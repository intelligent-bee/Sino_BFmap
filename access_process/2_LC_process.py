import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.features import geometry_window
from rasterio.windows import from_bounds
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
import os
from glob import glob
import argparse
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR

'''
Traffic route 1
Tree cover 2
Shrubland  3
Grassland  4
Cropland   5
Building   6
Bare / sparse vegetation 7
Snow and ice 8
Water 9
Wetland 10
Moss and lichen 12
'''

def speed_transform(band):
    cls_dict={1: 6, 2: 37, 3: 14, 4: 12.3, 5: 24, 6: 6, 7: 20, 8: 37, 9: 3, 10: 30, 12: 20}

    new_band=band.copy()
    for i in [1,2,3,4,5,6,7,8,9,10,12]:
        new_band[band==i]=cls_dict[i]
    return new_band

def clip_raster_by_shapefile(raster_path, shapefile_path, output_raster_path):
    """
    根据shp文件裁剪tif文件

    Args:
        raster_path (str): tif文件路径
        shapefile_path (str): shp文件路径
        output_raster_path (str): 裁剪后tif文件的保存路径
    """

    shapefile = gpd.read_file(shapefile_path)
    minx, miny, maxx, maxy = shapefile.total_bounds
    geom = box(minx, miny, maxx, maxy)
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [geom], crop=True)
        out_meta = src.meta.copy()
        
    out_image = speed_transform(out_image)
    out_meta.update({
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })
    with rasterio.open(output_raster_path, 'w', **out_meta) as dest:
        dest.write(out_image)
    print(f"Clipped raster saved to {output_raster_path}")
if __name__ == '__main__':
    city_name = 'Jiaxing'
    root=ROOT_DIR
    input_folder = os.path.join(root, 'input_data')
    output_folder = os.path.join(root, 'temp')
    gdf_path= os.path.join(input_folder, 'building_map', f'{city_name}.shp')
    LC_tif=os.path.join(input_folder,'osm_data/Jiaxing_l2hhz80w2.tif')
    output=os.path.join(output_folder,f'lc_costmap.tif')
    clip_raster_by_shapefile(LC_tif,gdf_path,output)

    #reproject tif file
    tif_file_path = output
    output_tif_path = os.path.join(output_folder,'lc_costmap_albers.tif')
    with open(os.path.join(input_folder,'osm_data','Albers_China.prj'), 'r') as prj_file:
        prj_text = prj_file.read()
    target_crs = CRS.from_wkt(prj_text)
    with rasterio.open(tif_file_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

    print(f"Reprojected TIFF file saved to {output_tif_path}")
