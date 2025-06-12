import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from shapely.validation import make_valid
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR,CITY_NAME

def clip_tif_with_shp(input_tif, input_shp, output_tif,class_name):
    # 读取矢量文件 (shp)
    shapefile = gpd.read_file(input_shp)
    shapefile = shapefile[shapefile[class_name] == 1]
    # print(shapefile.head())
    with rasterio.open(input_tif) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs

    raster_bbox = box(*raster_bounds)
    shapefile = shapefile.to_crs(raster_crs)
    shapefile = shapefile[shapefile.intersects(raster_bbox)]

    from shapely.geometry import shape, mapping

    # 移除 Z 坐标
    def remove_z(geometry):
        # 将几何数据转换为字典（GeoJSON风格）
        geom_2d = mapping(geometry)
        # print(geom_2d)
        # 处理 'Polygon' 类型的几何对象
        if geom_2d['type'] == 'Polygon':
            # for ring in geom_2d['coordinates']:
            #     print(ring,len(ring))
            # 只保留 XY 坐标，去除 Z 坐标
            geom_2d['coordinates'] = [
                [(value[0],value[1]) for value in ring] for ring in geom_2d['coordinates']
            ]
            
        # 处理 'MultiPolygon' 类型的几何对象
        elif geom_2d['type'] == 'MultiPolygon':
            geom_2d['coordinates'] = [
                [[(value[0],value[1]) for value in ring] if len(ring)>2 else ring for ring in polygon] 
                for polygon in geom_2d['coordinates']
            ]
        # print(geom_2d)
        # 将字典形式转换回几何对象
        return shape(geom_2d)
    shapes = [remove_z(geom) for geom in shapefile.geometry if geom.is_valid and not geom.is_empty]
    
    # 读取栅格文件 (tif)
    with rasterio.open(input_tif) as src:
        out_image, out_transform = mask(src, shapes, crop=True, nodata=0)
        out_meta = src.meta.copy()
    
    # 更新元数据
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": 0
    })
    
    # 写入输出裁剪后的TIFF文件
    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)


def calculate_histogram(band_data):
    # 计算直方图，以1为进度
    counts, _ = np.histogram(band_data, bins=np.arange(0, 120000, 1000))
    return counts

def calculate_statistics(band_data, threshold=6):
    # 计算最大值、最小值、平均数、中位数和标准差
    dark_num = np.count_nonzero(band_data <= threshold)
    light_num = np.count_nonzero(band_data > threshold)
    light_ratio = light_num / (dark_num + light_num)
    data = band_data[band_data < 120000]
    data = data[data > 0]

    max_val = np.max(band_data)
    min_val = np.min(band_data)
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)

    return light_ratio, max_val, min_val, mean_val, median_val, std_val

def process_GIU_tif(input_tif, output_excel):
    # 读取TIFF文件
    with rasterio.open(input_tif) as src:
        histograms = {}
        statistics = {}
        
        # 读取R、G、B波段数据并转换为uint8类型
        r = src.read(1).astype(np.uint8)
        g = src.read(2).astype(np.uint8)
        b = src.read(3).astype(np.uint8)
        
        # 计算各个波段的直方图
        histograms['Band_R'] = calculate_histogram(r)
        histograms['Band_G'] = calculate_histogram(g)
        histograms['Band_B'] = calculate_histogram(b)
        
        # 合成灰度图
        grayscale = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        
        # 计算灰度图的直方图
        histograms['Grayscale'] = calculate_histogram(grayscale)
        
        # 计算各个波段的统计数据
        statistics['Band_R'] = calculate_statistics(r)
        statistics['Band_G'] = calculate_statistics(g)
        statistics['Band_B'] = calculate_statistics(b)
        statistics['Grayscale'] = calculate_statistics(grayscale)
    
    # 保存直方图数据和统计数据到Excel
    save_histograms_and_statistics_to_excel(histograms, statistics, output_excel)


def process_BH_tif(input_tif, output_excel):
    # 读取TIFF文件
    with rasterio.open(input_tif) as src:
        histograms = {}
        statistics = {}
        
        # 读取R、G、B波段数据并转换为uint8类型
        gray = src.read(1).astype(np.float32)
        gray = gray[gray>0]
        # 计算各个波段的直方图
        histograms['gray'] = calculate_histogram(gray)
        # 计算各个波段的统计数据
        statistics['gray'] = calculate_statistics(gray, threshold = 0)

    # 保存直方图数据和统计数据到Excel
    save_histograms_and_statistics_to_excel(histograms, statistics, output_excel, True)


def save_histograms_and_statistics_to_excel(histograms, statistics, output_excel, single_band=False):
    # 检查输出文件扩展名
    if not output_excel.endswith('.xlsx'):
        raise ValueError("Output file must have a .xlsx extension")
    
    # 创建DataFrame
    df_histograms = pd.DataFrame(histograms)
    # 添加像素值列
    df_histograms.index.name = 'Value(minute)'
    df_histograms.reset_index(inplace=True)
    df_histograms=df_histograms.rename(columns={'gray': 'count'})
    # 创建统计数据的DataFrame
    # print(list(statistics['gray'])[1:5])
    # print([item/1000 for item in list(statistics['gray'])[1:5]])
    if single_band:
        stats_data = {
            'Statistic': ['Max', 'Min', 'Mean', 'Median'],
            'Value(minute)': [item/1000 for item in list(statistics['gray'])[1:5]],
        }
    df_statistics = pd.DataFrame(stats_data)
    
    # 保存到Excel文件
    with pd.ExcelWriter(output_excel) as writer:
        df_histograms.to_excel(writer, sheet_name='Data', index=False)
        df_statistics.to_excel(writer, sheet_name='Data', startrow=len(df_histograms) + 1, index=False)


if __name__ == "__main__":

    city_name = CITY_NAME
    root=ROOT_DIR
    input_folder = os.path.join(root, 'input_data')
    temp_folder = os.path.join(root, 'temp')
    class_dict={1:'Residential',2:'Commercial',3:'Public_service',4:'Public_health',5:'Sport_and_art',6:'Educational',7:'Industrial',8:'Administration'}
    for class_name in [3,4,6]:
        class_name_en=class_dict[class_name]
        input_shp=os.path.join(input_folder,'building_map',f'{city_name}.shp')
        input_tif=os.path.join(root,'access_process','result',f'accum_{class_name_en}.tif')
        crop_tif=os.path.join(temp_folder,f'accum_{class_name_en}_crop.tif')
        # clip_tif_with_shp(input_tif, input_shp, crop_tif,class_name='class')
        output_excel = input_tif.replace('.tif', '.xlsx')
        histograms = process_BH_tif(crop_tif, output_excel)


