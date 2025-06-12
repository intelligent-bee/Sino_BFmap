import rasterio
import numpy as np
from rasterio.merge import merge
import os
import argparse
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR,CITY_NAME

# 列出要合并的 TIFF 文件路径
if __name__ == '__main__':

    root=ROOT_DIR
    work_dir=os.path.join(root, 'temp')
    tif_files = [
        'railways_raster.tif',
        'roads_raster.tif',
        'waterways_raster.tif',
        'lc_costmap_albers.tif'
    ]

    # 打开所有的 TIFF 文件
    sources = [rasterio.open(os.path.join(work_dir,fp)) for fp in tif_files]
    merged_array, merged_transform = merge(sources,method='min')


    # 定义输出文件路径
    output_tif_path = os.path.join(work_dir,'costmap_merged.tif')

    # 写入输出文件
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=merged_array.shape[1],
        width=merged_array.shape[2],
        count=1,  # 一个波段
        dtype=merged_array.dtype,
        crs=sources[0].crs,
        transform=merged_transform,
    ) as dst:
        dst.write(merged_array[0], 1)

    # 关闭所有打开的文件
    for src in sources:
        src.close()

    print(f"Merged TIFF file saved to {output_tif_path}")