import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.validation import make_valid
import os 
import glob
from tqdm import tqdm
import copy
# import pyproj
import time

# os.environ['PROJ_LIB']='/home/wangziqiao/.conda/envs/geo/share/proj'
def pixel_to_coords(xmin, ymin, xmax, ymax, transform):
    minx, miny = rasterio.transform.xy(transform, ymin, xmin, offset='ul')
    maxx, maxy = rasterio.transform.xy(transform, ymax, xmax, offset='ul')
    return box(minx, miny, maxx, maxy)
def shp2tif_gdal(root='/home/wangziqiao/llx/data/buildings/Shanghai', outputroot=None, footprint='Shanghai.shp'):
    if outputroot is None:
        outputroot = root
    
    # 获取城市名并创建输出目录
    city_name = os.path.basename(root)
    outputroot = os.path.join(outputroot, city_name)
    os.makedirs(os.path.join(outputroot, 'footprint'), exist_ok=True)
    
    # 加载矢量数据
    vector_ds = ogr.Open(footprint)
    vector_layer = vector_ds.GetLayer()
    
    print('Load finished')

    # 搜索栅格数据文件
    directory = f'image_{city_name}'
    search_criteria = "*.tif"
    query = os.path.join(root, directory, search_criteria)
    tif_files = glob.glob(query)
    
    if not tif_files:
        print("No TIFF files found")
        return

    # 打开第一个栅格文件以获取 CRS 信息
    raster_ds = gdal.Open(tif_files[0])
    raster_proj = raster_ds.GetProjection()
    raster_geotrans = raster_ds.GetGeoTransform()

    # 将矢量数据转换为栅格数据的投影
    vector_srs = vector_layer.GetSpatialRef()
    raster_srs = osr.SpatialReference(wkt=raster_proj)
    if not vector_srs.IsSame(raster_srs):
        transform = osr.CoordinateTransformation(vector_srs, raster_srs)
        vector_layer = vector_ds.ExecuteSQL(
            "SELECT * FROM '{}'".format(vector_layer.GetName()), 
            dialect='SQLITE'
        )
        for feature in vector_layer:
            geom = feature.GetGeometryRef()
            geom.Transform(transform)
        vector_layer = ogr.GetDriverByName('MEMORY').CreateDataSource('').CopyLayer(vector_layer, '')
        vector_layer.GetLayer().GetSpatialRef().ImportFromWkt(raster_proj)
    
    # 处理每个 TIFF 文件
    for tif_file in tqdm(tif_files):
        img_name = os.path.basename(tif_file)
        try:
            raster_ds = gdal.Open(tif_file)
            raster_geotrans = raster_ds.GetGeoTransform()
            x_min = raster_geotrans[0]
            y_max = raster_geotrans[3]
            x_max = x_min + raster_geotrans[1] * raster_ds.RasterXSize
            y_min = y_max + raster_geotrans[5] * raster_ds.RasterYSize
            
            # 裁剪矢量数据
            vector_layer.SetSpatialFilterRect(x_min, y_min, x_max, y_max)
            
            # 创建输出栅格文件
            output_raster_path = os.path.join(outputroot, 'footprint', img_name)
            target_ds = gdal.GetDriverByName('GTiff').Create(output_raster_path, raster_ds.RasterXSize, raster_ds.RasterYSize, 1, gdal.GDT_Byte)
            target_ds.SetGeoTransform(raster_geotrans)
            target_ds.SetProjection(raster_proj)
            
            # 栅格化矢量数据
            gdal.RasterizeLayer(target_ds, [1], vector_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
            target_ds = None  # 关闭目标文件
            
        except Exception as e:
            print(f"Error processing file {tif_file}: {e}")

def shp2tif(root,outputroot,footprint):
    # 加载矢量和栅格数据
    city_name=root.split('/')[-1]
    outputroot=os.path.join(outputroot,city_name)
    vector_data = gpd.read_file(footprint)  
    # vector_data['geometry'] = vector_data['geometry'].buffer(0)
    if not os.path.exists(os.path.join(outputroot,'footprint')):
        os.makedirs(os.path.join(outputroot,'footprint'))

    print('load finished')
    
    directory = f'image_{city_name}'
    search_criteria = "*.tif"
    query = os.path.join(os.path.join(root,directory), search_criteria)
    tif_files = glob.glob(query)
    raster_data=rasterio.open(tif_files[0])
    if vector_data.crs != raster_data.crs:
            vector_data = vector_data.to_crs(raster_data.crs)
    vector_data = vector_data[vector_data['geometry'].notnull()]
    vector_data['geometry'] = vector_data['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    # vector_data=[vector_data['geometry'].notnull() & vector_data['geometry'].is_valid]
    count=0
    for tif_file in tqdm(tif_files, mininterval=1,miniters=10,maxinterval=100):
        # count+=1
        # if count<23:
        #     continue


        img_name=tif_file.split('/')[-1]
        # current=time.time()
        raster_data = rasterio.open(tif_file,num_workers=8)
        # print('open time:',time.time()-current)
        


        # 创建栅格数据的边界框
        raster_extent = box(*raster_data.bounds)

        # 将矢量数据裁剪到栅格数据的边界
        # current=time.time()
        clipped_vector = gpd.clip(vector_data, raster_extent)
        # print('clip time:',time.time()-current)
        clipped_vector = clipped_vector[clipped_vector['geometry'].notnull() & clipped_vector['geometry'].is_valid]
        if len(clipped_vector.geometry) == 0:
            # print(img_name)
            continue

        # 栅格化处理
        transform = raster_data.transform
        out_shape = (raster_data.height, raster_data.width)
        rasterized_image = rasterize(
            [(geom, 1) for geom in clipped_vector.geometry],
            out_shape=out_shape,
            fill=0,  # 背景用0填充
            transform=transform,
            dtype='uint8'
        )

        # 保存栅格化的图像
        # current=time.time()
        output_raster_path = os.path.join(outputroot,'footprint',img_name)
        with rasterio.open(
            output_raster_path,
            'w',
            driver='GTiff',
            height=out_shape[0],
            width=out_shape[1],
            count=1,
            dtype='uint8',
            crs=raster_data.crs,
            transform=transform,
        ) as dst:
            dst.write(rasterized_image, 1)
        # print('write time:',time.time()-current)
def add_box2shp(img_path,pred_box_path,footprint_path,output_path):

    search_criteria = "*.tif"
    query = os.path.join(img_path, search_criteria)
    tif_files = glob.glob(query)

    tot_pred=None
    count=0
    for tif_file in tqdm(tif_files,miniters=10,maxinterval=100):
        img_name=tif_file.split('/')[-1].split('_')[0]
        with rasterio.open(tif_file) as src:
            transform = src.transform  

        search_criteria = img_name+'_'+"**.csv"
        query = os.path.join(pred_box_path, search_criteria)
        pred_files = glob.glob(query)
        if pred_files==[]:
            continue
        temp_pred=None
        for pred_file in tqdm(pred_files,miniters=10,maxinterval=100):
            _,i,j=pred_file.split('/')[-1].split('.')[0].split('_')
            pred_data=pd.read_csv(pred_file)
            pred_data['class']+=1
            # class_data=pred_data['class'].values()
            pred_data['geometry'] = pred_data.apply(lambda row: pixel_to_coords(row['x_min']+float(i), row['y_min']+float(j), row['x_max']+float(i), row['y_max']+float(j), transform), axis=1)
            if temp_pred is None:
                temp_pred=pred_data
            else:
                temp_pred = pd.concat([temp_pred, pred_data], ignore_index=True)
            # pred_data.to_csv(os.path.join('/home/wangziqiao/llx/data/buildings/Shanghai/pred/pred_geo',f'{img_name}_{i}_{j}.csv'),index=False)
        if tot_pred is None:
            tot_pred=temp_pred
        else:
            tot_pred = pd.concat([tot_pred, temp_pred], ignore_index=True)
        

    footprint_data = gpd.read_file(footprint_path,num_workers=8)
    geo_annotations = gpd.GeoDataFrame(tot_pred, geometry='geometry', crs=src.crs)
    geo_annotations=geo_annotations.drop_duplicates(subset=['geometry'], keep='first')

    if footprint_data.crs != geo_annotations.crs:
        geo_annotations = geo_annotations.to_crs(footprint_data.crs)
    buildings_with_class = gpd.sjoin(footprint_data, geo_annotations[['geometry', 'class']], how='left', predicate='intersects')
    # outputname=footprint_name.spilit('.')[0]
    # outputname=outputname+'_pred.shp'
    buildings_with_class.to_file(output_path)



def padding_shp(origin,pred,class_name,output,display=True):
    # print('start')
    A = gpd.read_file(origin,num_workers=8)
    B = gpd.read_file(pred,num_workers=8)
    # print('read finished')
    # B=B.drop_duplicates(subset=['geometry'], keep='first')

    if display:
        pre_completion_non_null = A[class_name].notna().sum()
        completion_non_null=B[class_name].notna().sum()
        print(f'origin:{pre_completion_non_null}')
        print(f'pred:{completion_non_null}')

    if A.crs != B.crs:
        B = B.to_crs(A.crs)
    combined = gpd.overlay(A, B, how='union')
    combined=combined.drop_duplicates(subset=['geometry'], keep='first')
    combined[class_name] = combined.apply(
    lambda row: row[class_name+'_1'] if pd.notna(row[class_name+'_1']) else row[class_name+'_2'],
    axis=1
    )
    if display:
        print(combined.head())
    combined.drop(columns=[class_name+'_1', class_name+'_2'], inplace=True)

    if display:
        print(f'combined:{combined[class_name].notna().sum()}')
    if output is not None:
        combined.to_file(output)
def intersection_shp(origin,pred,class_name,output):
    A = gpd.read_file(pred)
    B = gpd.read_file(origin)

    # 确保两个Shapefile的坐标参考系统相同
    if A.crs != B.crs:
        A = A.to_crs(B.crs)

    # 使用空间关联将A的信息关联到B上
    B_with_A = gpd.sjoin(B, A, how="inner", predicate="intersects")
    matched_items = B_with_A[B_with_A[class_name+'_left'] == B_with_A[class_name+'_right']]
    if output is not None:
        matched_items.to_file(output)
        print('origin:',B[class_name].notna().sum())
        print('pred:',A[class_name].notna().sum())
        print('matched:',matched_items[class_name].notna().sum())

def area_cal(pred,class_name,crs_ref):
    pred=gpd.read_file(pred,num_workers=8).to_crs(crs_ref)
    class_counts = pred[class_name].value_counts().to_dict()
    pred['area']=pred['geometry'].area
    pred_byclass=pred.groupby(class_name)
    pred_byclass_area=pred_byclass['geometry'].apply(lambda g: g.area.sum()).to_dict()
    return sorted(class_counts.items(),key=lambda x:x[0]),sorted(pred_byclass_area.items(),key=lambda x:x[0])
if __name__ == '__main__':
    add_box2shp()





