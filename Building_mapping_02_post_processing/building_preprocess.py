import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
from tqdm import tqdm
from shapely.geometry import box
from shapely.geometry import Point,Polygon

def find_files(directory,type=".shp"):
    shp_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(type):
                shp_files.append(os.path.join(root, file))
    return shp_files
def province_filter(gdf,province_name,name_type='sheng_mc'):
    return gdf[gdf[name_type] == province_name]
def county_divided(gdf):
    county_list=[]
    for index, row in gdf.iterrows():
        new_gdf=gpd.GeoDataFrame([row],columns=gdf.columns,crs=gdf.crs)
        county_list.append(new_gdf)
    return county_list
def county_filter(gdf):
    gdf=gpd.read_file('/home/lilinxin/data/shapefile/GS0650_2024/county.shp')
    gdf=gdf[gdf['gb'].astype(str).str.slice(0, 5) == '15631']
    gdf=gdf.sort_values(by='gb')
    county_list=[]
    for index, row in gdf.iterrows():
        new_gdf=gpd.GeoDataFrame([row],columns=gdf.columns,crs=gdf.crs)
        county_list.append(new_gdf)
    return county_list

def raster2shp(output="/home/lilinxin/data/buildings/Shanghai/shanghai_functionalzone.shp"):
    building_list=find_files('/home/lilinxin/data/buildings/Shanghai')
    raster = rasterio.open('/home/lilinxin/data/buildings/mosaic.tif')
    raster_bounds = raster.bounds
    raster_box = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)

    results = []
    for buildings in tqdm(building_list):
        buildings=gpd.read_file(buildings)
        if buildings.crs != raster.crs:
            buildings = buildings.to_crs(raster.crs)
        for index, building in tqdm(buildings.iterrows()):
            if raster_box.intersects(building['geometry']):
                out_image, out_transform = mask(raster, [building['geometry']], crop=True)
                if out_image.sum() > 0:  # 检查是否有非零像素
                    counts = np.bincount(out_image.flatten(), minlength=8)[1:]  # 忽略背景值
                    max_index = np.argmax(counts)
                    dominant_class = max_index + 1
                    proportion_max = counts[max_index] / counts.sum()
                    if proportion_max>0.5:
                    # 存储结果
                        results.append({
                            "roof_type": building['roof_type'],
                            "height": building['height'],
                            "class": dominant_class,
                            "geometry": building['geometry']               
                        })

    results_df = gpd.GeoDataFrame(results,crs=raster.crs)
    results_df.to_file(output)
from shapely.geometry import Polygon, MultiPolygon
def remove_z_coordinate(geometry):
    def process_polygon(polygon):
        if polygon.has_z:
            # Remove Z coordinates
            exterior_coords = [(xy[0], xy[1]) for xy in polygon.exterior.coords]
            interior_coords = [
                [(xy[0], xy[1]) for xy in interior.coords]
                for interior in polygon.interiors
            ]
            return Polygon(exterior_coords, interior_coords), True
        return polygon, False
    
    if isinstance(geometry, Polygon):
        return process_polygon(geometry)
    elif isinstance(geometry, MultiPolygon):
        new_polygons = []
        modified = False
        for poly in geometry.geoms:  # Use 'geoms' to iterate over MultiPolygon
            new_poly, was_modified = process_polygon(poly)
            new_polygons.append(new_poly)
            modified = modified or was_modified
        return MultiPolygon(new_polygons), modified
    else:
        raise TypeError("Input geometry must be a Polygon or MultiPolygon")
def raster2shp1(raster_path,footprint_path,output="/home/lilinxin/data/buildings/Shanghai/shanghai_functionalzone.shp"):
    raster = rasterio.open(raster_path)
    raster_bounds = raster.bounds
    print(raster.dtypes[0])
    raster_box = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)

    print('read finished')
    results = []
    buildings=gpd.read_file(footprint_path)
    count=0
    if buildings.crs != raster.crs:
        buildings = buildings.to_crs(raster.crs)
    for index, building in tqdm(buildings.iterrows(),mininterval=10,miniters=1000,maxinterval=100):
        if raster_box.intersects(building['geometry']):
            corrected_geometry,flag = remove_z_coordinate(building['geometry'])
            # if flag:
            #     count+=1
            out_image, out_transform = mask(raster, [corrected_geometry], crop=True)
            if out_image.sum() > 0:  # 检查是否有非零像素
                counts1 = np.bincount(out_image.flatten(), minlength=8)  # 忽略背景值
                counts=counts1[1:]
                max_index = np.argmax(counts)
                dominant_class = max_index + 1
                proportion_max = counts[max_index] / counts1.sum()
                if proportion_max>0.1:
                # 存储结果
                    results.append({
                        "class": dominant_class,
                        "geometry": corrected_geometry            
                    })
    # print(footprint_path.split('/')[-2],count)
    # return
    results_df = gpd.GeoDataFrame(results,crs=raster.crs)
    results_df.to_file(output)

def raster2shp_accurancy(output="/home/lilinxin/data/buildings/Shanghai/shanghai_functionalzone.shp"):
    raster = rasterio.open('/home/lilinxin/data/buildings/mosaic.tif')
    raster_bounds = raster.bounds
    raster_box = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)

    print('read finished')
    results = []
    buildings=gpd.read_file('/home/lilinxin/data/buildings/Shanghai/Shanghai.shp')
    if buildings.crs != raster.crs:
        buildings = buildings.to_crs(raster.crs)
    for index, building in tqdm(buildings.iterrows()):
        if raster_box.intersects(building['geometry']):
            out_image, out_transform = mask(raster, [building['geometry']], crop=True)
            if out_image.sum() > 0:  # 检查是否有非零像素
                counts1 = np.bincount(out_image.flatten(), minlength=8)  # 忽略背景值
                counts=counts1[1:]
                max_index = np.argmax(counts)
                dominant_class = max_index + 1
                proportion_max = counts[max_index] / counts1.sum()
                if proportion_max>0.5:
                # 存储结果
                    results.append({
                        "class": dominant_class,
                        "geometry": building['geometry']               
                    })

    results_df = gpd.GeoDataFrame(results,crs=raster.crs)
    results_df.to_file(output)

def province_buildings(division,province_name,building_path):
    china_crs=gpd.read_file('/home/lilinxin/data/shapefile/GS1822_2019/县（等积投影）.shp').crs
    buildings=gpd.read_file(building_path).to_crs(china_crs)
    province=province_filter(division,province_name=province_name,name_type='省').to_crs(china_crs)
    county_list=county_divided(province)
    class_dic={1:'Residential',2:'Commercial',3:'Public service',4:'Public health',5:'Sport and art',6:'Educational',7:'Industrial'}
    class_list=list(class_dic.values())
    county_dict={'NAME':[],'PAC':[]}
    for class_name in class_dic.values():
        county_dict[class_name+'_count']=[]
        county_dict[class_name+'_area']=[]
        county_dict[class_name+'_volume']=[]
    for county in county_list:
        # county_dict['NAME'].append(county.iloc[0]['qu_mc'])
        # county_dict['PAC'].append(county.iloc[0]['qu_dm'])
        county_dict['NAME'].append(county.iloc[0]['NAME'])
        county_dict['PAC'].append(county.iloc[0]['PAC'])
        within_a=gpd.sjoin(buildings, county, predicate='within')
        if within_a.empty:
            continue
        class_counts = within_a['class'].value_counts().to_dict()
        within_a['area']=within_a['geometry'].area
        within_a['volumn']=within_a['area']*within_a['height']
        within_a_byclass=within_a.groupby('class')
        within_a_byclass_area=within_a_byclass['geometry'].apply(lambda g: g.area.sum()).to_dict()
        within_a_byclass_volume=within_a_byclass['volumn'].sum().to_dict()
        for key in within_a_byclass_area.keys():
                county_dict[class_dic[key]+'_area'].append(within_a_byclass_area[key])
                county_dict[class_dic[key]+'_count'].append(class_counts[key])
                county_dict[class_dic[key]+'_volume'].append(within_a_byclass_volume[key])
        for key in class_dic.keys():
            if key not in within_a_byclass_area.keys():
                county_dict[class_dic[key]+'_area'].append(0)
                county_dict[class_dic[key]+'_count'].append(0)
                county_dict[class_dic[key]+'_volume'].append(0)

    df=pd.DataFrame(county_dict)
    if province_name=='上海市':
        province_name='Shanghai'
    df.to_csv(os.path.join('/home/lilinxin/data/buildings/csvfile',province_name+'_buildings2.csv'),index=False,encoding='utf-8-sig')

def province_buildings1(division,province_name,building_path,output):
    china_crs=gpd.read_file('/home/lilinxin/data/shapefile/GS1822_2019/县（等积投影）.shp').crs
    buildings=gpd.read_file(building_path).to_crs(china_crs)
    division=division.to_crs(china_crs)
    county_list=county_filter(division)

    class_dic={1:'Residential',2:'Commercial',3:'Public service',4:'Public health',5:'Sport and art',6:'Educational',7:'Industrial'}
    class_list=list(class_dic.values())
    county_dict={'NAME':[],'PAC':[]}
    for class_name in class_dic.values():
        county_dict[class_name+'_count']=[]
        county_dict[class_name+'_area']=[]
        # county_dict[class_name+'_volume']=[]
    for county in county_list:
        county=county.to_crs(china_crs)
        county_dict['NAME'].append(county.iloc[0]['name'])
        county_dict['PAC'].append(county.iloc[0]['gb'])
        within_a=gpd.sjoin(buildings, county, predicate='within')
        if within_a.empty:
            continue
        class_counts = within_a['class'].value_counts().to_dict()
        within_a['area']=within_a['geometry'].area
        # within_a['volumn']=within_a['area']*within_a['height']
        within_a_byclass=within_a.groupby('class')
        within_a_byclass_area=within_a_byclass['geometry'].apply(lambda g: g.area.sum()).to_dict()
        # within_a_byclass_volume=within_a_byclass['volumn'].sum().to_dict()
        for key in within_a_byclass_area.keys():
                county_dict[class_dic[key]+'_area'].append(within_a_byclass_area[key])
                county_dict[class_dic[key]+'_count'].append(class_counts[key])
                # county_dict[class_dic[key]+'_volume'].append(within_a_byclass_volume[key])
        for key in class_dic.keys():
            if key not in within_a_byclass_area.keys():
                county_dict[class_dic[key]+'_area'].append(0)
                county_dict[class_dic[key]+'_count'].append(0)
                # county_dict[class_dic[key]+'_volume'].append(0)

    df=pd.DataFrame(county_dict)
    if province_name=='上海市':
        province_name='Shanghai'
    df.to_csv(os.path.join('/home/lilinxin/data/buildings/csvfile',output),index=False,encoding='utf-8-sig')


def merge():
    gdf1=gpd.read_file('/home/lilinxin/data/buildings/Shanghai/上海市_part1.shp')
    gdf2=gpd.read_file('/home/lilinxin/data/buildings/Shanghai/上海市_part2.shp')
    gdf2 = gdf2.to_crs(gdf1.crs)
    combined_gdf = pd.concat([gdf1, gdf2], ignore_index=True)
    combined_gdf.to_file('/home/lilinxin/data/buildings/Shanghai/shanghai_gable.shp')

def area_cal(pred,class_name):
    china_crs=gpd.read_file('/home/lilinxin/data/shapefile/GS1822_2019/县（等积投影）.shp',num_workers=8).crs
    # print(china_crs)
    # print(china_crs.axis_info[0].unit_name)
    pred=gpd.read_file(pred,num_workers=8).to_crs(china_crs)
    class_counts = pred[class_name].value_counts().to_dict()
    pred['area']=pred['geometry'].area
    pred_byclass=pred.groupby(class_name)
    pred_byclass_area=pred_byclass['geometry'].apply(lambda g: g.area.sum()).to_dict()
    print(class_counts)
    print(pred_byclass_area)
if __name__ == '__main__':
    # building_shape="/home/lilinxin/data/buildings/Shanghai/shanghai_functionalzone_precise.shp"
    # raster2shp_accurancy(building_shape)


    # # division=gpd.read_file('/home/lilinxin/data/shapefile/GS1873_2022/区县.shp')
    # division=gpd.read_file('/home/lilinxin/data/shapefile/GS1822_2019/县（等积投影）.shp')
    # province_buildings(division,province_name='上海市',building_path=building_shape)

    building_shape="/home/lilinxin/data/buildings/Shanghai/Shanghai_test.shp"
    division=gpd.read_file('/home/lilinxin/data/shapefile/GS0650_2024/county.shp')
    province_name='上海市'
    province_buildings1(division,province_name=province_name,building_path=building_shape,output=province_name+'_pred.csv')
    



