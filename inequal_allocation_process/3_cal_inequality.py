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
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import ROOT_DIR
city_dict={
    'Shanghai': '上海',
    'Beijing': '北京',
    'Shenzhen': '深圳',
    'Guangzhou': '广州',
    'Hongkong': '香港',
    'Dongguan': '东莞',
    'Foshan': '佛山',
    'Wuhan': '武汉',
    'Chengdu': '成都',
    'Changsha': '长沙',
    'Hangzhou': '杭州',
    'Tianjin': '天津',
    'Hefei': '合肥',
    'Nanjing': '南京',
    'Ningbo': '宁波',
    'Suzhou': '苏州',
    'Qingdao': '青岛',
    'Zhengzhou': '郑州',
    'Shenyang': '沈阳',
    'Xian': '西安',
    'Taipei': '台北',
    'Chongqing': '重庆',
    'Kunming': '昆明',
    'Dalian': '大连',
    'Jinan': '济南',
    'Wuxi': '无锡',
    'Fuzhou': '福州',
    'Wenzhou': '温州',
    'Harbin': '哈尔滨',
    'Guiyang': '贵阳',
    'Nanning': '南宁',
    'Shijiazhuang': '石家庄',
    'Changchun': '长春',
    'Nanchang': '南昌',
    'Taiyuan': '太原',
    'Lanzhou': '兰州',
    'Huizhou': '惠州',
    'Haikou': '海口',
    'Changzhou': '常州',
    'Nantong': '南通',
    'Xiamen': '厦门',
    'Zhuhai': '珠海',
    'Zhongshan': '中山',
    'Jiaxing': '嘉兴',
    'Shaoxing': '绍兴',
    'Weifang': '潍坊',
    'Anshan': '鞍山',
    'Yuncheng': '运城',
    'Zibo': '淄博',
    'Yinchuan': '银川',
    'Wulumuqi': '乌鲁木齐',
    'Yangzhou': '扬州',
    'Putian': '莆田',
    'Lijiang': '丽江',
    'Luoyang': '洛阳',
    'Jinhua': '金华',
    'Bengbu': '蚌埠',
    'Jilin': '吉林',
    'Xianyang': '咸阳',
    'Tangshan': '唐山',
    'Yichang': '宜昌',
    'Yueyang': '岳阳',
    'Zhanjiang': '湛江',
    'Guilin': '桂林',
    'Nanchong': '南充',
    'Jiujiang': '九江',
    'Zunyi': '遵义',
    'Sanya': '三亚',
    'Yantai': '烟台',
    'Beihai': '北海',
    'Weihai': '威海',
    'Linyi': '临沂',
    'Shantou': '汕头',
    'Zhenjiang': '镇江',
    'Langfang': '廊坊',
    'Taizhou': '泰州',
    'Huhehaote': '呼和浩特',
    'Huzhou': '湖州',
    'Dongying': '东营',
    'Zaozhuang': '枣庄',
    'Huangshi': '黄石',
    'Jingdezhen': '景德镇',
    'Hebi': '鹤壁',
    'Chaozhou': '潮州',
    'Wenchang': '文昌',
    'Dongfang': '东方',
    'Wanning': '万宁',
    'Qianjiang': '潜江',
    'Zigong': '自贡',
    'Neijiang': '内江',
    'Jiaozuo': '焦作',
    'Tianmen': '天门',
    'Xiantao': '仙桃',
    'Xining': '西宁',
    'Shuozhou': '朔州',
    'Panjin': '盘锦',
    'Luohe': '漯河',
    'Yunfu': '云浮',
    'Xinyu': '新余',
    'Tongchuan': '铜川',
    'Danzhou': '儋州',
    'Liaoyuan': '辽源',
    'Hengshui': '衡水',
    'Yangquan': '阳泉',
    'Panzhihua': '攀枝花',
    'Puyang': '濮阳',
    'Jinchang': '金昌',
    'Lasa': '拉萨'
}
def gini_coefficient(x, y):
    """Calculate Gini coefficient based on the Lorenz curve."""
    # Ensure the arrays are sorted

    # Calculate Gini using the trapezoidal rule
    n = len(x)
    B = np.trapz(y, x)  # Area under the Lorenz curve
    Gini = 1 - 2 * B  # Gini coefficient formula
    return Gini
if __name__ == '__main__':
    root=ROOT_DIR
    city_name='Jiaxing'
    class_name1=['Residential','Commercial','Public service','Public health','Sport and art','Educational','Industrial','Administrative']
    output_df = {'city': [], 'Gini_coefficient': []}
    division_file=gpd.read_file(os.path.join(root,'input_data/GS1822_2019/市（等积投影）.shp'))
    population_path=os.path.join(root,'input_data/PopSE_China2020_100m_rep.tif')
    pass
    with rasterio.open(population_path) as src:
        shapes=division_file.to_crs(src.crs)
        temp=shapes[shapes['市'].str.startswith(city_dict[city_name])]
        out_image, out_transform = mask(src, temp.geometry, crop=True, filled=True,nodata=src.nodata)
        out_image[out_image==src.nodata]=0
        pop=np.sum(out_image.flatten())
        output_df['Population'] = [pop]
    for class_name in class_name1:
        output_df[class_name+'_Area'] = []
        output_df[class_name+'_Area_Percapita'] = []
    output_df['city'].append(city_name)
    df=pd.read_csv(os.path.join(root,'inequal_allocation_process/result',f'{city_name}.csv'))
    for i in range(1,9):
        output_df[class_name1[i-1]+'_Area'].append(np.sum(df[df['class']==i]['Area'].values.flatten()))
        output_df[class_name1[i-1]+'_Area_Percapita'].append(np.sum(df[df['class']==i]['Area'].values.flatten())/pop)
    df=df[df['Height']>1]
    df['Volume']=df['Height']*df['Area']


    df=df[df['class']==1]
    df=df[df['Population']>0]
    df['Population']=df['Population']*df['Area']


    df['Pop_density']=df['Volume']/df['Population']
    totvol=df['Volume'].sum()
    df['Population']=df['Population']/(df['Population'].sum())
    df['Volume']=df['Volume']/(df['Volume'].sum())

    df = df.sort_values(by='Pop_density')
    df['Cumulative_Population'] = np.cumsum(df['Population'])
    df['Cumulative_Volume'] = np.cumsum(df['Volume'])

    Gini = gini_coefficient(df['Cumulative_Population'].values, df['Cumulative_Volume'].values)   
    output_df['Gini_coefficient'].append(Gini)
    output_df=pd.DataFrame(output_df)
        
    output_df.to_csv(os.path.join(root,'inequal_allocation_process/result',f'{city_name}_inequality.csv'), index=False)


