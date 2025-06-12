from PIL import Image
import numpy as np
import pandas as pd
import os
from scipy.ndimage import label, find_objects
import glob
import rasterio
from tqdm import tqdm
# import torch
import shutil
import random
from collections import Counter

def get_building_boxes(tile, label_image):
    boxes = []
    labeled_array, num_features = label(label_image)
    slices = find_objects(labeled_array)

    for sl in slices:
        y_min, x_min = sl[0].start, sl[1].start
        y_max, x_max = sl[0].stop, sl[1].stop
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes

def split_image(root='/home/wangziqiao/llx/data/buildings/Shanghai',outputroot=None,start=128,tile_size=(256, 256),ignore_list=[0],forval=True):

    city_name=root.split('/')[-1]
    height_directory=f'BH_{city_name}'
    ntl_directory=f'GIU_{city_name}'
    img_directory=f'image_{city_name}'
    label_directory=f'label_{city_name}'

    outputroot=os.path.join(outputroot,city_name)

    outputtrainroot=os.path.join(outputroot,'train_data')
    output_folder=os.path.join(outputtrainroot,'img')
    label_folder=os.path.join(outputtrainroot,'label')

    outputvalroot=os.path.join(outputroot,'val_data')
    outputval_folder=os.path.join(outputvalroot,'img')
    labelval_folder=os.path.join(outputvalroot,'label')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    if not os.path.exists(outputval_folder):
        os.makedirs(outputval_folder)
    if not os.path.exists(labelval_folder):
        os.makedirs(labelval_folder)
    ntl_path=os.path.join(root,ntl_directory)
    if not os.path.exists(ntl_path):
        ntl_path=os.path.join(root,f'NTL_{city_name}')
    height_path=os.path.join(root,height_directory)

    img_path=os.path.join(root,img_directory)

    directory = os.path.join(root,label_directory)
    search_criteria = "*.tif"
    query = os.path.join(directory, search_criteria)
    tif_files = glob.glob(query)
    save_count=0
    for fp in tqdm(tif_files,miniters=10,maxinterval=100):
        file_name=fp.split('/')[-1]
        img_name=file_name.split('_')[0]
        hr_path=os.path.join(img_path,img_name+'_image.tif')
        if not os.path.exists(hr_path):
            continue
        hr_img=Image.open(os.path.join(img_path,img_name+'_image.tif'))
        hr_img=np.array(hr_img)

        height_img_path=os.path.join(height_path,img_name+'_label.tif')
        if not os.path.exists(height_img_path):
            height_img=np.zeros((hr_img.shape[0],hr_img.shape[1]))
            # print(img_name,height_img.shape)
            height_img=Image.fromarray(height_img)
        else:
            height_img=Image.open(height_img_path)
        
        
        ntl_img_path=os.path.join(ntl_path,img_name+'_label.tif')
        if not os.path.exists(ntl_img_path):
            ntl_img=np.ones((3,hr_img.shape[0],hr_img.shape[1]))
            # print(img_name,ntl_img.shape)
        else:
            ntl_src=rasterio.open(ntl_img_path)
            ntl_img=ntl_src.read()

        label_img=Image.open(fp)
        label_img=np.array(label_img)

        img_height, img_width = label_img.shape


    # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

    # 切割图像并统计box
        for i in range(start, img_width, tile_size[0]):
            for j in range(start, img_height, tile_size[1]):
                # 计算当前块的坐标
                if i+tile_size[0]>img_width:
                    i=img_width-tile_size[0]
                if j+tile_size[1]>img_height:
                    j=img_height-tile_size[1]
                label_tile = label_img[j:j+tile_size[1], i:i+tile_size[0]]
                temp=np.unique(label_tile)
                if len(temp)<2 or (len(temp)==2 and 0 in temp and 8 in temp):
                    continue
                box_data = []
                tile=label_tile
                for cls in np.unique(tile):
                    if cls in [1,2,3,4,5,6,7,8]:
                        if cls not in ignore_list:  
                            label_image = (tile == cls).astype(int)
                            boxes = get_building_boxes(tile, label_image)
                            for box in boxes:
                                x_min, y_min, x_max, y_max=box
                                if (x_max-x_min)*(y_max-y_min)<10:
                                    continue
                                height_crop=np.array(height_img.crop(box))
                                ntl_crop=ntl_img[:,y_min:y_max,x_min:x_max]
                                ntl_mask=(ntl_crop<2)
                                ntl_crop[ntl_mask]

                                r,g,b=np.mean(ntl_crop[0], axis=(0, 1)),np.mean(ntl_crop[1], axis=(0, 1)),np.mean(ntl_crop[2], axis=(0, 1))
                                height=np.mean(height_crop, axis=(0, 1))
                                if height==0:
                                    continue
                                box_data.append([cls, x_min, y_min, x_max, y_max,r,g,b,height])
                if box_data==[]:
                    continue

                save_count+=1

                df = pd.DataFrame(box_data, columns=['class', 'x_min', 'y_min', 'x_max', 'y_max','r','g','b','height'])
                tile = hr_img[j:j+tile_size[1], i:i+tile_size[0]]
                tile_image = Image.fromarray(tile)
                if save_count%5==0:
                    if forval:
                        df.to_csv(f"{labelval_folder}/{img_name}_{i}_{j}.csv", index=False)
                        tile_path = f"{outputval_folder}/{img_name}_{i}_{j}.png"
                        tile_image.save(tile_path)                  
                else:
                    df.to_csv(f"{label_folder}/{img_name}_{i}_{j}.csv", index=False)
                    tile_path = f"{output_folder}/{img_name}_{i}_{j}.png"
                    tile_image.save(tile_path)
        ntl_src.close()
    # print(save_count)


def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False
def split_image_test(root='/home/wangziqiao/llx/data/buildings/Shanghai',outputroot=None,tile_size=(256, 256)):
    # 打开大图像

    city_name=root.split('/')[-1]
    height_directory=f'BH_{city_name}'
    ntl_directory=f'GIU_{city_name}'
    img_directory=f'image_{city_name}'
    label_directory=f'label_{city_name}'

    outputroot=os.path.join(outputroot,city_name)
    output_testroot=os.path.join(outputroot,'test_data')
    output_folder=os.path.join(output_testroot,'img')
    label_folder=os.path.join(output_testroot,'label')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    ntl_path=os.path.join(root,ntl_directory)
    height_path=os.path.join(root,height_directory)
    img_path=os.path.join(root,img_directory)

    directory=os.path.join(outputroot,'footprint')
    search_criteria = "*.tif"
    query = os.path.join(directory, search_criteria)
    tif_files = glob.glob(query)
    save_count=0
    passed_list=[]
    count=0
    for fp in tqdm(tif_files, mininterval=1,miniters=10,maxinterval=100):
        # count+=1
        # if count<196:
        #     continue
        file_name=fp.split('/')[-1]
        img_name=file_name.split('_')[0]
        hr_path=os.path.join(img_path,img_name+'_image.tif')
        if not os.path.exists(hr_path):
            passed_list.append(img_name)
            # print(img_name)
            continue
        if is_valid_image(hr_path)==False:
            passed_list.append(img_name)
            continue
        height_img_path=os.path.join(height_path,img_name+'_label.tif')
        ntl_img_path=os.path.join(ntl_path,img_name+'_label.tif')
        
        
        hr_img=Image.open(hr_path)
        hr_img=np.array(hr_img)
        if not os.path.exists(height_img_path):
            height_img=np.zeros((hr_img.shape[0],hr_img.shape[1]))
            # print(img_name,height_img.shape)
            height_img=Image.fromarray(height_img)
        else:
            height_img=Image.open(height_img_path)

        if not os.path.exists(ntl_img_path):
            ntl_img=np.ones((3,hr_img.shape[0],hr_img.shape[1]))
            # print(img_name,ntl_img.shape)
        else:
            ntl_src=rasterio.open(ntl_img_path)
            ntl_img=ntl_src.read()


        label_src=rasterio.open(fp)
        label_img=label_src.read()

        channel,img_height, img_width = label_img.shape


    # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        cls=1
    # 切割图像并统计box
        for i in range(0,img_width, tile_size[0]):
            for j in range(0,img_height, tile_size[1]):
                # 计算当前块的坐标
                if i+tile_size[0]>img_width:
                    i=img_width-tile_size[0]
                if j+tile_size[1]>img_height:
                    j=img_height-tile_size[1]
                label_tile = label_img[0,j:j+tile_size[1], i:i+tile_size[0]]
                box_data = []
                tile=label_tile
                label_image = (tile == cls).astype(int)
                boxes = get_building_boxes(tile, label_image)
                for box in boxes:
                    x_min, y_min, x_max, y_max=box
                    if (x_max-x_min)*(y_max-y_min)<5:
                        continue
                    height_crop=np.array(height_img.crop(box))
                    ntl_crop=ntl_img[:,y_min:y_max,x_min:x_max]
                    r,g,b=np.mean(ntl_crop[0], axis=(0, 1)),np.mean(ntl_crop[1], axis=(0, 1)),np.mean(ntl_crop[2], axis=(0, 1))
                    height=np.mean(height_crop, axis=(0, 1))
                    box_data.append([cls, x_min, y_min, x_max, y_max,r,g,b,height])
                
                if box_data==[]:
                    continue
                save_count+=1
                df = pd.DataFrame(box_data, columns=['class', 'x_min', 'y_min', 'x_max', 'y_max','r','g','b','height'])
                tile = hr_img[j:j+tile_size[1], i:i+tile_size[0]]
                tile_image = Image.fromarray(tile)
                df.to_csv(f"{label_folder}/{img_name}_{i}_{j}.csv", index=False)
                tile_path = f"{output_folder}/{img_name}_{i}_{j}.png"
                tile_image.save(tile_path)
        if os.path.exists(ntl_img_path):
            ntl_src.close()
        label_src.close()
    # print(save_count)
    # print(passed_list)


def shift_file():
    train_img_dir = '/home/wangziqiao/llx/data/buildings/Shanghai/train_data/img'
    train_label_dir = '/home/wangziqiao/llx/data/buildings/Shanghai/train_data/label'
    val_img_dir = '/home/wangziqiao/llx/data/buildings/Shanghai/val_data/img'
    val_label_dir = '/home/wangziqiao/llx/data/buildings/Shanghai/val_data/label'

    # 获取所有训练集图像文件名（不包括扩展名）
    train_files = [f.split('.')[0] for f in os.listdir(train_img_dir) if f.endswith('.png')]
    # val_files = [f.split('.')[0] for f in os.listdir(val_img_dir) if f.endswith('.png')]
    # print(len(train_files))
    # print(len(val_files))
    # 随机打乱文件列表
    random.shuffle(train_files)

    # 定义验证集比例（例如，20%）
    val_ratio = 0.2
    val_size = int(val_ratio * len(train_files))

    # 按比例划分验证集文件名列表
    val_files = train_files[:val_size]

    # 函数：移动文件
    def move_files(file_list, src_img_dir, src_label_dir, dest_img_dir, dest_label_dir):
        for file_name in file_list:
            img_src_path = os.path.join(src_img_dir, file_name + '.png')
            label_src_path = os.path.join(src_label_dir, file_name + '.csv')

            img_dest_path = os.path.join(dest_img_dir, file_name + '.png')
            label_dest_path = os.path.join(dest_label_dir, file_name + '.csv')

            shutil.move(img_src_path, img_dest_path)
            shutil.move(label_src_path, label_dest_path)

    # 移动验证集文件
    move_files(val_files, train_img_dir, train_label_dir, val_img_dir, val_label_dir)


def split_geo(tile_size=(256, 256),outputroot='/home/wangziqiao/llx/data/buildings/Shanghai/train_data'):
    root='/home/wangziqiao/llx/data/buildings/Shanghai'
    height_directory='height'
    ntl_directory='ntl'
    img_directory='img'

    
    height_folder=os.path.join(outputroot,'height')
    ntl_folder=os.path.join(outputroot,'ntl')

    ntl_path=os.path.join(root,ntl_directory)
    height_path=os.path.join(root,height_directory)
    img_path=os.path.join(root,img_directory)

    directory = 'img'
    search_criteria = "*.png"
    query = os.path.join(os.path.join(outputroot,directory), search_criteria)
    tif_files = glob.glob(query)

    tif_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))
    temp_name=None

    for fp in tqdm(tif_files):
        file_name=fp.split('/')[-1]
        img_name=file_name.split('.')[0]
        img_name, i, j = img_name.split('_')
        i, j = int(i), int(j)
        if temp_name is not None:
            if temp_name!=img_name:
                ntl_src.close()
                height_src.close()
                height_src=rasterio.open(os.path.join(height_path,img_name+'_label.tif'),num_workers=4)
                height_img=height_src.read()
                
                ntl_src=rasterio.open(os.path.join(ntl_path,img_name+'_label.tif'),num_workers=4)
                ntl_img=ntl_src.read()
        else:
            height_src=rasterio.open(os.path.join(height_path,img_name+'_label.tif'),num_workers=4)
            height_img=height_src.read()
            
            ntl_src=rasterio.open(os.path.join(ntl_path,img_name+'_label.tif'),num_workers=4)
            ntl_img=ntl_src.read()
        temp_name=img_name

        # print(ntl_img.shape,height_img.shape) #(3, 6000, 6000) (1, 6000, 6000)
        def minmaxnorm(array:np.ndarray):
            array=array.astype(np.float32)
            normalized_array = np.zeros_like(array)
            for i in range(array.shape[0]):
                channel = array[i]
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val - min_val == 0:
                    normalized_array[i] = channel
                else:
                    normalized_array[i] = (channel - min_val) / (max_val - min_val)
            return normalized_array*255
       
        ntl_tile = ntl_img[:,j:j+tile_size[1], i:i+tile_size[0]]
        ntl_tile=minmaxnorm(ntl_tile).astype(np.uint8)
        ntl_tile = ntl_tile.transpose(1,2,0)
        ntl_tile = Image.fromarray(ntl_tile)

        height_tile = height_img[:,j:j+tile_size[1], i:i+tile_size[0]]
        height_tile = height_tile.astype(np.uint8)
        height_tile = height_tile.squeeze(0)
        height_tile = Image.fromarray(height_tile)
        

        ntl_tile_path = f"{ntl_folder}/{img_name}_{i}_{j}.png"
        ntl_tile.save(ntl_tile_path)
        height_tile_path = f"{height_folder}/{img_name}_{i}_{j}.png"
        height_tile.save(height_tile_path)
    ntl_src.close()
    height_src.close()


def mean_std():
    root='/home/wangziqiao/llx/data/buildings/Shanghai'
    height_directory='height'
    ntl_directory='ntl'
    img_directory='img'

    outputroot='/home/wangziqiao/llx/data/buildings/Shanghai/train_data'
    height_folder=os.path.join(outputroot,'height')
    ntl_folder=os.path.join(outputroot,'ntl')

    ntl_path=os.path.join(root,ntl_directory)
    height_path=os.path.join(root,height_directory)
    img_path=os.path.join(root,img_directory)

    directory = 'height'
    search_criteria = "*.png"
    query = os.path.join(os.path.join(outputroot,directory), search_criteria)
    tif_files = glob.glob(query)
    pixel_count=0.0
    pixel_sum=0.0
    pixel_sum_squared=0.0
    for fp in tqdm(tif_files):
        file_name=fp.split('/')[-1]
        img_name=file_name.split('.')[0]
        img_name, i, j = img_name.split('_')
        img=Image.open(fp)
        img=np.array(img).astype(np.float64)
        # print(img.shape)
        # img=torch.tensor(img).unsqueeze(2)
        # print(img.shape)
        # break
        mask=(img!=0)
        img_=img[mask].flatten()
        img_sum=np.sum(img_)
        
        img_count=img_.shape[0]
        pixel_count+=img_count
        pixel_sum+=img_sum
        pixel_sum_squared+=np.sum(img_**2)
    print(pixel_sum,pixel_count,pixel_sum_squared)
    mean_rgb = pixel_sum / pixel_count
    var_rgb = (pixel_sum_squared / pixel_count) - (mean_rgb ** 2)
    std_rgb = np.sqrt(var_rgb)
    print(mean_rgb,std_rgb)

def building_num_cal(input='/home/wangziqiao/llx/data/buildings/Shanghai/val_data'):

    search_criteria = "*.csv"
    query = os.path.join(input, search_criteria)
    tif_files = glob.glob(query)

    class_count=Counter()
    for fp in tqdm(tif_files,mininterval=1,miniters=10,maxinterval=100):
        file_name=fp.split('/')[-1]
        img_name=file_name.split('.')[0]
        img_name, i, j = img_name.split('_')[:3]
        label=pd.read_csv(fp)
        class_count.update(label['class'].values)
    return sorted(class_count.items(), key=lambda x: x[0])




