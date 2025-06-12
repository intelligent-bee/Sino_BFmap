from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import math
import random
import scipy.misc
import skimage.io
import numpy as np
import skimage.color
import torch
import pandas as pd
import glob
from PIL import Image
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from lib.pycocotools.coco import COCO
# from lib.pycocotools.cocoeval import COCOeval
# from lib.pycocotools import mask as maskUtils
import torch.utils.data as data
from config import Config
from preprocess.InputProcess import resize_image,resize_mask,extract_bboxes,minimize_mask,compose_image_meta

def minmaxnorm(array:np.ndarray):
    array=array.astype(np.float32)
    normalized_array = np.zeros_like(array)
    for i in range(array.shape[0]):
        channel = array[i]
        min_val = np.min(channel)
        max_val = np.max(channel)
        normalized_array[i] = (channel - min_val) / (max_val - min_val)
    return normalized_array

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    #@property
    #def image_ids(self):
    #    return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

class NormalizeImage(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        image = image.astype(np.float32)
        image -=self.mean
        image /= self.std
        return torch.Tensor(image.transpose(2,0,1))

class CocoDataset(data.Dataset):

    def __init__(self,dataset_dir, subset,config, class_ids=None,
                  class_map=None, return_coco=False,argument=True,use_mini_mask = False,transform = None,root='/home/wangziqiao/llx/data/buildings/Shanghai/train_data'):
        self.image_info = []
        self.image_ids=[]
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        self.config = config
        self.argument = argument
        self.use_mini_mask = use_mini_mask
        self.transform = transform
        self.height_transform=(19.273743259595932,5.833628836943932)

        self.maxbox=500
        self.root=root
        self.imgsize=256
        self.load_dataset()



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        """Load and return ground truth data for an image (image, mask, bounding boxes).
        augment: If true, apply random image augmentation. Currently, only
            horizontal flipping is offered.
        use_mini_mask: If False, returns full-size masks that are the same height
            and width as the original image. These can be big, for example
            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
            224x224 and are generated by extracting the bounding box of the
            object and resizing it to MINI_MASK_SHAPE.
        Returns:
        image: [height, width, 3]
        shape: the original shape of the image before resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE.
        """
        img_id=self.image_ids[idx]
        image = self.load_image(os.path.join(self.root,'img',img_id+'.png'))

        ntl_image=np.array(Image.open(os.path.join(self.root,'ntl',img_id+'.png'))).astype(np.float32)
        ntl_image/=255
        height_image=np.array(Image.open(os.path.join(self.root,'height',img_id+'.png'))).astype(np.float32)
        height_image=(height_image-self.height_transform[0])/self.height_transform[1]



        sample=pd.read_csv(os.path.join(self.root,'label',img_id+'.csv')).values
        # if sample.shape[1]==9:
        class_id=sample[:,0]-1
        boxes=sample[:,1:5]
        geo_info=sample[:,5:]

        boxes = torch.tensor(boxes)
        boxes = torch.div(boxes,float(self.imgsize))
        num_boxes = boxes.size(0)

        padded_boxes = torch.zeros((self.maxbox, 4))
        padded_boxes[:num_boxes, :] = boxes

        padded_class = torch.zeros((self.maxbox, 1))
        padded_class[:num_boxes, :] = torch.tensor(class_id).unsqueeze(1)

        padded_geo_info = torch.zeros((self.maxbox, 4))
        padded_geo_info[:num_boxes, :] = torch.tensor(geo_info)

        if self.transform:
            image = self.transform(image)
            ntl_image=torch.Tensor(ntl_image).transpose(0,2)
            height_image=torch.Tensor(height_image).unsqueeze(2).transpose(0,2)
        image=torch.cat((image,ntl_image,height_image),0)
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        return image,num_boxes,padded_class, padded_boxes,padded_geo_info

    def load_dataset(self):

        search_criteria = "*.png"
        directory=os.path.join(self.root,'img')
        query = os.path.join(directory, search_criteria)
        img_files = glob.glob(query)
        for fp in img_files:
            file_name=fp.split('/')[-1]
            img_name=file_name.split('.')[0]
            self.image_ids.append(img_name)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(image_id)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

def det_mask_collate(batch):
    imgs = []
    dets = []
    masks = []
    image_metas = []
    class_ids = []

    for image,image_meta,class_id,bbox,mask in batch:
        imgs.append(torch.from_numpy(np.flip(image,axis=0).copy()))
        image_metas.append(image_meta)
        class_ids.append(class_id)
        dets.append(bbox)
        masks.append(mask)
    return (torch.stack(imgs,0),image_metas,class_ids,dets,masks)





    #import cv2
    #for i in range(bbox.shape[0]):
    #    image = cv2.rectangle(image,(bbox[i][1],bbox[i][0]),(bbox[i][3],bbox[i][2]),(255,0,0))
    #cv2.imwrite('img.png',image)
    #cv2.imwrite('mask.png',mask[:,:,1]*255)


