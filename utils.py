import numpy as np
IMAGE_MEANS =np.array([127,127,127,127,127,127,127]) 
IMAGE_STDS = np.array([127,127,127,127,127,127,127])
LABEL_CLASSES = [0,1,2,3,4,5,6,7,8,20]  
LABEL_CLASS_COLORMAP = { # Color map for different building functions
    0:  (0, 0, 0),# Non-building
    1:  (255, 127, 127),#Residential building
    2: (255, 255, 0),#Commercial building
    3: (190, 255, 232),#Industrial building
    4: (255, 235, 190),#Healthcare building
    5: (205, 170, 102),#Sport and art building
    6: (0, 170, 102),#Educational building
    7: (205, 0, 102),#Public service building
    8: (255, 255, 255),#Administrative building
    20: (255, 255, 255)#Buildings without annotations
}

LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c]
    for idx, c in enumerate(LABEL_CLASSES)
}

def get_label_class_to_idx_map():
    label_to_idx_map = []
    idx = 0
    for i in range(LABEL_CLASSES[-1]+1):
        if i in LABEL_CLASSES:
            label_to_idx_map.append(idx)
            idx += 1
        else:
            label_to_idx_map.append(0)
    label_to_idx_map = np.array(label_to_idx_map).astype(np.int64)
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()