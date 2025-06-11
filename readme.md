# Satellite Mapping of Every Building’s Function in Urban China Reveals Deep Built Environment Inequality
<!-- **Author: Zhuohong Li, Linxin Li, Ting Hu, Mofan Cheng, Wei He, Tong Qiu, Liangpei Zhang, and Hongyan Zhang** -->
<!-- ****Affiliation: Duke University, Wuhan University, Nanjing University of Information Science and Technology, and China University of Geosciences**** -->
In this study, we present the first building-level functional map of China, covering 110 million individual buildings across 109 cities using 69 terabytes of multi-modal satellite imagery. The national-scale map is validated by government reports and 5,280,695 observation points, showing strong agreement with external benchmarks. This enables the first nationwide, multi-dimensional assessment of inequality in the built environment across city tiers, geographical regions, and intra-city zones.

<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/Background.png" width="70%">
* This project contains the complete protocol for downloading the building-level map products and reproducing the (1) building-level mapping and (2) multi-dimensional built environment analysis process.
* This code should be run in a Python environment.

## The first building-level functional map of urban China
### The complete map product and user guide are released at: https://figshare.com/s/f3979d3199a394911337
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/Study_area.png" width="70%">
This version of the data includes (1) Building-level functional maps of 109 Chinese cities, and (2) In-situ validation point sets. The building-level functional maps of 109 Chinese cities are organized in the ESRI Shapefile format, which includes five components: “.cpg”, “.dbf”, “.shx”, “.shp”, and “.prj” files. These components are stored in “.zip” files. Each city is named “G_P_C.zip,” where “G” explains the geographical region (south, central, east, north, northeast, northwest, and southwest of China) information, “P” explains the provincial administrative region information, and “C” explains the city name.

The original data sources required to create the map product and analyse the multi-dimensional built environment are as follows: [**The 1-meter Google Earth optical imagery**](https://earth.google.com), [**the 10-meter nighttime lights (SGDSAT-1)**](https://sdg.casearth.cn/en), and [**the building height data (CNBH-10m)**](https://zenodo.org/records/7827315). Labels were derived from: (1) Building footprint data, including [**the CN-OpenData**](https://doi.org/10.11888/Geogra.tpdc.271702) and [**the East Asia Building Dataset**](https://zenodo.org/records/8174931); and (2) Land use and AOI data used for constructing urban functional annotation are retrieved from [**the OpenStreetMap**](https://www.openstreetmap.org) and [**the EULUC-China dataset**](https://doi.org/10.1016/j.scib.2019.12.007). The first 1-meter resolution national-scale land-cover map used to conduct the accessibility analysis is available in our previous study: [**SinoLC-1**](https://doi.org/10.5281/zenodo.7707461). The housing inequality and infrastructure allocation analysis was conducted based on the 100-meter gridded population dataset from [**China's seventh census**](https://figshare.com/s/d9dd5f9bb1a7f4fd3734?file=43847643).
## Multi-dimensional Built Environment Analysis
* Please run "pip install -r `requirements_analysis.txt`" to install the dependencies before executing any Python file.
* Please install the example dataset(Jiaxing City) to `./input_data` from:e
* The analysis is divided into three main components:

| Component                                                         | Folder                          |
|-------------------------------------------------------------------|---------------------------------|
| Accessibility                                                  | `./access_process`               |
| Availability and Diversity                                     | `./availability_diversity_process` |
| Residential Capacity Allocation and Infrastructure Occupation | `./inequal_allocation_process`   |

## Accessibility
We calculated the travel time for residents in 109 Chinese cities from their homes to the nearest Healthcare, Educational, and Public Service building.
* **Directory Structure**

    ```bash
    access_process/
    │
    ├── 1_OSM_process.py              # Process OSM network into speed raster
    ├── 2_LC_process.py               # Convert land cover types to speed values
    ├── 3_merge.py                    # Merge OSM and land cover into cost raster
    ├── 4_clip_buildingtype.py        # Clip selected building types
    ├── whitebox_tools/
    │   └── cost_distance.py          # Run cost-distance analysis using WhiteboxTools
    └── 6_access_stats.py             # Extract travel time for residential buildings

* **To conduct the accessibility analysis, follow these steps:**

1. Assign speeds to OSM road network and convert to raster
    ```bash
    python access_process/1_OSM_process.py

2. Assign speeds to land cover types  
    ```bash
    python access_process/2_LC_process.py

3. Merge land cover and road network to generate the cost raster  
    ```bash
   python access_process/3_merge.py

4. Clip relevant building types (Healthcare, Educational, Public Services)
    ```bash
    python access_process/4_clip_buildingtype.py

5. Run Cost Distance Analysis with WhiteboxTools
    ```bash
    python access_process/whitebox_tools/cost_distance.py

6. Calculate travel time from tesidential buildings to nearest target building
    ```bash
    python access_process/6_access_stats.py

* **output**


### Availability and Diversity
python availability_diversity_process/1_nearby_buildings_stats.py

python availability_diversity_process/2_cal_availability_diversity.py

### Residential capacity allocation and Infrastructure occupation
python inequal_allocation_process/1_raster_clip.py

python inequal_allocation_process/2_cal_attribute.py

python inequal_allocation_process/3_cal_inequality.py


   
* **To train and test the framework on any 109 cities contained in this study:**

* **To reproduce the analysis on any 109 cities contained in this study:**

1. Download the population map, building height map, and building function map to the `./input_data` folder
2. Run the command above by changing the city name.

## Building functional mapping
The mapping process contains segmentation and object classification parts that are shown below:
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/Building_function-mapping-l.png" width="70%">
### 01 Training Instructions of Semantic Segmentation Model (Paraformer)
We provide our original training lists for all 109 cities in the ` ./All_109_cities_trainlists/`  directory.
* **To train and conduct the mapping with the Paraformer, follow these steps:**
  
1. Download the imagenet21k ViT pre-train model at [**Pre-train ViT**](https://drive.google.com/file/d/10Ao75MEBlZYADkrXE4YLg6VObvR0b2Dr/view?usp=sharing) and put it at ` ./networks/pre-train_model/imagenet21k` 
   
2. Taking the urban building mapping of Jiaxing City as an example, download the preprocessed training dataset (approximately 80GB per city) and put it at ` ./dataset/Chesapeake_NewYork_dataset` .
   
3. Run the "Train" command to train the Paraformer at the example city:
   ```bash
   python train.py --dataset 'Building_mapping_sample_Jiaxing' --batch_size 10 --max_epochs 20 --savepath *save path of your folder* --gpu 0
4. After training, run the "Test" command to conduct the city-scale mapping:
   ```bash
   python test.py --dataset 'Building_mapping_sample_Jiaxing' --model_path *The path of trained .pth file* --save_path *To save the inferred results* --gpu 0
   
* **To train and test the framework on any 109 cities contained in this study:**

1. Edit the root storage direction of the train and test list (.csv).
2. Add your dataset_config in the "train.py" and "test.py" files.
3. Run the command above by changing the city name.

### 02 Post-processing Based on the Object Classification Model (Mask RCNN)
## Description: 
* To reproduce all experimental results, please download and preprocess the complete dataset as described in our paper: https://figshare.com/s/f3979d3199a394911337.

