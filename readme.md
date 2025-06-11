# Sino_BFmap
Research Paper: Satellite Mapping of Every Building’s Function in Urban China Reveals Deep Built Environment Inequality. Nature Cities.   
Author: Zhuohong Li, Linxin Li, Ting Hu, Mofan Cheng, Wei He, Tong Qiu, Liangpei Zhang, and Hongyan Zhang  
##Background
As the world’s most rapidly urbanizing country, China now faces mounting challenges from growing inequalities in the built environment, including disparities in access to essential infrastructure and diverse functional facilities. Yet these urban inequalities have remained unclear due to coarse observation scales and limited analytical scopes. In this study, we present the first building-level functional map of China, covering 110 million individual buildings across 109 cities using 69 terabytes of 1-meter resolution multi-modal satellite imagery. The national-scale map is validated by government reports and 5,280,695 observation points, showing strong agreement with external benchmarks. This enables the first nationwide, multi-dimensional assessment of inequality in the built environment across city tiers, geographical regions, and intra-city zones.

* This project contains the complete protocol for downloading the building-level map products and reproducing the (1) building-level mapping and (2) multi-dimensional built environment analysis process.
* This code should be run in a Python environment.
## The first building-level functional map of urban China
### The complete map product is released at: https://figshare.com/s/f3979d3199a394911337
Based on the Paraformer framework that we proposed previously, we produced the first nationwide building-level functional map of urban China, processing over 69 TB of satellite data, including 1-meter Google Earth optical imagery (https://earth.google.com), 10-meter nighttime lights (SGDSAT-1) (https://sdg.casearth.cn/en), and building height data (CNBH-10m) (https://zenodo.org/records/7827315). Labels were derived from: (1) Building footprint data, including the CN-OpenData (https://doi.org/10.11888/Geogra.tpdc.271702) and the East Asia Building Dataset (https://zenodo.org/records/8174931); and (2) Land use and AOI data used for constructing urban functional annotation are retrieved from OpenStreetMap (https://www.openstreetmap.org) and the EULUC-China dataset (https://doi.org/10.1016/j.scib.2019.12.007). The first 1-meter resolution national-scale land-cover map used to conduct the accessibility analysis is available in our previous study: SinoLC-1 (https://doi.org/10.5281/zenodo.7707461). The housing inequality and infrastructure allocation analysis was conducted based on the 100-meter gridded population dataset from China's seventh census (https://figshare.com/s/d9dd5f9bb1a7f4fd3734?file=43847643).

This version of the data includes (1) Building-level functional maps of 109 Chinese cities, and (2) In-situ validation point sets. The building-level functional maps of 109 Chinese cities are organized in the ESRI Shapefile format, which includes five components: “.cpg”, “.dbf”, “.shx”, “.shp”, and “.prj” files. These components are stored in “.zip” files. Each city is named “G_P_C.zip,” where “G” explains the geographical region (south, central, east, north, northeast, northwest, and southwest of China) information, “P” explains the provincial administrative region information, and “C” explains the city name. For example, the building functional map for Wuhan City, Hubei Province is named “Central_Hubei_Wuhan.zip”.

Furthermore, each shapefile of a city contains the building functional types from 1 to 8, where the corresponding relationship between the values and the building functions is shown below:

1.Residential building
2.Commercial building
3.Industrial building
4.Healthcare building
5.Sport and art building
6.Educational building
7.Public service building
8.Administrative building
## Multi-dimensional Built Environment Analysis
* Please run "pip install -r `requirements_analysis.txt`" to install the dependencies before executing any python file.
* Please install the demo dataset to `input_data` from:
* The analysis is divided into three main components:

| Component                                                         | Folder                          |
|-------------------------------------------------------------------|---------------------------------|
| Accessibility                                                  | `access_process`               |
| Availability and Diversity                                     | `availability_diversity_process` |
| Residential Capacity Allocation and Infrastructure Occupation | `inequal_allocation_process`   |

To reproduce each component of the analysis, please refer to the `readme.md` file in the corresponding folder and follow the instructions.

## Building functional mapping
## Description: 
* To reproduce all experimental results, please download and preprocess the complete dataset as described in our paper: https://figshare.com/s/f3979d3199a394911337.

