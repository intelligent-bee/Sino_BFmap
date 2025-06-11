# Sino_BFmap
Research Paper: Satellite Mapping of Every Building’s Function in Urban China Reveals Deep Built Environment Inequality. Nature Cities. 
Author: Zhuohong Li, Linxin Li, Ting Hu3, Mofan Cheng, Wei He, Tong Qiu, Liangpei Zhang, and Hongyan Zhang
* This code should be run in a python environment.

## Analysis
* Please run "pip install -r requirements_analysis.txt" to install the dependencies before executing any Python file.
* The analysis is divided into three main components:
| Component                                                         | Folder                          |
|-------------------------------------------------------------------|---------------------------------|
| 1. Accessibility                                                  | `#access_process`               |
| 2. Availability and Diversity                                     | `#availability_diversity_process` |
| 3. Residential Capacity Allocation and Infrastructure Occupation | `#inequal_allocation_process`   |
* To reproduce the analysis for each component, please refer to the readme file in the corresponding folder and follow the instructions.
### Accessibility
python access_process/1_OSM_process.py

python access_process/2_LC_process.py

python access_process/3_merge.py

python access_process/4_clip_buildingtype.py

python access_process/whitebox_tools/cost_distance.py

python access_process/6_access_stats.py


### Availability and Diversity
python availability_diversity_process/1_nearby_buildings_stats.py

python availability_diversity_process/2_cal_availability_diversity.py

### Residential capacity allocation and Infrastructure occupation
python inequal_allocation_process/1_raster_clip.py

python inequal_allocation_process/2_cal_attribute.py

python inequal_allocation_process/3_cal_inequality.py

Description: 
* For other experiments, please download and prepocess the datasets according to our Paper.

