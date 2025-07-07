# RockDiscontinuityExtraction
## 1. Function
Automatically extract rock mass orientation information from point cloud data.
## 2. File Description
- main.py: Entry point of the program

- ClusteringAnalysis.py: Orientation group clustering analysis

- DBSCAN.py: Rock plane segmentation

- FitPlane.py: Plane fitting and orientation extraction for structural surfaces

- utils.py: Utility functions
## 3. Command Line Usage
```bash
python main.py \
    --dataset_root (Root directory of the dataset) \
    --data_set (Dataset name) \
    --res_root (Directory to save results) \
    --log_root (Directory to save logs) \
    --data_suffix (Suffix of the input data files) \
    --file_suffix (Suffix for the output files) \
    --angle (Threshold angle for plane similarity) \
    --iter (Number of clustering iterations) \
    --min_clusters (Minimum number of points per plane) \
    --show_cluster_res {True, False} \
    --save_cluster_res {True, False} \
    --show_planes_res {True, False} \
    --save_planes_res {True, False} \
    --save_orientation_res {True, False}
```
## 4.Updates
More related source code will be updated in the future.

## 5. Example （video）
<video id="video" controls="" preload="none" poster="封面">
      <source id="mp4" src="./video/test.mp4" type="video/mp4">
</videos>