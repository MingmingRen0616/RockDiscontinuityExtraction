'''
@Author: Mingming Ren 任铭铭 
@Date: 2024-11-26 18:49:12 
@Last Modified by:   Mingming Ren ren 
@Last Modified time: 2025-07-06 18:49:12 
'''
import open3d as o3d
import numpy as np

def fit_plane_ransac(cloud):
    plane_model, inliers = cloud.segment_plane(distance_threshold=2,  
                                               ransac_n=100,            
                                               num_iterations=1000)
    return inliers, plane_model

def calculate_dip(a, b, c):
    dip_rad = np.arctan(np.sqrt(a**2 + b**2) / abs(c))
    dip_deg = np.degrees(dip_rad)
    return dip_deg

def calculate_trend(a, b):
    # Trend = arctan2(b, a)
    trend_rad = np.arctan2(a, b)
    trend_deg = np.degrees(trend_rad)  
    if trend_deg < 0:
        trend_deg += 360
    return trend_deg

def fit_plane(planes, logger):
    idxs = np.unique(planes[:,-1])
    parameters = []
    orientations = []
    for i in range(len(idxs)):
        item = idxs[i]
        cond = planes[:,-1]==item
        points = planes[cond]
        points = points[:, 0:3]
        pcl_tmp = o3d.geometry.PointCloud()
        pcl_tmp.points = o3d.utility.Vector3dVector(points)
        _, plane_model = fit_plane_ransac(pcl_tmp)
        a, b, c, d = plane_model
        parameters.append([a,b,c,d,i])
        dip_angle = calculate_dip(a, b, c)
        trend_angle = calculate_trend(a, b)
        orientations.append([dip_angle, trend_angle, i])
    return parameters, orientations


