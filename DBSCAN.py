'''
@Author: Mingming Ren 任铭铭 
@Date: 2024-11-26 18:49:12 
@Last Modified by:   Mingming Ren ren 
@Last Modified time: 2025-07-06 18:49:12 
'''
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

def idbscan(clusters, min_cluster, logger):
    # clusters = np.array(clusters)
    joints = np.unique(clusters[:, -1])
    start = 0 # 起始平面索引
    total_res = []
    for item in joints:
        condition = clusters[:, -1] == item
        J_cur = clusters[condition]
        X = J_cur[:, 0:3]
        N = J_cur[:, 3:6]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ks = [6, 10, 15, 20, 25, 30]
        peak = 0
        eps_ = 0
        min_samples = 6
        for k in ks:
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors.fit(X_scaled)
            distances, _ = neighbors.kneighbors(X_scaled)
            k_distances = np.sort(distances[:, -1])
            smoothed_data = savgol_filter(k_distances, window_length=int(len(X)/20), polyorder=2)
            first_derivative = np.gradient(smoothed_data)
            second_derivative = np.gradient(first_derivative)
            peak_tmp = np.max(second_derivative)
            eps_tmp = k_distances[np.argmax(second_derivative)]
            if peak_tmp > peak:
                peak = peak_tmp
                eps_ = eps_tmp
                min_samples = k
        db = DBSCAN(eps=eps_, min_samples=min_samples, algorithm='kd_tree')
        labels = db.fit_predict(X_scaled)
        clazz =np.unique(labels)
        all_plane = []
        all_normal = []
        for i in range(len(clazz)):
            index = clazz[i]
            if index == -1:
                continue
            indexs = np.where(labels==index)
            tmp = X[indexs]
            tmp_n = N[indexs]
            if tmp.shape[0] < min_cluster:
                continue
            all_plane.append(tmp)
            all_normal.append(tmp_n)
        logger.info("The number of planes of the {}-th joint set is {}.".format(item, len(all_plane)))
        for i in range(len(all_plane)):
            seg = all_plane[i]
            nor = all_normal[i]
            for j in range(len(seg)):
                tmp = []
                tmp.append(seg[j][0])
                tmp.append(seg[j][1])
                tmp.append(seg[j][2])
                tmp.append(nor[j][0])
                tmp.append(nor[j][1])
                tmp.append(nor[j][2])
                tmp.append(item)
                tmp.append(start+i)
                total_res.append(tmp)
        start += len(all_plane)

    return total_res