import numpy as np
from sklearn.cluster import DBSCAN

def merge_close_points_by_class_custom(arr, distance_dict=None, default_thresh=10):
    """
    合并相同类别、距离接近的点（加权平均），返回合并后的数组。
    arr: numpy array of shape [N, 4] with columns [cls, x, y, conf]
    distance_dict: dict mapping cls -> threshold
    default_thresh: default distance threshold if class not in distance_dict
    """
    merged_results = []

    classes = np.unique(arr[:, 0].astype(int))
    for cls in classes:
        sub_arr = arr[arr[:, 0] == cls]
        if len(sub_arr) == 0:
            continue

        coords = sub_arr[:, 1:3]
        confs = sub_arr[:, 3]
        thresh = distance_dict.get(int(cls), default_thresh) if distance_dict else default_thresh

        clustering = DBSCAN(eps=thresh, min_samples=1).fit(coords)
        labels = clustering.labels_
        num_clusters = labels.max() + 1

        for cluster_id in range(num_clusters + 1):
            cluster_points = sub_arr[labels == cluster_id]
            if len(cluster_points) == 0:
                continue  # ✅ 跳过空聚类

            weights = cluster_points[:, 3]
            if np.sum(weights) == 0:
                weighted_x = np.mean(cluster_points[:, 1])
                weighted_y = np.mean(cluster_points[:, 2])
                total_conf = 0.0
            else:
                weighted_x = np.average(cluster_points[:, 1], weights=weights)
                weighted_y = np.average(cluster_points[:, 2], weights=weights)
                total_conf = np.sum(weights) / len(weights)

            merged_results.append([cls, weighted_x, weighted_y, total_conf])

    return np.array(merged_results)


# 定义不同类别合并半径
distance_dict = {
    0: 12,
    1: 8,
    2: 20,
    3: 10
}
final_array= np.array([[          4   ,   51.153    ,  63.906  ,   0.62784],
 [          4   ,   11.155   ,   107.16   ,  0.56461],
 [          2    ,  8.7812    , 34.481  ,   0.52464],
 [          4   ,   135.67 ,      470.4  ,   0.87638],
 [          4    ,  80.329  ,    653.76   ,  0.83299],
 [          2     , 160.05   ,   388.72    , 0.76276],
 [          4      ,452.83    , -77.524     ,0.93143],
 [          2      ,545.17     ,-21.516     ,0.61965]])

# 使用
merged_array = merge_close_points_by_class_custom(final_array, distance_dict=distance_dict, default_thresh=10)
print(merged_array)