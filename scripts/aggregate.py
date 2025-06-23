import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from generate import HomographyProjector

class PointMerger:
    def __init__(self, distance_dict=None, default_thresh=10):
        self.distance_dict = distance_dict or {}
        self.default_thresh = default_thresh
        self.merged_array = None  # 保存 merge 的结果以供可视化使用

    def merge(self, arr):
        merged_results = []
        classes = np.unique(arr[:, 0].astype(int))

        for cls in classes:
            sub_arr = arr[arr[:, 0] == cls]
            if len(sub_arr) == 0:
                continue

            coords = sub_arr[:, 1:3]
            confs = sub_arr[:, 3]
            thresh = self.distance_dict.get(int(cls), self.default_thresh)

            clustering = DBSCAN(eps=thresh, min_samples=1).fit(coords)
            labels = clustering.labels_
            num_clusters = labels.max() + 1

            for cluster_id in range(num_clusters + 1):
                cluster_points = sub_arr[labels == cluster_id]
                if len(cluster_points) == 0:
                    continue

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

        self.merged_array = np.array(merged_results)
        return self.merged_array

    def show(self):
        if self.merged_array is None:
            raise RuntimeError("请先调用 merge() 方法生成数据。")

        plt.figure(figsize=(8, 8))
        for cls, x, y, conf in self.merged_array:
            plt.scatter(x, y, label=f'class {int(cls)} ({conf:.2f})', alpha=0.6)
        plt.title("Merged Points After Clustering")
        plt.xlabel("X (World)")
        plt.ylabel("Y (World)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def write(self, filename="merged_points.npy"):
        """
        将合并后的结果保存为 numpy 文件。
        支持 .npy 或 .csv 文件格式。
        """
        if self.merged_array is None:
            raise RuntimeError("请先调用 merge() 方法生成数据。")

        if filename.endswith(".npy"):
            np.save(filename, self.merged_array)
        elif filename.endswith(".csv"):
            np.savetxt(filename, self.merged_array, delimiter=",", fmt="%.4f", header="class,x,y,confidence", comments='')
        else:
            raise ValueError("仅支持 .npy 或 .csv 格式的文件")
    def export_json(self, filename="merged_points.json"):
        """
        将合并后的结果保存为 JSON 格式。
        每个目标是一个 dict，字段为 class, x, y, confidence。
        """
        if self.merged_array is None:
            raise RuntimeError("请先调用 merge() 方法生成数据。")

        import json
        output = []
        for row in self.merged_array:
            obj = {
                "class": int(row[0]),
                "x": float(row[1]),
                "y": float(row[2]),
                "confidence": float(row[3])
            }
            output.append(obj)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)



# ✅ 示例用法
if __name__ == "__main__":
    projector = HomographyProjector()
    projector.run(conf_thresh=0.4)
    final_array = projector.final_array

    distance_dict = {
        0: 10,
        1: 10,
        2: 20,
        3: 10,
        4: 20
    }

    merger = PointMerger(distance_dict=distance_dict, default_thresh=10)
    merged_array = merger.merge(final_array)
    merger.write("merged_points.npy")
    merger.export_json("merged_points.json")


    print("合并前点数：", len(final_array))
    print("合并后点数：", len(merged_array))
    print("合并后坐标：\n", merged_array)

    # ✅ 可视化合并结果
    merger.show()
