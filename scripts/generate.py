import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class HomographyProjector:
    def __init__(self):
        # 加载单应矩阵
        with open("camera_homography.json", "r") as f:
            self.homographies = json.load(f)
        for k in self.homographies:
            H = self.homographies[k]["H"]
            self.homographies[k]["H"] = np.array(H).reshape(3, 3)

        # 获取路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

        # 加载模型
        model_path = os.path.join(self.project_root, "models", "yolov11", "cmp_best.pt")
        self.model = YOLO(model_path)

        # 图像路径
        self.image_paths = [os.path.join(self.project_root, "datasets", "images", f"cam{i}.jpg") for i in range(4)]

        self.final_array = None

    def run(self, conf_thresh=0.0):
        all_results = []

        for idx, path in enumerate(self.image_paths):
            img = cv2.imread(path)
            results = self.model(img)[0]
            H = self.homographies.get(str(idx), {}).get("H", None)
            if H is None:
                continue

            for box in results.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())

                if conf < conf_thresh:
                    continue  # 👈 排除低置信度

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                ground_cx = (x1 + x2) / 2
                ground_cy = (y1 + y2) / 2

                pt = np.array([ground_cx, ground_cy, 1.0])
                proj = H @ pt
                proj /= proj[2]

                all_results.append([cls, proj[0], proj[1], conf])

        self.final_array = np.array(all_results)

    def show(self, conf_thresh=0.0):
        if self.final_array is None:
            raise RuntimeError("请先调用 run() 方法处理图像。")

        plt.figure(figsize=(8, 8))
        for cls, x, y, conf in self.final_array:
            if conf >= conf_thresh:
                plt.scatter(x, y, label=f'class {int(cls)} ({conf:.2f})', alpha=0.6)
        plt.title(f"Unified 2D Map (conf > {conf_thresh})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    projector = HomographyProjector()
    projector.run(conf_thresh=0.5)  # 👈 只处理置信度大于 0.5 的目标
    print(projector.final_array)
    projector.show(conf_thresh=0.5)
