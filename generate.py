import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# 1. 加载单应矩阵
with open("scripts/camera_homography.json", "r") as f:
    homographies = json.load(f)

# 将1维数组转为3x3矩阵
for k in homographies:
    H = homographies[k]["H"]
    homographies[k]["H"] = np.array(H).reshape(3, 3)

# 2. 加载YOLO模型
model = YOLO("models/yolov11/cmp_best.pt")

# 3. 定义图像路径
image_paths = [f"cam{i}.jpg" for i in range(4)]

# 4. 处理每张图像
all_results = []

for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    results = model(img)[0]
    H = homographies.get(str(idx), {}).get("H", None)

    if H is None:
        continue

    for box in results.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        ground_cx = (x1 + x2) / 2
        ground_cy = y2  # 底边的 y 坐标

        pt = np.array([ground_cx, ground_cy, 1.0])
        proj = H @ pt
        proj /= proj[2]

        all_results.append([cls, proj[0], proj[1], conf])


# 5. 转为 numpy array
final_array = np.array(all_results)

print(final_array)

# 6. 可视化
plt.figure(figsize=(8, 8))
for cls, x, y, _ in final_array:
    plt.scatter(x, y, label=f'class {int(cls)}', alpha=0.6)
plt.title("Unified 2D Map of All Objects")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()
