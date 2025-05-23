import cv2
import numpy as np

def project_points(H, pixel_points):
    """
    输入像素坐标点 (N, 2)，输出世界坐标点 (N, 2)
    """
    pixel_points = np.array(pixel_points, dtype=np.float32)
    pixel_points = np.concatenate([pixel_points, np.ones((len(pixel_points), 1))], axis=1)
    world_points = (H @ pixel_points.T).T
    world_points /= world_points[:, 2:3]
    return world_points[:, :2]
