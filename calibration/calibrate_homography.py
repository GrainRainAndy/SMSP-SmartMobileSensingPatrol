import cv2
import numpy as np
from tkinter import simpledialog, messagebox
from tkinter import Tk
from matplotlib import pyplot as plt
from calibration.multi_camera_calibration import MultiCameraCalibration

clicked_points = []

def onclick(event):
    if event.xdata and event.ydata:
        clicked_points.append([event.xdata, event.ydata])
        plt.plot(event.xdata, event.ydata, 'ro')
        plt.draw()

def select_enabled_cameras(camera_indices):
    if len(camera_indices) == 1:
        return camera_indices
    root = Tk()
    root.withdraw()
    selected = simpledialog.askstring("选择相机", f"输入要启用的相机编号，用逗号分隔（可用: {camera_indices}）:")
    root.destroy()
    selected_ids = list(map(int, selected.strip().split(',')))
    return selected_ids

def calibrate_camera(cam_frames):
    """
    输入: cam_frames: dict{cam_id: frame (np.ndarray)}
    调用GUI完成标定，返回字典 {cam_id: H矩阵}
    """
    calibrator = MultiCameraCalibration(cam_frames)
    Hs = calibrator.run()
    if not Hs:
        raise ValueError("未完成任何摄像头标定")
    # 关闭所有窗口
    cv2.destroyAllWindows()
    return Hs

