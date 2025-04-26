import os
import cv2

from glob import glob

def get_all_frames_from_directory(directory):
    """从一个文件夹中按时间顺序获取所有图像帧路径"""
    image_paths = sorted(glob(os.path.join(directory, '*.jpg')))
    return image_paths

def predict_single_frame(model, frame, conf=0.25):
    """对单帧图像进行YOLO推理并返回渲染后的图像"""
    results = model(frame, conf=conf)
    return results[0].plot()

def save_frame_as_image(frame, save_path):
    """保存单帧图像到指定路径"""
    cv2.imwrite(save_path, frame)

def initialize_video_writer(example_frame, output_path, fps=20):
    """根据图像初始化一个视频写入器"""
    h, w = example_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))