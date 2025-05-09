# 用于测试其余脚本内容作为函数调用的正确性
# 运行模型下载脚本
import os

from ultralytics import YOLO
import threading

from predict_live import predict_live
from capture_to_cache import *

def download_model(model_name, save_path='../models/'):
    """
    下载YOLO模型并保存到指定路径
    :param model_name:
    :param save_path:
    :return: model
    """
    model = YOLO(model_name)
    nameWithoutExtension = os.path.splitext(os.path.basename(model_name))[0]
    path = os.path.join(save_path, nameWithoutExtension)
    model.download(save_dir=path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时摄像头 YOLO 推理")
    parser.add_argument('--camera_ids', type=int, default=[0], help='摄像头 ID')
    parser.add_argument('--model', type=str, default='../models/yolov10/yolov10x.pt', help='YOLO模型路径')
    parser.add_argument('--save_video', action='store_true', default=False, help='是否保存视频')
    parser.add_argument('--save_dir', type=str, default=None, help='视频保存目录')
    parser.add_argument('--no_window', action='store_true', default=False, help='不显示窗口')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO置信度阈值')
    parser.add_argument('--fps', type=int, default=20, help='保存视频帧率')
    args = parser.parse_args()



