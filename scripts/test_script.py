# 用于测试其余脚本内容作为函数调用的正确性
# 运行模型下载脚本
import os

from ultralytics import YOLO

from predict_live import predict_live

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
    predict_live(
        model_path='../models/yolov11/yolo11x.pt',
        cam_id=0,
        save_video=False,
        save_dir=None,
        show_window=True,
        conf=0.25,
        fps=60
    )