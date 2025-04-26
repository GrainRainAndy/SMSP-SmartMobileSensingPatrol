import os
import cv2
from ultralytics import YOLO

from utils.predict_utils import (
    get_all_frames_from_directory,
    predict_single_frame,
    save_frame_as_image,
    initialize_video_writer
)

def process_camera_directory(cam_dir, model, save_dir=None, save_video=False,
                              show_window=True, conf=0.25, fps=20):
    """
    处理某个摄像头目录下的所有帧：推理、显示、保存图像或视频
    """
    print(f"\n正在处理目录：{cam_dir}")
    image_paths = get_all_frames_from_directory(cam_dir)
    cam_name = os.path.basename(cam_dir)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        image_save_dir = os.path.join(save_dir, cam_name)
        os.makedirs(image_save_dir, exist_ok=True)
        if save_video:
            video_path = os.path.join(save_dir, f"{cam_name}.mp4")
            example = cv2.imread(image_paths[0])
            writer = initialize_video_writer(example, video_path, fps)
        else:
            writer = None
    else:
        writer = None

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        result_frame = predict_single_frame(model, frame, conf)

        if show_window:
            cv2.imshow(f"{cam_name}", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if save_dir:
            filename = os.path.basename(img_path)
            save_path = os.path.join(image_save_dir, filename)
            save_frame_as_image(result_frame, save_path)

        if writer:
            writer.write(result_frame)

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"摄像头 {cam_name} 图像处理完成")

def predict_from_cache(model_path, cache_dir, save_dir=None, save_video=False,
                        show_window=True, conf=0.25, fps=20):
    """
    从缓存目录读取各摄像头图像序列并执行 YOLO 推理
    """
    model = YOLO(model_path)

    cam_dirs = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir)
                if os.path.isdir(os.path.join(cache_dir, d))]

    for cam_dir in cam_dirs:
        process_camera_directory(cam_dir, model, save_dir, save_video, show_window, conf, fps)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="从缓存图像序列中运行YOLO预测")
    parser.add_argument('--model', type=str, default='../models/yolov10/yolov10s.pt', help='YOLO模型路径')
    parser.add_argument('--cache_dir', type=str, default='../Cache', help='缓存图像目录')
    parser.add_argument('--save_dir', type=str, default='../runs/detect/video_proc', help='推理结果保存目录')
    parser.add_argument('--save_video', action='store_true', default=True, help='是否保存推理结果为视频')
    parser.add_argument('--no_window', action='store_true', default=False, help='是否不显示窗口')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO置信度阈值')
    parser.add_argument('--fps', type=int, default=20, help='视频保存帧率')

    args = parser.parse_args()


    predict_from_cache(
        model_path=args.model,
        cache_dir=args.cache_dir,
        save_dir=args.save_dir,
        save_video=args.save_video,
        show_window=not args.no_window,
        conf=args.conf,
        fps=args.fps
    )
