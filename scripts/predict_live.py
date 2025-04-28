import cv2
import os
import argparse
from ultralytics import YOLO
from utils.data_proc_utils import get_timestamp
from utils.predict_utils import (
    predict_single_frame,
    initialize_video_writer
)


def predict_live(model_path, cam_id=0, save_video=False, save_dir=None,
                  show_window=True, conf=0.25, fps=20):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"无法打开摄像头 {cam_id}")
        return

    writer = None
    if save_video and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ret, example = cap.read()
        if not ret:
            print(f"摄像头 {cam_id} 无法获取初始化帧")
            return
        video_path = os.path.join(save_dir, f"live_{get_timestamp()}.mp4")
        writer = initialize_video_writer(example, video_path, fps)
        print(f"正在保存视频至：{video_path}")

    print(f"摄像头 {cam_id} 开始实时推理（按 q 退出）")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"摄像头 {cam_id} 读取帧失败")
            break

        result_frame = predict_single_frame(model, frame, conf)

        if show_window:
            cv2.imshow(f"Live YOLO Camera {cam_id}", result_frame)

        if writer:
            writer.write(result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("手动退出")
            break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"推理结束，摄像头 {cam_id} 关闭")

# ---------------------
# CLI 入口
# ---------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="实时摄像头 YOLO 推理")
    parser.add_argument('--model', type=str, default='../models/yolov10/yolov10s.pt', help='YOLO模型路径')
    parser.add_argument('--cam_id', type=int, default=0, help='摄像头编号')
    parser.add_argument('--save_video', action='store_true', default=False, help='是否保存视频')
    parser.add_argument('--save_dir', type=str, default=None, help='视频保存目录')
    parser.add_argument('--no_window', action='store_true', default=False, help='不显示窗口')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO置信度阈值')
    parser.add_argument('--fps', type=int, default=20, help='保存视频帧率')
    args = parser.parse_args()

    predict_live(
        model_path=args.model,
        cam_id=args.cam_id,
        save_video=args.save_video,
        save_dir=args.save_dir,
        show_window=not args.no_window,
        conf=args.conf,
        fps=args.fps
    )