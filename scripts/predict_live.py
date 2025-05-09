import cv2
import os
import argparse
import threading
from ultralytics import YOLO
from utils.data_proc_utils import get_timestamp
from utils.predict_utils import (
    predict_single_frame,
    initialize_video_writer
)


def predict_live(model_path, cam_id=0, save_video=False, save_dir=None,
                  show_window=True, conf=0.25, fps=20):
    """
    实时摄像头推理
    :param model_path: 模型路径
    :param cam_id:
    :param save_video:
    :param save_dir:
    :param show_window:
    :param conf:
    :param fps:
    :return: None
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        temp_str = f"无法打开摄像头 {cam_id}"
        print(f"{temp_str:-^30}")
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
        temp_str = f"正在保存视频至：{video_path}"
        print(f"{temp_str:-^30}")

    temp_str = f"摄像头 {cam_id} 开始实时推理（按 q 退出）"
    print(f"{temp_str:-^30}")

    while True:
        ret, frame = cap.read()
        if not ret:
            temp_str = f"摄像头 {cam_id} 读取帧失败"
            print(f"{temp_str:-^30}")
            break

        result_frame = predict_single_frame(model, frame, conf)

        if show_window:
            cv2.imshow(f"Live YOLO Camera {cam_id}", result_frame)

        if writer:
            writer.write(result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            temp_str = f"摄像头 {cam_id} 手动退出"
            print(f"{temp_str:-^30}")
            break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()
    temp_str = f"摄像头 {cam_id} 推理结束"
    print(f"{temp_str:-^30}")


def predict_live_threads(model_path, cam_ids, save_video=False, save_dir=None,
                           show_window=True, conf=0.25, fps=20):
    """
    多线程实时推理
    """

    threads = []
    for cam_id in cam_ids:
        save_dir_temp = None
        if not save_dir is None:
            save_dir_temp = os.path.join(save_dir, f"camera_{cam_id}")
        t = threading.Thread(target=predict_live, args=(model_path, cam_id, save_video, save_dir_temp,
                                                         show_window, conf, fps))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    temp_str = f"所有摄像头推理线程完成"
    print(f"{temp_str:-^30}")

# ---------------------
# CLI 入口
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="实时摄像头 YOLO 推理")
    parser.add_argument('--model', type=str, default='../models/yolov10/yolov10x.pt', help='YOLO 模型路径')
    parser.add_argument('--cam_ids', type=int, default=[0], help='摄像头编号')
    parser.add_argument('--save_video', action='store_true', default=False, help='是否保存视频')
    parser.add_argument('--save_dir', type=str, default=None, help='视频保存目录')
    parser.add_argument('--no_window', action='store_true', default=False, help='不显示窗口')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO 置信度阈值')
    parser.add_argument('--fps', type=int, default=20, help='保存视频帧率')
    args = parser.parse_args()

    predict_live_threads(
        model_path=args.model,
        cam_ids=args.cam_ids,
        save_video=args.save_video,
        save_dir=args.save_dir,
        show_window=not args.no_window,
        conf=args.conf,
        fps=args.fps
    )

if __name__ == '__main__':
    main()