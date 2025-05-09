import cv2
import os
import argparse
import threading
from utils.data_proc_utils import get_timestamp
import json


def capture_camera_thread(cam_id, save_dir, max_frames, interval, show_window=True):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"摄像头 {cam_id} 打开失败")
        return

    os.makedirs(save_dir, exist_ok=True)
    frame_count = 0
    saved_count = 0
    metadata = []

    print(f"摄像头 {cam_id} 开始采集")

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"摄像头 {cam_id} 采集失败")
            break

        frame_count += 1
        if frame_count % interval == 0:
            timestamp = get_timestamp()
            filename = f"{timestamp}_cam{cam_id}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

            # 收集元数据
            metadata.append({
                "timestamp": timestamp,
                "camera_id": cam_id,
                "filename": filename
            })

            print(f"📸 Camera {cam_id} saved: {filename}")

        if show_window:
            cv2.imshow(f"Camera {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"摄像头 {cam_id} 手动终止，共保存 {saved_count} 张图像")
            break

    # 保存 metadata.json
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    cap.release()
    if show_window:
        cv2.destroyWindow(f"Camera {cam_id}")
    print(f"摄像头 {cam_id} 采集完成，共保存 {saved_count} 张图像")

def capture_from_cameras_threaded(
    camera_ids=[0],
    max_frames=100,
    interval=1,
    save_root=None,
    show_window=True
):
    if save_root is None:
        save_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cache'))
    os.makedirs(save_root, exist_ok=True)

    threads = []
    for cam_id in camera_ids:
        cam_save_dir = os.path.join(save_root, f"camera_{cam_id}")
        t = threading.Thread(target=capture_camera_thread, args=(
            cam_id, cam_save_dir, max_frames, interval, show_window))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("所有摄像头采集线程完成")

# -------------------------------
# CLI 入口
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="多线程多摄像头采集工具")
    parser.add_argument('--camera_ids', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--no_window', action='store_true')
    args = parser.parse_args()

    capture_from_cameras_threaded(
        camera_ids=args.camera_ids,
        max_frames=args.max_frames,
        interval=args.interval,
        save_root=args.save_dir,
        show_window=not args.no_window
    )

if __name__ == '__main__':
    main()
