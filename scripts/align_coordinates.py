import cv2
import numpy as np
from calibration import calibrate_homography


def detect_cameras(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    camera_indices = detect_cameras()
    camera_indices.pop(0)

    # print("检测到摄像头:", camera_indices)
    # if not camera_indices:
    #     print("未检测到摄像头")
    #     return

    # 采集一帧
    # cam_frames = {}
    # for cam_id in camera_indices:
    #     cap = cv2.VideoCapture(cam_id)
    #     ret, frame = cap.read()
    #     cap.release()
    #     if ret:
    #         cam_frames[cam_id] = frame
    #     else:
    #         print(f"摄像头 {cam_id} 采集帧失败")

    # 示例：每个摄像头 ID 对应一张图片路径
    image_paths = {
        0: r'E:\images\cam0.jpg',
        1: r'E:\images\cam1.jpg',
        2: r'E:\images\cam2.jpg',
        3: r'E:\images\cam3.jpg'
    }

    cam_frames = {}
    for cam_id, path in image_paths.items():
        frame = cv2.imread(path)
        if frame is not None:
            cam_frames[cam_id] = frame
        else:
            print(f"摄像头 {cam_id} 图像读取失败，路径：{path}")





    # 标定并得到单应矩阵
    try:
        Hs = calibrate_homography.calibrate_camera(cam_frames)
    except Exception as e:
        print("标定失败:", e)
        return

    # 打开摄像头，实时显示世界坐标
    caps = {cid: cv2.VideoCapture(cid) for cid in Hs.keys()}

    print("按 q 键退出")
    while True:
        for cid, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            # 获取当前摄像头对应的单应矩阵
            H = Hs[cid]

            # ======== 显示绝对坐标系下的参考点 ========
            ref_world_pts = np.array([
                [[0, 0]],
                [[100, 0]],
                [[0, 100]],
                [[50, 500]]
            ], dtype=np.float32)  # (4, 1, 2)

            # 将世界坐标点映射到图像坐标系中（使用逆H）
            ref_img_pts = cv2.perspectiveTransform(ref_world_pts, np.linalg.inv(H))

            labels = ["O", "X", "Y", "M"]
            for i, pt in enumerate(ref_img_pts):
                x, y = int(pt[0][0]), int(pt[0][1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, labels[i], (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示图像
            cv2.imshow(f"Camera {cid}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
