import cv2

def list_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def capture_frame(cam_id):
    cap = cv2.VideoCapture(cam_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"摄像头 {cam_id} 获取图像失败")
    return frame
