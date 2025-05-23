import cv2
import numpy as np
from camera.capture import capture_frame
from utils.projector import project_points

def show_projected_points(cam_ids, H_matrices):
    while True:
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
        for cam_id in cam_ids:
            frame = capture_frame(cam_id)
            # 模拟检测框（此处应由YOLO调用替换）
            center = (frame.shape[1]//2, frame.shape[0]//2)
            points = [center]

            H = H_matrices[cam_id]
            projected = project_points(H, points)
            for x, y in projected:
                cv2.circle(canvas, (int(x * 100), int(y * 100)), 8, (0, 0, 255), -1)
                cv2.putText(canvas, f"({x:.2f}, {y:.2f})", (int(x*100)+5, int(y*100)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.imshow("绝对坐标投影图", canvas)
        if cv2.waitKey(500) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
