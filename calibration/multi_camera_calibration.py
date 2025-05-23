import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os

class MultiCameraCalibration:
    def __init__(self, cam_frames):
        """
        cam_frames: dict {cam_id: image(np.ndarray)}
        """
        self.cam_frames = cam_frames
        self.cam_ids = list(cam_frames.keys())
        self.index = 0

        self.src_points = []
        self.dst_points = []
        self.Hs = {}  # {cam_id: H矩阵}

        self.clicked_points = []

        self.root = tk.Tk()
        self.root.title("多摄像头标定工具")

        # Canvas显示图像
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # 控件框架
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X)

        self.label_info = tk.Label(control_frame, text="")
        self.label_info.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(control_frame, text="下一摄像头", command=self.next_camera)
        self.btn_next.pack(side=tk.RIGHT, padx=5)

        self.dst_entry = tk.Entry(control_frame, width=40)
        # 默认目标坐标为单位正方形四点
        self.dst_entry.insert(0, "0 0 100 0 100 100 0 100")
        self.dst_entry.pack(side=tk.RIGHT, padx=5)

        self.img_tk = None
        self.show_image()

    def show_image(self):
        cam_id = self.cam_ids[self.index]
        frame = self.cam_frames[cam_id]
        self.current_frame = frame.copy()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        self.clicked_points = []
        self.update_label()

    def on_click(self, event):
        if len(self.clicked_points) >= 4:
            messagebox.showinfo("提示", "已选择4个点，点击‘下一摄像头’或修改目标坐标后保存")
            return
        x, y = event.x, event.y
        self.clicked_points.append([x, y])
        self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red")
        self.update_label()

    def update_label(self):
        cam_id = self.cam_ids[self.index]
        self.label_info.config(text=f"摄像头 {cam_id} 标定：已选 {len(self.clicked_points)}/4 个点")

    def save_current_calibration(self):
        if len(self.clicked_points) != 4:
            messagebox.showerror("错误", "必须选择4个点才能保存")
            return False

        try:
            dst_vals = list(map(float, self.dst_entry.get().strip().split()))
            if len(dst_vals) != 8:
                raise ValueError()
            dst_pts = [dst_vals[i:i+2] for i in range(0,8,2)]
        except:
            messagebox.showerror("错误", "目标坐标输入格式错误，应为8个数字")
            return False

        src_pts = np.array(self.clicked_points, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        H, status = cv2.findHomography(src_pts, dst_pts)
        if H is None:
            messagebox.showerror("错误", "计算单应矩阵失败")
            return False

        cam_id = self.cam_ids[self.index]
        self.Hs[cam_id] = H

        # 保存到json文件
        file_path = "camera_homography.json"
        data = {}
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except:
                    data = {}

        data[str(cam_id)] = {
            "src": self.clicked_points,
            "dst": dst_pts.tolist(),
            "H": H.flatten().tolist()
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("提示", f"摄像头 {cam_id} 标定保存成功！")
        return True

    def next_camera(self):
        # 尝试保存当前标定
        if not self.save_current_calibration():
            return
        # 切换到下一个摄像头
        self.index += 1
        if self.index >= len(self.cam_ids):
            self.root.quit()  # 所有摄像头标定完成，退出 GUI
        else:
            self.show_image()

    def run(self):
        self.root.mainloop()
        return self.Hs
