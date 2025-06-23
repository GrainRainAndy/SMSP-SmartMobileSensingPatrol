import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os


class MultiCameraCalibration:
    def __init__(self, cam_frames):
        self.cam_frames = cam_frames
        self.cam_ids = list(cam_frames.keys())
        self.index = 0

        self.src_points = []
        self.dst_points = []
        self.Hs = {}

        self.clicked_points = []

        self.root = tk.Tk()
        self.root.title("多摄像头标定工具")

        # Canvas显示图像
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # 控件区域
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 摄像头编号输入
        tk.Label(control_frame, text="摄像头编号:").pack(side=tk.LEFT)
        self.cam_id_entry = tk.Entry(control_frame, width=5)
        self.cam_id_entry.pack(side=tk.LEFT)
        self.cam_id_entry.insert(0, str(self.cam_ids[self.index]))

        # 删除按钮
        self.btn_delete = tk.Button(control_frame, text="重新标定", command=self.delete_last_point)
        self.btn_delete.pack(side=tk.LEFT, padx=5)

        # 下一摄像头按钮
        self.btn_next = tk.Button(control_frame, text="下一摄像头", command=self.next_camera)
        self.btn_next.pack(side=tk.RIGHT, padx=5)

        # 状态标签
        self.label_info = tk.Label(control_frame, text="")
        self.label_info.pack(side=tk.LEFT, padx=5)

        # 目标坐标输入框们
        self.dst_entries = []
        self.coord_frame = tk.Frame(self.root)
        self.coord_frame.pack(fill=tk.X, padx=5, pady=5)
        for i in range(4):
            label = tk.Label(self.coord_frame, text=f"点 {i+1} 坐标:")
            label.grid(row=i, column=0)
            entry_x = tk.Entry(self.coord_frame, width=6)
            entry_y = tk.Entry(self.coord_frame, width=6)
            entry_x.grid(row=i, column=1)
            entry_y.grid(row=i, column=2)
            entry_x.insert(0, str((i % 2) * 100))
            entry_y.insert(0, str((i // 2) * 100))
            self.dst_entries.append((entry_x, entry_y))

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
            messagebox.showinfo("提示", "已选择4个点，点击‘下一摄像头’或删除后重新选择")
            return
        x, y = event.x, event.y
        self.clicked_points.append([x, y])
        self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red")
        self.canvas.create_text(x+10, y, text=str(len(self.clicked_points)), fill="red")
        self.update_label()

    def delete_last_point(self):
        if not self.clicked_points:
            return
        self.clicked_points.pop()
        self.show_image()
        for i, pt in enumerate(self.clicked_points):
            x, y = pt
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red")
            self.canvas.create_text(x+10, y, text=str(i+1), fill="red")
        self.update_label()

    def update_label(self):
        cam_id = self.cam_ids[self.index]
        self.label_info.config(text=f"摄像头 {cam_id} 标定：已选 {len(self.clicked_points)}/4 个点")

    def save_current_calibration(self):
        if len(self.clicked_points) != 4:
            messagebox.showerror("错误", "必须选择4个点才能保存")
            return False

        try:
            dst_pts = []
            for x_entry, y_entry in self.dst_entries:
                x = float(x_entry.get())
                y = float(y_entry.get())
                dst_pts.append([x, y])
        except:
            messagebox.showerror("错误", "目标坐标输入格式错误")
            return False

        src_pts = np.array(self.clicked_points, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        H, status = cv2.findHomography(src_pts, dst_pts)
        if H is None:
            messagebox.showerror("错误", "计算单应矩阵失败")
            return False

        try:
            cam_id = int(self.cam_id_entry.get())
        except:
            messagebox.showerror("错误", "摄像头编号输入不合法")
            return False

        self.Hs[cam_id] = H

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
        if not self.save_current_calibration():
            return
        self.index += 1
        if self.index >= len(self.cam_ids):
            self.root.quit()
        else:
            self.cam_id_entry.delete(0, tk.END)
            self.cam_id_entry.insert(0, str(self.cam_ids[self.index]))
            self.show_image()

    def run(self):
        self.root.mainloop()
        return self.Hs
