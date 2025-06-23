import argparse
import cv2
from ultralytics import YOLO


def predict(model, source, save_dir, imgsz = 640):
    img = cv2.imread(source)
    model = YOLO(model)
    model.predict(source=source, save=True, imgsz=imgsz, conf=0.3, project=save_dir)
    results = model(img)[0]


# -----------------------
# CLI 入口函数
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/yolov11/cmp_best.pt')
    parser.add_argument('--source', type=str, default='../datasets/normal.jpg')
    parser.add_argument('--save_dir', type=str, default='../runs/detect')
    args = parser.parse_args()

    predict(model=args.model,
            source='D:\\Github\\SMSP-SmartMobileSensingPatrol\\Cache\\cam0.jpg',
            save_dir=args.save_dir,
            imgsz=640)


if __name__ == "__main__":
    main()
