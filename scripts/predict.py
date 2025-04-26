import argparse

from ultralytics import YOLO


def predict(model, source, save_dir, imgsz = 640):
    model = YOLO(model)
    model.predict(source=source, save=True, imgsz=imgsz, conf=0.25, project=save_dir)

# -----------------------
# CLI 入口函数
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/yolov10/coco8test.pt')
    parser.add_argument('--source', type=str, default='../datasets/example.jpg')
    parser.add_argument('--save_dir', type=str, default='../runs/detect')
    args = parser.parse_args()

    predict(model=args.model,
            source=args.source,
            save_dir=args.save_dir,
            imgsz=640)


if __name__ == "__main__":
    main()
