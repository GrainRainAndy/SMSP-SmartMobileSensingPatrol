from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/yolov11/best.pt')
    parser.add_argument('--source', type=str, default='../datasets/example.jpg')
    parser.add_argument('--project', type=str, default='../runs/detect')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(source=args.source, save=True, imgsz=640, conf=0.25, project=args.project)

if __name__ == "__main__":
    main()
