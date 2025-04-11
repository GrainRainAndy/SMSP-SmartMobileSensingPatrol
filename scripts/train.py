import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/yolov11/yolo11s.pt')
    parser.add_argument('--data', type=str, default='../configs/coco8.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='../runs/detect/train')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--proj_name', type=str, default='yolo11s_coco8_test')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data,
                epochs=args.epochs,
                imgsz=640,
                device=args.device,
                project=args.save_dir,
                name='yolo11s_coco8_test')

if __name__ == "__main__":
    main()
