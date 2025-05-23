import argparse

from ultralytics import YOLO


def train(model, data, epochs, device, save_dir, proj_name, imgsz = 600):
    model = YOLO(model)
    model.train(data=data,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                project=save_dir,
                name=proj_name)

# -----------------------
# CLI 入口函数
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/yolov10/yolov10s.pt')
    parser.add_argument('--data', type=str, default='../configs/data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='../runs/detect/train')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--proj_name', type=str, default='yolo10s_cmp')
    args = parser.parse_args()

    train(model=args.model,
          data=args.data,
          epochs=args.epochs,
          device=args.device,
          save_dir=args.save_dir,
          imgsz=args.imgsz,
          proj_name=args.proj_name)


if __name__ == "__main__":
    main()
