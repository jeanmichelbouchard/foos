"""Train the ball detector (YOLO) on the prepared dataset.

Usage: python train_ball.py [epochs] [imgsz] [starting_model] [run_name] [batch] [fraction]
Fine-tuning example: python train_ball.py 30 960 /path/to/foos-ml/runs/ball_v1_backup/weights/best.pt ball_v2 4
Output: $FOOS_ML_DATA/runs/<run_name>/weights/best.pt
"""
import sys

from ultralytics import YOLO

from common import DATA_ROOT


def main(epochs=100, imgsz=1280, model="yolo11n.pt", name="ball", batch=4, fraction=1.0):
    m = YOLO(model)
    m.train(data=str(DATA_ROOT / "ball.yaml"), epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), device=0,
            workers=0, project=str(DATA_ROOT / "runs"), name=name, exist_ok=True, fraction=float(fraction),
            fliplr=0.5, flipud=0.5, degrees=0, mosaic=1.0, patience=15, plots=True)


if __name__ == "__main__":
    args = [int(a) if a.isdigit() else a for a in sys.argv[1:]]
    main(*args)
