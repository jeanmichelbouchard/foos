"""Apply the ball detector to every frame of every clip and save the trajectories (single best detection per frame).

Usage: python track_clips.py [max_number_of_clips]
Output: $FOOS_ML_DATA/tracks/<clip>.npy  (N x 4 array: detected, x, y, confidence; one row per frame)
Resumes automatically: clips already processed are skipped.
"""
import sys
import time

import numpy as np
from ultralytics import YOLO

from common import DATA_ROOT, list_clips
from tracking import read_frames


def track_clip(model, path, conf=0.15, imgsz=1280):
    frames = read_frames(path)
    out = np.zeros((len(frames), 4), np.float32)
    # Frame by frame: a batched call returned no detections (checked on 2026-11-06)
    for k, frame in enumerate(frames):
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, device=0)[0]
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            x, y = r.boxes.xywh[i][:2].tolist()
            out[k] = (1, x, y, float(r.boxes.conf[i]))
    return out


def main(limit=None, weights=None, imgsz=1280, outname="tracks", sessions=None):
    """sessions: set of session numbers to process (None = all)."""
    out_dir = DATA_ROOT / outname
    out_dir.mkdir(exist_ok=True)
    model = YOLO(str(weights or DATA_ROOT / "runs" / "ball" / "weights" / "best.pt"))
    clips = list_clips()
    if sessions is not None:
        clips = [c for c in clips if c["session"] in sessions]
    todo = [c for c in clips if not (out_dir / (c["name"] + ".npy")).exists()]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for n, c in enumerate(todo, 1):
        np.save(out_dir / (c["name"] + ".npy"), track_clip(model, c["path"], imgsz=imgsz))
        el = time.time() - t0
        print("%d/%d  %s  (%.0f s elapsed, ~%.0f min left)" % (n, len(todo), c["name"], el,
              el / n * (len(todo) - n) / 60), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
