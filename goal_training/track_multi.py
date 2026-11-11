"""Like track_clips.py, but keeps the K best detections per frame (for the continuity tracking).

Usage: python track_multi.py [max_number_of_clips] [weights.pt]
Output: $FOOS_ML_DATA/tracks_top5/<clip>.npy  (N x K x 4 array: detected, x, y, confidence)
Resumes automatically: clips already processed are skipped. The tracking code itself is shared with the
service (goal_service/tracking.py).
"""
import sys
import time

import numpy as np
from ultralytics import YOLO

from common import DATA_ROOT, list_clips
from tracking import track_clip_multi


def main(limit=None, weights=None, imgsz=1280, outname="tracks_top5"):
    out_dir = DATA_ROOT / outname
    out_dir.mkdir(exist_ok=True)
    model = YOLO(str(weights or DATA_ROOT / "runs" / "ball" / "weights" / "best.pt"))
    todo = [c for c in list_clips() if not (out_dir / (c["name"] + ".npy")).exists()]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for n, c in enumerate(todo, 1):
        np.save(out_dir / (c["name"] + ".npy"), track_clip_multi(model, c["path"], imgsz=imgsz))
        el = time.time() - t0
        print("%d/%d  %s  (%.0f s elapsed, ~%.0f min left)" % (n, len(todo), c["name"], el,
              el / n * (len(todo) - n) / 60), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
