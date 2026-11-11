"""Run the continuity tracker over the saved top-K detections.

The tracker (a chain of detections where the ball moves less than VMAX px/frame, with gaps interpolated; isolated
detections such as shadows, hands or figurine feet never enter a chain long enough) lives in goal_service/tracking.py,
shared with the service.

Usage: python continuity.py [min_chain_length]
Input:  $FOOS_ML_DATA/tracks_top5/<clip>.npy  (N x K x 4)
Output: $FOOS_ML_DATA/tracks_cont/<clip>.npy  (N x 4: detected, x, y, confidence), same format as tracks/
"""
import sys

import numpy as np

import tracking
from common import DATA_ROOT


def main(min_chain=None):
    if min_chain:
        tracking.MIN_CHAIN = int(min_chain)
    src = DATA_ROOT / "tracks_top5"
    dst = DATA_ROOT / "tracks_cont"
    dst.mkdir(exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.npy")):
        np.save(dst / p.name, tracking.track_continuous(np.load(p)))
        n += 1
    print("%d clips processed -> %s" % (n, dst))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
