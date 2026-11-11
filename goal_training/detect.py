"""Rough pre-detection of the white ball by colour and shape.

Only used to propose candidates in the annotation tool; the user corrects them.
The ball is white, round, ~22-35 px wide on a 1280x720 frame, and surrounded by green felt.
The field lines and the chrome rods are thin and disappear with a morphological opening; the cream-coloured
borders/corners of the field are eliminated by the "green ring" test.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from common import DATA_ROOT

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
_GREEN_LO = (60, 40, 40)
_GREEN_HI = (100, 255, 255)


_BLACKLIST_FILE = DATA_ROOT / "static_marks.json"
_blacklist = None


def _static_marks():
    """Fixed marks of the field (centre spot, penalty spots...) to ignore; see build_blacklist."""
    global _blacklist
    if _blacklist is None:
        _blacklist = json.loads(_BLACKLIST_FILE.read_text()) if _BLACKLIST_FILE.exists() else []
    return _blacklist


def build_blacklist(images, min_count=8, tol=8):
    """Find the positions where a candidate reappears in many frames (fixed marks of the field)."""
    global _blacklist
    _blacklist = []
    pts = []
    for img in images:
        pts.extend((c[0], c[1]) for c in candidates(img, 50))
    pts = np.array(pts) if pts else np.zeros((0, 2))
    marks = []
    for p in pts:
        near = pts[np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) < tol]
        if len(near) >= min_count:
            m = near.mean(axis=0).tolist()
            if all(np.hypot(m[0] - q[0], m[1] - q[1]) > tol for q in marks):
                marks.append(m)
    _BLACKLIST_FILE.write_text(json.dumps(marks))
    _blacklist = marks
    return marks


def candidates(img_bgr, max_candidates=5):
    """Return a list [(x, y, radius, score)] sorted by decreasing score."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 165), (180, 80, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, _KERNEL)
    green = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(white)
    h, w = white.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 150 or area > 1800:
            continue
        aspect = min(bw, bh) / max(bw, bh)
        fill = area / (np.pi * (max(bw, bh) / 2) ** 2)
        if aspect < 0.65 or fill < 0.55:
            continue
        cx, cy = cents[i]
        r = max(bw, bh) / 2.0
        if any(np.hypot(cx - mx, cy - my) < 10 for mx, my in _static_marks()):
            continue
        # Ring around the ball: it must contain green (felt), not only cream or chrome
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r * 2.4), 255, -1)
        cv2.circle(mask, (int(cx), int(cy)), int(r * 1.3), 0, -1)
        ring = mask > 0
        green_frac = float(np.count_nonzero(green[ring])) / max(1, np.count_nonzero(ring))
        if green_frac < 0.25:
            continue
        # The marks of the field (centre spot) are smaller than the ball: small penalty
        size_pen = 0.6 if r < 11 else 1.0
        score = aspect * min(fill, 1.0) * size_pen * (0.5 + green_frac)
        out.append((float(cx), float(cy), float(r), float(score)))
    out.sort(key=lambda c: -c[3])
    return out[:max_candidates]
