"""Image sheets to examine the goals that the "camera alone" rule does not find.

For every missed clip: on the left the frame of the last detection (red circle = last position, line = path of the
last 40 detections), on the right the frame 10 frames later (is the ball visible there?).
Also writes the complete list in failures.csv.
"""
import csv

import cv2
import numpy as np

from analyze_tracks import last_seen, load
from common import DATA_ROOT
from goal_from_track import MIN_ABSENCE, X_LEFT, X_RIGHT, in_goal_zone


def classify(c):
    L, x, y = last_seen(c)
    gap = len(c["det"]) - 1 - L
    if in_goal_zone(x, y) and gap >= MIN_ABSENCE:
        return None
    if gap < MIN_ABSENCE:
        return "still_seen_at_end"
    if not (x < X_LEFT or x > X_RIGHT):
        return "middle_of_field"
    return "edge_outside_opening"


def frame_at(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else np.zeros((720, 1280, 3), np.uint8)


def tile(c, L, label):
    t = c["track"]
    det = np.where(c["det"])[0]
    recent = [i for i in det if i <= L][-40:]
    a = frame_at(c["path"], L)
    for i, j in zip(recent, recent[1:]):
        cv2.line(a, (int(t[i, 1]), int(t[i, 2])), (int(t[j, 1]), int(t[j, 2])), (0, 255, 255), 3)
    cv2.circle(a, (int(t[L, 1]), int(t[L, 2])), 30, (0, 0, 255), 3)
    b = frame_at(c["path"], min(len(c["det"]) - 1, L + 10))
    pair = cv2.hconcat([cv2.resize(a, (640, 360)), cv2.resize(b, (640, 360))])
    cv2.rectangle(pair, (0, 0), (1280, 28), (0, 0, 0), -1)
    cv2.putText(pair, "%s | %s %s | session %d | ball %s | last seen: frame %d" %
                (label, c["rod"], c["subtype"], c["session"], c["ball"], L),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return pair


def main():
    clips = [c for c in load() if c["kind"] == "short"]
    rows, groups = [], {}
    for c in clips:
        k = classify(c)
        if k:
            L, x, y = last_seen(c)
            rows.append({"clip": c["name"], "cause": k, "rod": c["rod"], "subtype": c["subtype"],
                         "session": c["session"], "ball": c["ball"], "last_frame": L, "x": round(x), "y": round(y)})
            groups.setdefault(k, []).append(c)
    with open(DATA_ROOT / "failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for k, cs in groups.items():
        # sample spread over the sessions
        cs = sorted(cs, key=lambda c: (c["session"], c["ts"]))
        pick = cs[:: max(1, len(cs) // 5)][:5]
        tiles = [tile(c, last_seen(c)[0], k) for c in pick]
        sheet = cv2.vconcat(tiles)
        out = DATA_ROOT / ("fail_%s.jpg" % k)
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(k, len(cs), "->", out)


if __name__ == "__main__":
    main()
