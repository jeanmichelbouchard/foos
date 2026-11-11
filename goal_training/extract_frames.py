"""Extract a varied sample of frames for the ball annotation.

Usage: python extract_frames.py [number_of_frames]   (default 300)

Sampling: the frames are spread over the groups (rod, sub-type, session, ball) to cover every condition;
all the frames of red/cork balls are added (up to a quota).
"""
import csv
import random
import sys
from collections import defaultdict

import cv2

from common import FRAMES_DIR, SAMPLES_CSV, list_clips


def main(total):
    rng = random.Random(42)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    clips = [c for c in list_clips() if c["kind"] == "short"]

    groups = defaultdict(list)
    for c in clips:
        groups[(c["rod"], c["subtype"], c["session"], c["ball"])].append(c)
    for g in groups.values():
        rng.shuffle(g)

    # Quota for non-white balls: all of them, up to 40
    chosen = [c for g in groups.values() for c in g if c["ball"] != "white"][:40]
    taken = {c["name"] for c in chosen}
    keys = list(groups)
    # round-robin between groups for diversity
    while len(chosen) < total and any(groups[k] for k in keys):
        for k in keys:
            while groups[k] and groups[k][-1]["name"] in taken:
                groups[k].pop()
            if groups[k] and len(chosen) < total:
                c = groups[k].pop()
                chosen.append(c)
                taken.add(c["name"])

    rows = []
    for c in chosen:
        cap = cv2.VideoCapture(c["path"])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 200
        # random frame, including near the end of the clip (ball close to the goal)
        idx = rng.randint(5, max(6, n - 5))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("cannot read:", c["name"], idx)
            continue
        fname = "%s_f%03d.jpg" % (c["name"].split("_")[0], idx)
        cv2.imwrite(str(FRAMES_DIR / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"image": fname, "clip": c["name"], "frame": idx, "rod": c["rod"],
                     "subtype": c["subtype"], "session": c["session"], "ball": c["ball"]})

    with open(SAMPLES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("%d frames written to %s" % (len(rows), FRAMES_DIR))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
