"""Harvest hard examples from the trajectories already computed.

1) NEGATIVES: after a goal (ball vanished into the narrow zone of a goal + margin), the ball is no longer visible
   (confirmed by the user). Any later detection is therefore a false positive (shadows, hands, figurine feet...).
   We keep a few frames per clip as "no ball" examples.
2) HARD POSITIVES: gaps of 2 to 12 frames in the trajectory before the goal (ball lost, often because of motion
   blur). The interpolated position is proposed, and the user corrects it with annotate.py hard.
"""
import csv
import random
import shutil

import cv2
import numpy as np

from analyze_tracks import load
from common import DATA_ROOT
from goal_events import goal_events

FRAMES_HARD = DATA_ROOT / "frames_hard"
SAMPLES_HARD = DATA_ROOT / "samples_hard.csv"
MARGIN = 20           # frames after the disappearance before everything is considered a false positive
N_NEG_PER_CLIP = 3
N_POS_MAX = 120
MAX_NEG = 260


def tight_event(c):
    """First goal event whose last position is really inside the goal opening."""
    for f, x, y, dur, _ in goal_events(c):
        if (x < 120 or x > 1160) and 250 <= y <= 480:
            return f
    return None


def main():
    rng = random.Random(7)
    if FRAMES_HARD.exists():
        shutil.rmtree(FRAMES_HARD)
    FRAMES_HARD.mkdir(parents=True)
    clips = [c for c in load() if c["kind"] == "short"]
    rng.shuffle(clips)

    jobs = []   # (clip, frame_index, kind, hx, hy)
    neg_total = 0
    pos_candidates = []
    for c in clips:
        t = c["track"]
        ev = tight_event(c)
        if ev is None:
            continue
        # negatives: raw detections (conf >= 0.15) after the disappearance
        post = [i for i in range(ev + MARGIN, len(t)) if t[i, 0] > 0]
        picked = []
        for i in rng.sample(post, len(post)):
            if all(abs(i - j) >= 15 for j in picked):
                picked.append(i)
            if len(picked) >= N_NEG_PER_CLIP:
                break
        for i in picked:
            if neg_total < MAX_NEG:
                jobs.append((c, i, "neg", "", ""))
                neg_total += 1
        # hard positives: gaps in the trajectory before the goal
        det = c["det"]
        idx = np.where(det[: ev + 1])[0]
        for a, b in zip(idx, idx[1:]):
            gap = b - a - 1
            if 2 <= gap <= 12:
                dist = float(np.hypot(*(t[b, 1:3] - t[a, 1:3])))
                if dist / (b - a) < 80:    # plausible speed (px/frame)
                    m = (a + b) // 2
                    f = (m - a) / (b - a)
                    hx, hy = t[a, 1] + f * (t[b, 1] - t[a, 1]), t[a, 2] + f * (t[b, 2] - t[a, 2])
                    pos_candidates.append((dist / (b - a), c, m, hx, hy))
    # one gap per clip, the fastest (hence the blurriest) first
    pos_candidates.sort(key=lambda p: -p[0])
    seen = set()
    for speed, c, m, hx, hy in pos_candidates:
        if c["name"] in seen or len([j for j in jobs if j[2] == "pos"]) >= N_POS_MAX:
            continue
        seen.add(c["name"])
        jobs.append((c, m, "pos", round(hx), round(hy)))

    by_clip = {}
    for c, i, kind, hx, hy in jobs:
        by_clip.setdefault(c["name"], (c, []))[1].append((i, kind, hx, hy))
    rows = []
    for name, (c, items) in by_clip.items():
        wanted = {i: (kind, hx, hy) for i, kind, hx, hy in items}
        cap = cv2.VideoCapture(c["path"])
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in wanted:
                kind, hx, hy = wanted[i]
                fname = "%s_%s_f%03d.jpg" % (kind, c["name"].split("_")[0], i)
                cv2.imwrite(str(FRAMES_HARD / fname), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"image": fname, "clip": name, "frame": i, "rod": c["rod"], "subtype": c["subtype"],
                             "session": c["session"], "ball": c["ball"], "kind": kind, "hx": hx, "hy": hy})
            i += 1
        cap.release()
    with open(SAMPLES_HARD, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("negatives: %d, hard positives to annotate: %d" %
          (sum(r["kind"] == "neg" for r in rows), sum(r["kind"] == "pos" for r in rows)))


if __name__ == "__main__":
    main()
