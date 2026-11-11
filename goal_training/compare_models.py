"""Compare two ball models on the validation sessions (never seen in training).

Usage: python compare_models.py <tracks_folder_A> <tracks_folder_B> [sessions, e.g. 3,4]
The trajectories must already exist (track_clips.py). On the same clips we compare:
  - recall of the "vanishing into a goal" rule (goal_service/rules.py: goal_events)
  - detections after the goal (false positives, since the ball is invisible after a goal)
  - ball detection rate before the goal
"""
import sys

import numpy as np

from analyze_tracks import load
from common import DATA_ROOT
from rules import goal_events


def stats(clips):
    found = fp_frames = fp_clips = 0
    rates = []
    for c in clips:
        ev = goal_events(c)
        if ev:
            found += 1
            f = ev[0][0]
            post = c["det"][f + 20:]
            fp_frames += int(post.sum())
            fp_clips += int(post.sum() > 0)
            pre = c["det"][: f + 1]
            rates.append(pre.mean())
    n = len(clips)
    return {"clips": n, "recall": found / n, "clips_with_false_positives_after_goal": fp_clips,
            "false_positive_frames_after_goal": fp_frames, "detection_rate_before_goal": float(np.mean(rates))}


def main(dir_a, dir_b, sessions):
    a = {c["name"]: c for c in load(DATA_ROOT / dir_a) if c["kind"] == "short" and c["session"] in sessions}
    b = {c["name"]: c for c in load(DATA_ROOT / dir_b) if c["kind"] == "short" and c["session"] in sessions}
    common = sorted(set(a) & set(b))
    print("clips in common (sessions %s): %d" % (sorted(sessions), len(common)))
    for label, d in ((dir_a, a), (dir_b, b)):
        s = stats([d[n] for n in common])
        print("\n%s" % label)
        print("  recall (ball vanishes into a goal): %.1f %%" % (100 * s["recall"]))
        print("  ball detection before the goal: %.1f %% of the frames" % (100 * s["detection_rate_before_goal"]))
        print("  false positives after the goal: %d frames in %d clips" %
              (s["false_positive_frames_after_goal"], s["clips_with_false_positives_after_goal"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], {int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "3,4").split(",")})
