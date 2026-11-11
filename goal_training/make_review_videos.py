"""Create annotated videos (mp4) to examine clips by hand.

Every frame shows: frame number, green circle = detected ball (with a trail of the last 20 detections), and a red
banner "vanishing into a goal" from the frame where the rule declares the disappearance.
Slowed down (15 frames/s while the camera films at 49).
Needs ffmpeg on the PATH (or the FFMPEG environment variable).
"""
import os
import shutil
import subprocess
import sys

import cv2

from analyze_tracks import load
from common import DATA_ROOT
from fail_sheets import classify
from goal_events import goal_events

OUT = DATA_ROOT / "review"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def render(c, out_path):
    t, det = c["track"], c["det"]
    events = goal_events(c)
    ev_frame = events[0][0] if events else None
    cap = cv2.VideoCapture(c["path"])
    tmp = str(out_path) + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 15, (960, 540))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (960, 540))
        s = 960 / 1280
        recent = [k for k in range(max(0, i - 20), i + 1) if det[k]]
        for a, b in zip(recent, recent[1:]):
            cv2.line(f, (int(t[a, 1] * s), int(t[a, 2] * s)), (int(t[b, 1] * s), int(t[b, 2] * s)), (0, 255, 255), 2)
        if det[i]:
            cv2.circle(f, (int(t[i, 1] * s), int(t[i, 2] * s)), 22, (0, 255, 0), 3)
        cv2.rectangle(f, (0, 0), (960, 30), (0, 0, 0), -1)
        txt = "frame %d/200 | %s %s | %s" % (i, c["rod"], c["subtype"], "ball detected" if det[i] else "ball NOT detected")
        cv2.putText(f, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if ev_frame is not None and i >= ev_frame:
            cv2.rectangle(f, (0, 510), (960, 540), (0, 0, 200), -1)
            cv2.putText(f, "the rule declares: vanishing into a goal (frame %d)" % ev_frame, (8, 532),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        vw.write(f)
        i += 1
    vw.release()
    cap.release()
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
                   check=True)
    os.remove(tmp)


def main(n_per_cause=3):
    OUT.mkdir(parents=True, exist_ok=True)
    picks = {}
    for c in load():
        if c["kind"] != "short":
            continue
        k = classify(c)
        if k in ("edge_outside_opening", "still_seen_at_end", "middle_of_field"):
            picks.setdefault(k, []).append(c)
    for k, cs in picks.items():
        cs = sorted(cs, key=lambda c: (c["session"], c["ts"]))
        for c in cs[:: max(1, len(cs) // n_per_cause)][:n_per_cause]:
            out = OUT / ("%s_%s.mp4" % (k, c["name"].split("_")[0]))
            render(c, out)
            print(out.name, "|", c["rod"], c["subtype"], "| session", c["session"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
