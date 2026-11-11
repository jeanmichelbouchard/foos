"""Ball annotation tool (OpenCV window).

Run:  python annotate.py

Every frame proposes numbered candidates (the best one in green). Keys:
  Enter / Space / 1     accept candidate 1 (green)
  2 .. 5                accept the corresponding candidate
  left click            place the ball by hand (replaces any candidate)
  n                     the ball is not visible in this frame
  b                     go back to the previous frame
  q / Esc               save and quit
Every decision is saved immediately in ball_annotations.csv (you can resume later).
"""
import csv
import sys

import cv2

from common import ANNOT_CSV, DATA_ROOT, FRAMES_DIR, SAMPLES_CSV
from detect import candidates

# "hard" mode: python annotate.py hard
# frames with difficult (blurred) balls, with an interpolated position proposed as candidate 1.
HARD = len(sys.argv) > 1 and sys.argv[1] == "hard"
if HARD:
    SAMPLES_CSV = DATA_ROOT / "samples_hard.csv"
    ANNOT_CSV = DATA_ROOT / "ball_annotations_hard.csv"
    FRAMES_DIR = DATA_ROOT / "frames_hard"

WIN = "Ball annotation" + (" (blurred balls)" if HARD else "")
COLORS = [(0, 255, 0)] + [(0, 220, 255)] * 4


def load_annotations():
    ann = {}
    if ANNOT_CSV.exists():
        for r in csv.DictReader(open(ANNOT_CSV, encoding="utf-8")):
            ann[r["image"]] = r
    return ann


def save_annotations(ann):
    with open(ANNOT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "status", "x", "y"])
        w.writeheader()
        w.writerows(ann.values())


def main():
    samples = list(csv.DictReader(open(SAMPLES_CSV, encoding="utf-8")))
    if HARD:
        samples = [s for s in samples if s["kind"] == "pos"]   # the negatives need no annotation
    ann = load_annotations()
    todo = [i for i, s in enumerate(samples) if s["image"] not in ann]
    if not todo:
        print("All the frames are already annotated.")
        return
    pos = {"i": todo[0]}
    click = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.setMouseCallback(WIN, on_mouse)

    while 0 <= pos["i"] < len(samples):
        s = samples[pos["i"]]
        img = cv2.imread(str(FRAMES_DIR / s["image"]))
        cands = candidates(img)
        if HARD and s.get("hx"):
            # the interpolated position comes first: Enter accepts it; click to correct
            cands = [(float(s["hx"]), float(s["hy"]), 18.0, 1.0)] + cands[:4]
        view = img.copy()
        for k, (cx, cy, r, _) in enumerate(cands):
            cv2.circle(view, (int(cx), int(cy)), int(r) + 6, COLORS[k], 2)
            cv2.putText(view, str(k + 1), (int(cx) + int(r) + 8, int(cy) - int(r)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[k], 2)
        done = sum(1 for x in samples if x["image"] in ann)
        previous = ann.get(s["image"])
        label = "%d/%d  %s %s  ball:%s" % (done + 1, len(samples), s["rod"], s["subtype"], s["ball"])
        if previous:
            label += "  [already: %s]" % previous["status"]
        cv2.rectangle(view, (0, 0), (1280, 34), (0, 0, 0), -1)
        cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if previous and previous["status"] == "ball":
            cv2.drawMarker(view, (int(float(previous["x"])), int(float(previous["y"]))), (255, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)
        cv2.imshow(WIN, view)

        click["pt"] = None
        decision = None
        while decision is None:
            key = cv2.waitKey(30) & 0xFF
            if click["pt"] is not None:
                decision = ("ball", click["pt"][0], click["pt"][1])
            elif key in (13, 32, ord("1")) and cands:
                decision = ("ball", round(cands[0][0]), round(cands[0][1]))
            elif key in (ord("2"), ord("3"), ord("4"), ord("5")) and len(cands) > key - ord("1"):
                c = cands[key - ord("1")]
                decision = ("ball", round(c[0]), round(c[1]))
            elif key == ord("n"):
                decision = ("none", "", "")
            elif key == ord("b"):
                decision = ("back",)
            elif key in (ord("q"), 27):
                save_annotations(ann)
                cv2.destroyAllWindows()
                print("Saved: %d/%d annotated." % (len(ann), len(samples)))
                return
            elif cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                save_annotations(ann)
                return

        if decision[0] == "back":
            pos["i"] = max(0, pos["i"] - 1)
            continue
        ann[s["image"]] = {"image": s["image"], "status": decision[0], "x": decision[1], "y": decision[2]}
        save_annotations(ann)
        # next frame not annotated yet
        nxt = [i for i in range(pos["i"] + 1, len(samples)) if samples[i]["image"] not in ann]
        if not nxt:
            break
        pos["i"] = nxt[0]

    save_annotations(ann)
    cv2.destroyAllWindows()
    print("Done: %d annotations." % len(ann))


if __name__ == "__main__":
    sys.exit(main())
