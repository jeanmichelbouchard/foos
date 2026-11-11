"""Build the YOLO dataset from ball_annotations.csv.

Split by SESSION (not by frame) so the validation never uses conditions already seen in training.
The bounding box is a square of BOX px centred on the click (the ball is ~25-30 px wide).
"""
import csv
import shutil
from collections import Counter

from common import ANNOT_CSV, DATA_ROOT, FRAMES_DIR, SAMPLES_CSV

BOX = 30
W, H = 1280, 720
DATASET = DATA_ROOT / "dataset"
VAL_FRACTION = 0.2


def main():
    samples = {r["image"]: r for r in csv.DictReader(open(SAMPLES_CSV, encoding="utf-8"))}
    ann = {r["image"]: r for r in csv.DictReader(open(ANNOT_CSV, encoding="utf-8"))}
    per_session = Counter(samples[i]["session"] for i in ann)

    # Validation sessions: take sessions (without the cork ball) up to ~20 % of the frames
    cork_sessions = {samples[i]["session"] for i in ann if samples[i]["ball"] == "cork"}
    val_sessions, total = [], 0
    for sess, n in sorted(per_session.items(), key=lambda kv: kv[1]):
        if sess in cork_sessions:
            continue
        if total + n <= VAL_FRACTION * len(ann) + 5:
            val_sessions.append(sess)
            total += n
    print("validation sessions:", val_sessions, "->", total, "frames")

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    counts = Counter()
    for image, a in ann.items():
        split = "val" if samples[image]["session"] in val_sessions else "train"
        shutil.copy(FRAMES_DIR / image, DATASET / "images" / split / image)
        label = DATASET / "labels" / split / (image.rsplit(".", 1)[0] + ".txt")
        if a["status"] == "ball":
            x, y = float(a["x"]), float(a["y"])
            label.write_text("0 %.6f %.6f %.6f %.6f\n" % (x / W, y / H, BOX / W, BOX / H))
            counts[split, "ball"] += 1
        else:
            label.write_text("")  # frame without a ball: negative example
            counts[split, "none"] += 1
    # Hard examples (see harvest_hard.py): automatic negatives + blurred balls annotated by hand
    hard_samples_csv = DATA_ROOT / "samples_hard.csv"
    hard_annot_csv = DATA_ROOT / "ball_annotations_hard.csv"
    if hard_samples_csv.exists() and hard_annot_csv.exists():
        hs = list(csv.DictReader(open(hard_samples_csv, encoding="utf-8")))
        hann = {r["image"]: r for r in csv.DictReader(open(hard_annot_csv, encoding="utf-8"))}
        for r in hs:
            image = r["image"]
            split = "val" if r["session"] in val_sessions else "train"
            if r["kind"] == "pos":
                a = hann.get(image)
                if a is None:
                    continue
            else:
                a = {"status": "none"}
            shutil.copy(DATA_ROOT / "frames_hard" / image, DATA_ROOT / "dataset" / "images" / split / image)
            label = DATASET / "labels" / split / (image.rsplit(".", 1)[0] + ".txt")
            if a["status"] == "ball":
                label.write_text("0 %.6f %.6f %.6f %.6f\n" % (float(a["x"]) / W, float(a["y"]) / H, BOX / W, BOX / H))
                counts[split, "ball (blurred)"] += 1
            else:
                label.write_text("")
                counts[split, "none (hard negative)"] += 1
    print(dict(counts))

    (DATA_ROOT / "ball.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n" % DATASET.as_posix(),
        encoding="utf-8")


if __name__ == "__main__":
    main()
