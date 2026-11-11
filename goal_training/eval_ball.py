"""Evaluate the detector on the validation frames: position error and a sheet with the worst cases."""
import csv
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from common import ANNOT_CSV, DATA_ROOT, SAMPLES_CSV

W, H = 1280, 720


def main(weights=None, conf=0.25):
    weights = weights or str(DATA_ROOT / "runs" / "ball" / "weights" / "best.pt")
    model = YOLO(weights)
    val_dir = DATA_ROOT / "dataset" / "images" / "val"
    ann = {r["image"]: r for r in csv.DictReader(open(ANNOT_CSV, encoding="utf-8"))}
    samples = {r["image"]: r for r in csv.DictReader(open(SAMPLES_CSV, encoding="utf-8"))}

    results = []
    for path in sorted(val_dir.glob("*.jpg")):
        a = ann[path.name]
        r = model.predict(str(path), imgsz=1280, conf=conf, verbose=False, device=0)[0]
        boxes = r.boxes.xywh.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        best = None
        if len(boxes):
            i = int(confs.argmax())
            best = (float(boxes[i][0]), float(boxes[i][1]), float(confs[i]))
        has_ball = a["status"] == "ball"
        err = None
        if has_ball and best:
            err = float(np.hypot(best[0] - float(a["x"]), best[1] - float(a["y"])))
        results.append({"image": path.name, "has_ball": has_ball, "gt": (a["x"], a["y"]), "pred": best,
                        "err": err, "n_det": len(boxes), "ball": samples[path.name]["ball"]})

    tp = [r for r in results if r["has_ball"] and r["err"] is not None and r["err"] <= 20]
    miss = [r for r in results if r["has_ball"] and r["pred"] is None]
    wrong = [r for r in results if r["has_ball"] and r["pred"] is not None and r["err"] > 20]
    fp = [r for r in results if not r["has_ball"] and r["pred"] is not None]
    nb = sum(r["has_ball"] for r in results)
    print("validation frames: %d (with a ball: %d)" % (len(results), nb))
    print("ball found correctly (<20 px): %d/%d" % (len(tp), nb))
    print("ball missed: %d | wrong position: %d | false detection on a frame without a ball: %d" %
          (len(miss), len(wrong), len(fp)))
    errs = [r["err"] for r in tp]
    if errs:
        print("median position error: %.1f px (max %.1f)" % (np.median(errs), max(errs)))

    bad = miss + wrong + fp
    tiles = []
    for r in bad[:6]:
        img = cv2.imread(str(val_dir / r["image"]))
        if r["has_ball"]:
            cv2.circle(img, (int(float(r["gt"][0])), int(float(r["gt"][1]))), 28, (0, 255, 0), 3)
        if r["pred"]:
            cv2.circle(img, (int(r["pred"][0]), int(r["pred"][1])), 28, (0, 0, 255), 3)
        tiles.append(cv2.resize(img, (640, 360)))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = cv2.vconcat([cv2.hconcat(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        out = DATA_ROOT / "val_errors.jpg"
        cv2.imwrite(str(out), sheet)
        print("error sheet (green = ground truth, red = prediction):", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
