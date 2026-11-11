"""Full analysis of a goal clip: video -> ball detection -> tracking -> goal type.

    a = Analyzer()
    a.analyze("clip.h264")  ->  {"type": "slice", "rod": "attack", "confidence": 0.8, "goal_frame": 123, "goal_side": "left", ...}
"""
import time

import os
from pathlib import Path

import joblib
import numpy as np
from ultralytics import YOLO

import rules
from rules import CONF, goal_events
from tracking import track_clip_multi, track_continuous

# Models: ball.pt (YOLO detector), rod.joblib, attack.joblib. Default folder: goal_service/models,
# or the FOOS_MODELS environment variable.
MODELS_DIR = Path(os.environ.get("FOOS_MODELS", Path(__file__).resolve().parent / "models"))

TYPE_NAMES = {"goalie": "goalie", "defense": "defense", "midfield": "center", "attack": "attack"}
# Thresholds chosen with calibrate.py (cross-validated probabilities): the rod confidence is only weakly informative
# (even at 95 %, 6 % errors), the sub-type confidence a little more; below the threshold we announce plain "attack".
MIN_CONFIDENCE = 0.50   # rod: below this, "unknown"
MIN_SUBTYPE_CONFIDENCE = 0.80


class Analyzer:
    def __init__(self, weights=None, models=None):
        self.yolo = YOLO(str(weights or MODELS_DIR / "ball.pt"))
        d = Path(models) if models else MODELS_DIR
        self.rod = joblib.load(d / "rod.joblib")
        self.attack = joblib.load(d / "attack.joblib")
        self.yolo.predict(np.zeros((720, 1280, 3), np.uint8), imgsz=1280, verbose=False, device=0)  # warm-up

    def analyze(self, path, fps=49.0):
        """fps: real frame rate of the clip (config.video_fps on the Pi); the classification settings are calibrated at 49 fps."""
        t0 = time.time()
        raw = track_clip_multi(self.yolo, path, fps=fps)
        res = self.classify_raw(raw)
        res["seconds"] = round(time.time() - t0, 2)
        return res

    def classify_raw(self, raw):
        """raw: N x K x 4 array of the best detections per frame (at 49 fps) -> goal type."""
        track = track_continuous(raw)
        c = {"track": track, "det": (track[:, 0] > 0) & (track[:, 3] >= CONF)}
        ev = goal_events(c)
        res = {"type": "unknown", "rod": None, "confidence": 0.0, "goal_frame": None, "goal_side": None}
        if ev:
            f, x, y, _, _ = ev[0]
            res.update(goal_frame=int(f), goal_side="left" if x < rules.W / 2 else "right")
            fr = rules.rod_features(c)
            p = self.rod["model"].predict_proba(fr[None])[0]
            rod = self.rod["names"][int(self.rod["model"].classes_[int(p.argmax())])]
            res.update(rod=rod, type=TYPE_NAMES[rod], confidence=float(p.max()))
            if rod == "attack":
                fa = np.concatenate([fr, rules.shape_features(c)])
                pa = self.attack["model"].predict_proba(fa[None])[0]
                sub = self.attack["names"][int(self.attack["model"].classes_[int(pa.argmax())])]
                res["subtype"] = sub
                res["subtype_confidence"] = float(pa.max())
                if sub != "normal" and pa.max() >= MIN_SUBTYPE_CONFIDENCE:
                    res["type"] = sub.replace(" ", "_").replace("-", "_")
            if res["confidence"] < MIN_CONFIDENCE:
                res["type"] = "unknown"
        return res
