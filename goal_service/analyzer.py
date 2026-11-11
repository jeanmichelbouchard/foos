"""Analyse complète d'un clip de but: vidéo -> détection de la balle -> suivi -> type de but.

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

# Modeles: ball.pt (detecteur YOLO), rod.joblib, attack.joblib. Dossier par defaut: goal_service/models,
# ou la variable d'environnement FOOS_MODELS.
MODELS_DIR = Path(os.environ.get("FOOS_MODELS", Path(__file__).resolve().parent / "models"))

TYPE_NAMES = {"goalie": "goalie", "defense": "defense", "midfield": "center", "attack": "attack"}
# Seuils choisis avec calibrate.py (probabilités en validation croisée): la confiance de la tige est peu informative
# (même à 95 %, 6 % d'erreurs), celle du sous-type l'est un peu plus; en dessous du seuil on annonce « attaque ».
MIN_CONFIDENCE = 0.50   # tige: en dessous, « unknown »
MIN_SUBTYPE_CONFIDENCE = 0.80


class Analyzer:
    def __init__(self, weights=None, models=None):
        self.yolo = YOLO(str(weights or MODELS_DIR / "ball.pt"))
        d = Path(models) if models else MODELS_DIR
        self.rod = joblib.load(d / "rod.joblib")
        self.attack = joblib.load(d / "attack.joblib")
        self.yolo.predict(np.zeros((720, 1280, 3), np.uint8), imgsz=1280, verbose=False, device=0)  # préchauffage

    def analyze(self, path, fps=49.0):
        """fps: cadence réelle du clip (config.video_fps du Pi); les réglages du classement sont calibrés à 49 i/s."""
        t0 = time.time()
        raw = track_clip_multi(self.yolo, path, fps=fps)
        res = self.classify_raw(raw)
        res["seconds"] = round(time.time() - t0, 2)
        return res

    def classify_raw(self, raw):
        """raw: tableau N x K x 4 des meilleures détections par image (à 49 i/s) -> type du but."""
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
