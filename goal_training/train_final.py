"""Train the two classifiers (rod, attack sub-type) on ALL the goals and save them for the service.

Usage: python train_final.py [tracks_folder=tracks_cont]
Output: $FOOS_ML_DATA/models/rod.joblib, $FOOS_ML_DATA/models/attack.joblib
(The figures printed by rod_classify.py / attack_subtype.py remain the honest measure; this model has seen all the goals.)
Copy the two files, together with the ball detector (ball.pt), into goal_service/models/.
"""
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import attack_subtype
import rod_classify
from common import DATA_ROOT
from rules import NAMES, ROD_NAMES


def fit(X, y, depth):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=depth,
                                       class_weight="balanced", random_state=0)
    return m.fit(X, y)


def main(tracks_dir="tracks_cont"):
    out = DATA_ROOT / "models"
    out.mkdir(exist_ok=True)
    X, y, g, meta = rod_classify.build(tracks_dir)
    keep = (np.array([c["ball"] for c in meta]) == "white") & (np.array([c["autogoal"] for c in meta]) == 0)
    joblib.dump({"model": fit(X[keep], y[keep], 4), "names": ROD_NAMES}, out / "rod.joblib")
    print("rod: %d goals" % keep.sum())
    X, y, g, meta = attack_subtype.build(tracks_dir)
    keep = (np.array([c["ball"] for c in meta]) == "white") & (np.array([c["autogoal"] for c in meta]) == 0)
    joblib.dump({"model": fit(X[keep], y[keep], 3), "names": NAMES}, out / "attack.joblib")
    print("sub-type: %d goals" % keep.sum())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tracks_cont")
