"""Classify the rod that scored (goalie / defense / center / attack) from the ball trajectory.

Principle: take the trajectory (continuity tracking), find the frame where the ball vanishes into a goal, rotate the
image by 180 degrees if the goal is on the left (canonical frame: the goal scored is always on the right), then describe
the last ~60 frames (positions, speeds) and train a tree-based model. The rod that shoots is more or less far from the
goal scored: attack is close, goalie the furthest. The features themselves are defined in goal_service/rules.py
(`rod_features`), shared with the service so training and inference can never diverge.

Usage: python rod_classify.py [tracks_folder=tracks_cont]
Evaluation: cross-validation "one session at a time" (no session is seen both in training and in testing).
"""
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from analyze_tracks import load
from common import DATA_ROOT
from rules import ROD_NAMES, rod_features as features


def build(tracks_dir):
    X, y, groups, meta = [], [], [], []
    for c in load(DATA_ROOT / tracks_dir):
        if c["kind"] != "short":
            continue
        v = features(c)
        if v is None:
            continue
        X.append(v)
        y.append(ROD_NAMES.index(c["rod"]))
        groups.append(c["session"])
        meta.append(c)
    return np.array(X), np.array(y), np.array(groups), meta


def evaluate(X, y, groups, meta):
    pred = np.full(len(y), -1)
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4,
                                           class_weight="balanced", random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def report(y, pred, label, mask=None):
    if mask is not None:
        y, pred = y[mask], pred[mask]
    print("\n== %s (%d goals) ==  accuracy %.1f %%" % (label, len(y), 100 * np.mean(y == pred)))
    cm = confusion_matrix(y, pred, labels=range(4))
    print("rows = true rod, columns = predicted  (%s)" % ", ".join(ROD_NAMES))
    for i, row in enumerate(cm):
        tot = row.sum()
        print("  %-8s %s   recall %s" % (ROD_NAMES[i], " ".join("%4d" % v for v in row),
                                       "%.0f %%" % (100 * row[i] / tot) if tot else "-"))


def main(tracks_dir="tracks_cont"):
    X, y, groups, meta = build(tracks_dir)
    print("goals with a trajectory:", len(y), dict(Counter(ROD_NAMES[i] for i in y)))
    pred = evaluate(X, y, groups, meta)
    auto = np.array([c["autogoal"] for c in meta]) == 1
    white = np.array([c["ball"] for c in meta]) == "white"
    report(y, pred, "all goals")
    report(y, pred, "white ball, without own goals", mask=white & ~auto)
    report(y, pred, "own goals only", mask=auto)
    np.save(DATA_ROOT / "rod_pred.npy", np.array([[c["name"], ROD_NAMES[y[i]], ROD_NAMES[pred[i]]]
                                                   for i, c in enumerate(meta)]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tracks_cont")
