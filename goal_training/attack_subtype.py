"""Sub-types of attack goals (normal / slice / wall shot / chevreuil / one-timer) from the trajectory.

Reuses the canonical frame of rod_classify.py (goal scored on the right) and adds features about the shape of the shot:
start position and angle, direction reversal before the shot (chevreuil: the opponent's ball is intercepted), arrival
speed before the hit (one-timer: no stop), distance to the edge of the table (wall shot). The shape features are defined
in goal_service/rules.py (`shape_features`), shared with the service.

Usage: python attack_subtype.py [tracks_folder=tracks_cont]
"""
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from analyze_tracks import load
from common import DATA_ROOT
from rules import NAMES, goal_events, rod_features as features, shape_features

# Folder name of the clips -> sub-type index
SUB = {"": 0, "Slice": 1, "Wall Shot": 2, "Chevreuil": 3, "One-Timer": 4}


def build(tracks_dir):
    X, y, groups, meta = [], [], [], []
    for c in load(DATA_ROOT / tracks_dir):
        if c["kind"] != "short" or c["rod"] != "attack" or not goal_events(c):
            continue
        X.append(np.concatenate([features(c), shape_features(c)]))
        y.append(SUB[c["subtype"]])
        groups.append(c["session"])
        meta.append(c)
    return np.array(X), np.array(y), np.array(groups), meta


def main(tracks_dir="tracks_cont"):
    X, y, groups, meta = build(tracks_dir)
    white = np.array([c["ball"] for c in meta]) == "white"
    auto = np.array([c["autogoal"] for c in meta]) == 1
    keep = white & ~auto
    X, y, groups = X[keep], y[keep], groups[keep]
    print("attack goals (white ball, without own goals):", len(y), dict(Counter(NAMES[i] for i in y)))
    pred = np.full(len(y), -1)
    for g in sorted(set(groups)):
        te = groups == g
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=3,
                                           class_weight="balanced", random_state=0)
        m.fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    print("\naccuracy %.1f %%  (always \"normal\": %.1f %%)" % (100 * np.mean(pred == y), 100 * np.mean(y == 0)))
    cm = confusion_matrix(y, pred, labels=range(5))
    print("rows = true, columns = predicted (%s)" % ", ".join(NAMES))
    for i, row in enumerate(cm):
        print("  %-10s %s   recall %.0f %%" % (NAMES[i], " ".join("%4d" % v for v in row), 100 * row[i] / row.sum()))
    # "special" vs "normal" grouping
    sp_true, sp_pred = y > 0, pred > 0
    print("\nspecial vs normal: detects %.0f %% of the special shots, %.0f %% of the normal ones labelled special"
          % (100 * np.mean(sp_pred[sp_true]), 100 * np.mean(sp_pred[~sp_true])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tracks_cont")
