"""Honest measure of the confidence: probabilities obtained by cross-validation (one session at a time), then accuracy
and coverage as a function of the confidence threshold. Used to choose MIN_CONFIDENCE in goal_service/analyzer.py.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import attack_subtype
import rod_classify


def cv_proba(X, y, groups, depth):
    P = np.zeros((len(y), 4 if depth == 4 else 5))
    for g in sorted(set(groups)):
        te = groups == g
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=depth,
                                           class_weight="balanced", random_state=0).fit(X[~te], y[~te])
        P[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
    return P


def table(title, P, y):
    pred, conf = P.argmax(1), P.max(1)
    print("\n%s" % title)
    print("threshold  answers   accuracy of the answers   errors avoided")
    err_all = np.sum(pred != y)
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        keep = conf >= t
        if keep.sum() == 0:
            continue
        acc = np.mean(pred[keep] == y[keep])
        print("%5.2f   %5.1f %%   %5.1f %%                   %d/%d" % (
            t, 100 * keep.mean(), 100 * acc, err_all - np.sum(pred[keep] != y[keep]), err_all))


def main():
    X, y, g, meta = rod_classify.build("tracks_cont")
    k = (np.array([c["ball"] for c in meta]) == "white") & (np.array([c["autogoal"] for c in meta]) == 0)
    table("ROD (white ball, without own goals)", cv_proba(X[k], y[k], g[k], 4), y[k])
    X, y, g, meta = attack_subtype.build("tracks_cont")
    k = (np.array([c["ball"] for c in meta]) == "white") & (np.array([c["autogoal"] for c in meta]) == 0)
    table("ATTACK SUB-TYPE", cv_proba(X[k], y[k], g[k], 3), y[k])


if __name__ == "__main__":
    main()
