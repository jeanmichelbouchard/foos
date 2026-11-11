"""Regles de but et caracteristiques de classement (copie fidele des modules de recherche, sans dependance aux donnees).

Les nombres ci-dessous sont lies aux modeles entraines (rod.joblib, attack.joblib): ne les changez pas sans reentrainer.
Toutes les positions sont en pixels sur une image de 1280 x 720 prise par la camera du Pi a 49 images/s.
"""
import numpy as np

CONF = 0.30      # confiance minimale pour compter une detection comme « balle vue »
FPS = 49.0

X_LEFT, X_RIGHT = 240, 1040      # zone de but: x < X_LEFT ou x > X_RIGHT
Y_MIN, Y_MAX = 230, 500          # ouverture du but (axe vertical)
MIN_ABSENCE = 15                 # ~0,3 s à 49 images/s


def in_goal_zone(x, y):
    return (x < X_LEFT or x > X_RIGHT) and Y_MIN <= y <= Y_MAX


def goal_events(c, min_absence=MIN_ABSENCE):
    """Liste des (image_de_disparition, x, y, durée_absence, revient) pour chaque disparition dans une zone de but."""
    det, t = c["det"], c["track"]
    n = len(det)
    events, i = [], 0
    while i < n:
        if det[i]:
            i += 1
            continue
        j = i
        while j < n and not det[j]:
            j += 1
        if i > 0 and (j - i) >= min_absence and in_goal_zone(t[i - 1, 1], t[i - 1, 2]):
            events.append((i - 1, float(t[i - 1, 1]), float(t[i - 1, 2]), j - i, j < n))
        i = j
    return events


W, H = 1280.0, 720.0
LAGS = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60]
ROD_NAMES = ["goalie", "defense", "midfield", "attack"]


def canonical(track, f):
    """Positions (x, y) des images <= f, retournées de 180° si le but est à gauche; NaN si pas de détection."""
    xy = np.where(track[:, :1] > 0, track[:, 1:3], np.nan).astype(np.float64)
    if track[f, 1] < W / 2:
        xy = np.stack([W - xy[:, 0], H - xy[:, 1]], axis=1)
    return xy


def rod_features(c):
    ev = goal_events(c)
    if not ev:
        return None
    f = ev[0][0]
    xy = canonical(c["track"], f)
    feats = []
    for k in LAGS:
        t = f - k
        feats += list(xy[t]) if t >= 0 else [np.nan, np.nan]
    # vitesses (px/image) sur 2 images aux mêmes retards
    sp = np.full(len(xy), np.nan)
    sp[2:] = np.hypot(*(xy[2:] - xy[:-2]).T) / 2
    for k in LAGS:
        t = f - k
        feats.append(sp[t] if t >= 0 else np.nan)
    win = sp[max(0, f - 60): f + 1]
    if np.isfinite(win).any():
        i = int(np.nanargmax(win))
        t_peak = max(0, f - 60) + i
        feats += [float(np.nanmax(win)), float(xy[t_peak, 0]), float(f - t_peak)]
        # point de départ du tir: dernière image avant le pic où la balle était lente (< 35 % du pic)
        slow = np.where(sp[: t_peak + 1] < 0.35 * np.nanmax(win))[0]
        t0 = int(slow.max()) if len(slow) else t_peak
        feats += [float(xy[t0, 0]), float(xy[t0, 1]), float(f - t0)]
    else:
        feats += [np.nan] * 6
    ok = np.isfinite(xy[max(0, f - 60): f + 1, 0])
    feats.append(float(ok.sum()))
    return np.array(feats, np.float64)


NAMES = ["normal", "slice", "wall shot", "chevreuil", "one-timer"]

def shape_features(c):
    ev = goal_events(c)
    f = ev[0][0]
    xy = canonical(c["track"], f)
    lo = max(0, f - 90)
    seg = xy[lo: f + 1]
    ok = np.isfinite(seg[:, 0])
    sp = np.full(len(xy), np.nan)
    sp[2:] = np.hypot(*(xy[2:] - xy[:-2]).T) / 2
    vx = np.full(len(xy), np.nan)
    vx[2:] = (xy[2:, 0] - xy[:-2, 0]) / 2
    win = sp[lo: f + 1]
    out = []
    if np.isfinite(win).any():
        t_peak = lo + int(np.nanargmax(win))
        peak = float(np.nanmax(win))
        slow = np.where(sp[: t_peak + 1] < 0.35 * peak)[0]
        t0 = int(slow.max()) if len(slow) else t_peak
    else:
        t_peak, peak, t0 = f, np.nan, f
    # départ du tir
    out += [xy[t0, 0], xy[t0, 1], min(xy[t0, 1], H - xy[t0, 1])]
    # direction du tir: angle entre le départ et l'arrivée
    d = xy[f] - xy[t0]
    out += [np.degrees(np.arctan2(d[1], d[0])) if np.isfinite(d).all() else np.nan, float(np.hypot(*d)) if np.isfinite(d).all() else np.nan]
    # avant le tir (60 images avant le départ): vitesse, mouvement vers le but adverse ou vers nous
    pre = slice(max(0, t0 - 40), t0 + 1)
    out += [np.nanmean(sp[pre]) if np.isfinite(sp[pre]).any() else np.nan,
            np.nanmax(sp[pre]) if np.isfinite(sp[pre]).any() else np.nan,
            np.nanmean(vx[pre]) if np.isfinite(vx[pre]).any() else np.nan,
            np.nanmin(vx[pre]) if np.isfinite(vx[pre]).any() else np.nan]
    # arrêt avant le tir: nombre d'images lentes (< 3 px/image) dans les 20 images précédant le départ
    sl = sp[max(0, t0 - 20): t0 + 1]
    out += [float(np.sum(sl < 3)) if np.isfinite(sl).any() else np.nan, float(np.isfinite(sl).sum())]
    # tir lui-même
    out += [peak, float(f - t0), float(np.nanstd(seg[ok, 1])) if ok.sum() > 2 else np.nan]
    out += [float(np.nanmin(seg[:, 1])) if ok.any() else np.nan, float(np.nanmax(seg[:, 1])) if ok.any() else np.nan]
    # nombre de retournements de direction en x dans les 90 dernières images (rebonds / interception)
    v = vx[lo: f + 1]
    v = v[np.isfinite(v)]
    s = np.sign(v[np.abs(v) > 4]) if len(v) else np.array([])
    out.append(float(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0.0)
    return np.array(out, np.float64)
