"""Lecture d'un clip, detection de la balle (K meilleures candidates par image) et suivi par continuite."""
import cv2
import numpy as np

def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


K = 5


TRAIN_FPS = 49.0     # cadence des clips d'entraînement: tous les réglages du classement sont en images à 49 i/s
TRAIN_FRAMES = 200


def resample_to_train_fps(frames, fps):
    """Ramène un clip filmé à `fps` à ~49 i/s en gardant les ~200 dernières images équivalentes (la fin du clip compte)."""
    if abs(fps - TRAIN_FPS) < 1:
        return frames
    n = len(frames)
    step = fps / TRAIN_FPS
    idx = sorted({max(0, int(round(n - 1 - j * step))) for j in range(TRAIN_FRAMES)})
    return [frames[i] for i in idx]


def track_clip_multi(model, path, conf=0.10, imgsz=1280, k=K, fps=TRAIN_FPS):
    frames = resample_to_train_fps(read_frames(path), fps)
    out = np.zeros((len(frames), k, 4), np.float32)
    for n, frame in enumerate(frames):
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, device=0)[0]
        if len(r.boxes):
            order = r.boxes.conf.argsort(descending=True)[:k]
            for j, i in enumerate(order.tolist()):
                x, y = r.boxes.xywh[i][:2].tolist()
                out[n, j] = (1, x, y, float(r.boxes.conf[i]))
    return out


VMAX = 200.0       # px par image (~9 m/s à 49 i/s sur cette table)
SLACK = 25.0       # tolérance de position
GAP = 14           # images manquantes tolérées dans une chaîne
CONF_MIN = 0.12
REWARD_BASE = 0.10   # récompense = conf - CONF_BASE... voir ci-dessous
GAP_COST = 0.15      # pénalité par image manquante
MAX_CHAINS = 3
MIN_CHAIN = 6        # nombre minimal de détections dans une chaîne conservée
INTERP_CONF = 0.30


def _nodes(arr):
    out = []
    for t in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            d, x, y, c = arr[t, k]
            if d > 0 and c >= CONF_MIN:
                out.append((t, x, y, c))
    return out


def _best_chain(nodes, used):
    """Chaîne de poids maximal (programmation dynamique sur les détections, triées par image)."""
    n = len(nodes)
    score = np.full(n, -np.inf)
    prev = np.full(n, -1)
    for i, (t, x, y, c) in enumerate(nodes):
        if used[i]:
            continue
        best = 0.0
        arg = -1
        for j in range(i - 1, -1, -1):
            tj = nodes[j][0]
            g = t - tj
            if g > GAP:
                break
            if g < 1 or used[j] or score[j] == -np.inf:
                continue
            dist = np.hypot(x - nodes[j][1], y - nodes[j][2])
            if dist > VMAX * g + SLACK:
                continue
            s = score[j] - GAP_COST * (g - 1) - 0.002 * dist
            if s > best:
                best, arg = s, j
        score[i] = best + c
        prev[i] = arg
    if not np.isfinite(score).any():
        return []
    i = int(np.argmax(score))
    chain = []
    while i != -1:
        chain.append(i)
        i = int(prev[i])
    return chain[::-1]


def track_continuous(arr):
    nodes = _nodes(arr)
    out = np.zeros((arr.shape[0], 4), np.float32)
    if not nodes:
        return out
    used = np.zeros(len(nodes), bool)
    for _ in range(MAX_CHAINS):
        chain = _best_chain(nodes, used)
        if len(chain) < MIN_CHAIN:
            break
        for i in chain:
            used[i] = True
        for a, b in zip(chain, chain[1:]):
            ta, xa, ya, ca = nodes[a]
            tb, xb, yb, cb = nodes[b]
            ta, tb = int(ta), int(tb)
            out[ta] = (1, xa, ya, ca)
            for t in range(ta + 1, tb):
                f = (t - ta) / (tb - ta)
                out[t] = (1, xa + f * (xb - xa), ya + f * (yb - ya), INTERP_CONF)
            out[tb] = (1, xb, yb, cb)
    return out


def main(min_chain=None):
    global MIN_CHAIN
    if min_chain:
        MIN_CHAIN = int(min_chain)
    src = DATA_ROOT / "tracks_top5"
    dst = DATA_ROOT / "tracks_cont"
    dst.mkdir(exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.npy")):
        np.save(dst / p.name, track_continuous(np.load(p)))
        n += 1
    print("%d clips traités -> %s" % (n, dst))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
