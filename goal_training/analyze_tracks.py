"""Track analysis: where and when the ball disappears at the end of each clip."""
import numpy as np

from common import DATA_ROOT, list_clips
from rules import CONF, FPS

TRACKS = DATA_ROOT / "tracks"


def load(tracks_dir=TRACKS):
    out = []
    for c in list_clips():
        p = tracks_dir / (c["name"] + ".npy")
        if p.exists():
            t = np.load(p)
            c["track"] = t
            c["det"] = (t[:, 0] > 0) & (t[:, 3] >= CONF)
            out.append(c)
    return out


def last_seen(c):
    idx = np.where(c["det"])[0]
    if len(idx) == 0:
        return None
    L = int(idx.max())
    return L, float(c["track"][L, 1]), float(c["track"][L, 2])


def main():
    clips = [c for c in load() if c["kind"] == "short"]
    print("short clips analysed:", len(clips))
    rows = []
    for c in clips:
        ls = last_seen(c)
        n = len(c["det"])
        rate = c["det"][: (ls[0] + 1)].mean() if ls else 0.0
        rows.append((c, ls, n, rate))
    nodet = [r for r in rows if r[1] is None]
    print("clips without any detection:", len(nodet))
    rows = [r for r in rows if r[1] is not None]
    L = np.array([r[1][0] for r in rows])
    X = np.array([r[1][1] for r in rows])
    Y = np.array([r[1][2] for r in rows])
    gap = np.array([r[2] - 1 - r[1][0] for r in rows])
    rate = np.array([r[3] for r in rows])
    print("detection rate before the disappearance: median %.2f, 10th percentile %.2f" %
          (np.median(rate), np.percentile(rate, 10)))
    print("last frame with the ball: median %d, 10th-90th pc: %d-%d (clip of %d frames)" %
          (np.median(L), np.percentile(L, 10), np.percentile(L, 90), rows[0][2]))
    print("final absence (frames): median %d, minimum %d" % (np.median(gap), gap.min()))
    print("histogram of x of the last position (80 px bins):")
    h, edges = np.histogram(X, bins=np.arange(0, 1361, 80))
    for k, v in enumerate(h):
        print("  x %4d-%4d : %s %d" % (edges[k], edges[k + 1], "#" * int(v / 4), v))
    print("histogram of y (80 px bins):")
    h, edges = np.histogram(Y, bins=np.arange(0, 721, 80))
    for k, v in enumerate(h):
        print("  y %4d-%4d : %s %d" % (edges[k], edges[k + 1], "#" * int(v / 4), v))


if __name__ == "__main__":
    main()
