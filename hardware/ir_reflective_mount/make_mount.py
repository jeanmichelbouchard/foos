"""Printable mount for a reflective goal detector: one 5 mm IR led and one TSSP58038 receiver side by side.

    python make_mount.py            writes ir_reflective_mount.stl (and preview.png if matplotlib is installed)

Only numpy is needed. Change the dimensions below and run again.

Axes (mm):  x along the ball channel, y up (y = 0 is the channel floor), z across the channel (z = 0 is the wall the
mount is stuck on, z = DEPTH is the face that looks at the ball).  Print it with the z = 0 face on the bed: no supports.

The led goes in from the back (wall side) until its dome rests on the front iris; its legs are bent in the groove that
leaves by the top of the block.  The receiver slides down from the top into its slot, legs up, lens behind the window.
Both ends are bevelled, so a ball that rolls along the wall is pushed aside smoothly instead of hitting a square edge.
"""
import struct
import sys
from pathlib import Path

import numpy as np

# ---- dimensions -------------------------------------------------------------------------------------------------
WIDTH, HEIGHT, DEPTH = 44.0, 28.0, 15.0
AXIS_Y = 17.5            # height of the optical axis = centre of a 35 mm ball rolling on the floor

LED_X = -9.0             # led position along the block (0 = middle)
LED_BORE = 7.0           # 5 mm led (5.6 mm flange, measured) + heat-shrink tube
LED_IRIS = 4.5           # opening in front of the led dome
IRIS_DEPTH = 1.5
GROOVE_W, GROOVE_DEPTH = 4.0, 5.0     # groove for the bent led legs and wires, towards the top

RAMP_ANGLE = 25.0        # degrees between the wall and each bevelled end (smaller = gentler and longer)
RAMP_TIP = 0.8           # thickness left at the thin end of each bevel
RAMP = (DEPTH - RAMP_TIP) / np.tan(np.radians(RAMP_ANGLE))     # length of each bevel along the channel
LENGTH = WIDTH + 2 * RAMP

RX_X = 9.0               # receiver position along the block
RX_W = 5.6               # slot width  (TSSP58038 body measured: 5.0 mm wide, + foil tape)
RX_T = 5.2               # slot thickness along z (body measured: 4.8 mm with the lens)
RX_FLOOR = AXIS_Y - 3.5  # the receiver body (7 mm high, measured) rests here
RX_BODY_H = 7.3          # above this height the slot only holds the legs
RX_LEGS_W = 6.8          # slot width around the legs (they splay out to 5.8 mm, measured)
FRONT_WALL = 2.0         # plastic in front of the receiver = depth of the window (a small hood)
WIN_W, WIN_H = 4.0, 6.5   # narrower than the body: the receiver cannot fall out through the window

CIRCLE_STEP = 0.1        # resolution of the round holes


# ---- shape: True where there is plastic -------------------------------------------------------------------------
def thickness(x):
    """Depth of the part at position x, as a fraction of DEPTH: 1 on the body, going down to the tip on the bevels."""
    return 1 - (1 - RAMP_TIP / DEPTH) * np.clip((np.abs(x) - WIDTH / 2) / RAMP, 0, 1)


def shape(x, y, z):
    """The real part (bevels included)."""
    x, y, z = np.broadcast_arrays(x, y, z)
    return solid(x, y, z / thickness(x)) & (np.abs(x) < LENGTH / 2) & (y > 0) & (y < HEIGHT) & (z > 0) & (z < DEPTH * thickness(x))


def solid(x, y, z):
    """The part before the ends are bevelled: a plain block of LENGTH x HEIGHT x DEPTH with its holes."""
    s = np.ones(np.broadcast(x, y, z).shape, bool)
    r = np.hypot(x - LED_X, y - AXIS_Y)
    s &= ~((r < LED_BORE / 2) & (z < DEPTH - IRIS_DEPTH))
    s &= ~(r < LED_IRIS / 2)
    s &= ~((np.abs(x - LED_X) < GROOVE_W / 2) & (y > AXIS_Y) & (z < GROOVE_DEPTH))
    z0 = DEPTH - FRONT_WALL - RX_T
    w = np.where(y > RX_FLOOR + RX_BODY_H, RX_LEGS_W, RX_W)
    s &= ~((np.abs(x - RX_X) < w / 2) & (y > RX_FLOOR) & (z > z0) & (z < DEPTH - FRONT_WALL))
    s &= ~((np.abs(x - RX_X) < WIN_W / 2) & (y > RX_FLOOR + 0.5) & (y < RX_FLOOR + 0.5 + WIN_H) & (z >= DEPTH - FRONT_WALL))
    return s


def grid():
    fine = lambda c, r: np.arange(c - r, c + r + 1e-9, CIRCLE_STEP)
    xs = [-LENGTH / 2, -WIDTH / 2, WIDTH / 2, LENGTH / 2, LED_X - GROOVE_W / 2, LED_X + GROOVE_W / 2,
          RX_X - RX_W / 2, RX_X + RX_W / 2, RX_X - WIN_W / 2, RX_X + WIN_W / 2, RX_X - RX_LEGS_W / 2, RX_X + RX_LEGS_W / 2]
    ys = [0, HEIGHT, AXIS_Y, RX_FLOOR, RX_FLOOR + 0.5, RX_FLOOR + 0.5 + WIN_H, RX_FLOOR + RX_BODY_H]
    zs = [0, GROOVE_DEPTH, DEPTH - FRONT_WALL - RX_T, DEPTH - FRONT_WALL, DEPTH - IRIS_DEPTH, DEPTH]
    xs = np.unique(np.round(np.concatenate([xs, fine(LED_X, LED_BORE / 2)]), 4))
    ys = np.unique(np.round(np.concatenate([ys, fine(AXIS_Y, LED_BORE / 2)]), 4))
    zs = np.unique(np.round(zs, 4))
    return xs, ys, zs


def build():
    """All faces lie on one common grid, so the mesh is closed and every edge is shared by exactly two faces."""
    xs, ys, zs = grid()
    mid = lambda a: (a[1:] + a[:-1]) / 2
    X, Y, Z = np.meshgrid(mid(xs), mid(ys), mid(zs), indexing="ij")
    occ = np.pad(solid(X, Y, Z), 1)            # one empty cell all around
    P = [xs, ys, zs]
    tris = []
    for axis in range(3):
        a, b = [k for k in range(3) if k != axis]
        d = np.diff(occ.astype(np.int8), axis=axis)        # +1: empty -> solid (face looks to -axis), -1: the reverse
        for sign in (1, -1):
            idx = np.argwhere(d == sign)
            if not len(idx):
                continue
            c = P[axis][idx[:, axis]]                       # plane coordinate
            ia, ib = idx[:, a] - 1, idx[:, b] - 1
            a0, a1, b0, b1 = P[a][ia], P[a][ia + 1], P[b][ib], P[b][ib + 1]
            q = np.zeros((len(idx), 4, 3))
            q[:, :, axis] = c[:, None]
            q[:, :, a] = np.stack([a0, a1, a1, a0], 1)
            q[:, :, b] = np.stack([b0, b0, b1, b1], 1)
            # corner order a0b0, a1b0, a1b1, a0b1 is counter-clockwise seen from +axis when (axis, a, b) is cyclic
            ccw_plus = (axis, a, b) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            outward_plus = sign == -1
            if ccw_plus != outward_plus:
                q = q[:, ::-1]
            tris.append(q[:, [0, 1, 2]])
            tris.append(q[:, [0, 2, 3]])
    tris = np.concatenate(tris)
    tris[:, :, 2] *= thickness(tris[:, :, 0])       # bevel the two ends: the grid follows, the mesh stays closed
    return tris


def check(tris):
    v, inv = np.unique(np.round(tris.reshape(-1, 3), 4), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, n = np.unique(e, axis=0, return_counts=True)                       # directed edges: each exactly once
    _, m = np.unique(np.sort(e, 1), axis=0, return_counts=True)           # undirected edges: each exactly twice
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    return int((n != 1).sum()), int((m != 2).sum()), volume


def write_stl(path, tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tris
    with open(str(path), "wb") as f:
        f.write(b"ir_reflective_mount".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


def preview(path):
    """Four cuts through the part (plastic in grey), with the main dimensions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = 0.05
    x = np.arange(-LENGTH / 2 + st / 2, LENGTH / 2, st)
    y = np.arange(st / 2, HEIGHT, st)
    z = np.arange(st / 2, DEPTH, st)
    views = [
        ("Front face (looks at the ball)", shape(x[None, :], y[:, None], DEPTH - 0.2), (x, y), ("x: along the channel (mm)", "y: height above the floor (mm)")),
        ("Back face (stuck on the wall), seen from the wall", shape(-x[None, :], y[:, None], 0.2), (x, y), ("x, mirrored (mm)", "y (mm)")),
        ("Cut just under the optical axis, seen from above (wall at the bottom)", shape(x[None, :], AXIS_Y - 0.5, z[:, None]), (x, z), ("x (mm)", "z: across the channel (mm)")),
        ("Cut through the receiver slot, seen from the end (wall on the left)", shape(RX_X, y[:, None], z[None, :]), (z, y), ("z (mm)", "y (mm)")),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), gridspec_kw={"height_ratios": [HEIGHT, HEIGHT, DEPTH, HEIGHT]})
    for ax, (title, img, (u, v), (lu, lv)) in zip(axes.ravel(), views):
        ax.imshow(np.where(img, 0.35, 1.0), cmap="gray", vmin=0, vmax=1, origin="lower",
                  extent=(u[0] - st / 2, u[-1] + st / 2, v[0] - st / 2, v[-1] + st / 2))
        ax.set_title(title, fontsize=10); ax.set_xlabel(lu); ax.set_ylabel(lv); ax.set_aspect("equal")
        ax.grid(True, color="#4a90d9", alpha=0.25, linewidth=0.5)
    fig.suptitle("Reflective goal detector mount  %.0f x %g x %g mm (length x height x depth), ends bevelled at %g degrees, optical axis %.1f mm above the floor" % (LENGTH, HEIGHT, DEPTH, RAMP_ANGLE, AXIS_Y))
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    tris = build()
    bad1, bad2, vol = check(tris)
    print("triangles: %d | open or doubled edges: %d / %d | volume: %.1f cm3" % (len(tris), bad1, bad2, vol / 1000))
    write_stl(out / "ir_reflective_mount.stl", tris)
    if "--no-preview" not in sys.argv:
        try:
            preview(out / "preview.png")
        except ImportError:
            print("matplotlib not installed: no preview")
