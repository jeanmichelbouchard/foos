"""Printable key cards (French) for the keyboard plugin (io_evdev_keyboard): letter page, two cards to cut out.

    python make_key_cards.py            writes doc/touches_clavier.pdf

The layout assumes keyboard_left_team = "yellow" (keys 1-2-3 = bleus, 8-9-0 = rouges): swap the colours if yours differs.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "doc" / "touches_clavier")
PW, PH = 215.9, 279.4          # letter, mm
BLUE, RED, GREY, DARK = "#1f3f8f", "#b01818", "#888888", "#222222"

fig = plt.figure(figsize=(PW / 25.4, PH / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, PW); ax.set_ylim(PH, 0); ax.axis("off")


def key(x, y, label, color=DARK, w=9.0, h=9.0, fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="white", ec=color, lw=1.6))
    ax.add_patch(FancyBboxPatch((x + 0.9, y + h - 1.6), w - 1.8, 0.9, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=color, ec="none", alpha=0.18))
    ax.text(x + w / 2, y + h / 2 - 0.3, label, ha="center", va="center", fontsize=fs, weight="bold", color=color)


def text(x, y, s, fs=10.5, **kw):
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=fs, color=kw.pop("color", DARK), **kw)


def card(x, y, w, h, title, subtitle=None):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=GREY, lw=0.8, ls=(0, (4, 3))))
    text(x + 5, y + 7, title, fs=14, weight="bold")
    if subtitle:
        text(x + w - 5, y + 7.3, subtitle, fs=8.5, color=GREY, ha="right")
    ax.plot([x + 5, x + w - 5], [y + 11.5, y + 11.5], color=DARK, lw=0.8)


# ---- card 1: score -------------------------------------------------------------------------------------------------
X, Y, W, H = 15.0, 22.0, 186.0, 78.0
card(X, Y, W, H, "POINTAGE", "les chiffres du haut du clavier")
blocks = [(X + 6, "BLEUS", BLUE, "123"), (X + W - 6 - 3 * 18, "ROUGES", RED, "890")]
for bx, name, col, keys in blocks:
    text(bx + 27, Y + 18, name, fs=11, weight="bold", color=col, ha="center")
    for i, (k, lab) in enumerate(zip(keys, ["−1 point", "But", "+1 point"])):
        key(bx + i * 18 + 4.5, Y + 23, k, col)
        text(bx + i * 18 + 9, Y + 37, lab, fs=9.5, ha="center")
cx = X + W / 2
text(cx, Y + 18, "REPRISE", fs=11, weight="bold", ha="center")
key(cx - 4.5, Y + 23, "5")
text(cx, Y + 37, "revoir le but", fs=9.5, ha="center")

ly = Y + 47
rows = [
    (["1", "3"], "ensemble  (ou ", ["8", "0"], ")  :  remettre le pointage à zéro"),
    (["1", "3"], "maintenus  :  menu    (", ["3", "1", "5"], ":  monter, descendre, choisir)"),
    (["."], "3 fois en 2 secondes  :  quitter le programme", None, ""),
]
for keys1, mid, keys2, end in rows:
    x = X + 7
    for k in keys1:
        key(x, ly - 3.5, k, w=7, h=7, fs=11); x += 8.5
    if keys2 is None:
        text(x + 1, ly, mid, fs=10)
    else:
        t = ax.text(x + 1, ly, mid, fontsize=10, va="center", color=DARK)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        x = bb.x1 + 1
        for k in keys2:
            key(x, ly - 3.5, k, w=7, h=7, fs=11); x += 8.5
        text(x + 0.5, ly, end, fs=10)
    ly += 10

# ---- card 2: goal type ---------------------------------------------------------------------------------------------
X, Y, W, H = 15.0, 115.0, 186.0, 100.0
card(X, Y, W, H, "TYPE DU DERNIER BUT", "corrige ce que la caméra a vu")
left = [("G", "Gardien", None), ("D", "Défense", None), ("C", "Centre", "le point est annulé"),
        ("T", "Attaque", None), ("A", "Auto-but", "le point est rendu")]
right = [("P", "Pissette", None), ("M", "Main courante", None), ("H", "Chevreuil", None),
         ("O", "One-timer", None), ("X", "Pas certain", None)]
for col_x, items in ((X + 8, left), (X + W / 2 + 4, right)):
    for i, (k, lab, note) in enumerate(items):
        yy = Y + 18 + i * 13.5
        key(col_x, yy, k, w=10, h=10, fs=15)
        if note:
            text(col_x + 14, yy + 3.2, lab, fs=12, weight="bold")
            text(col_x + 14, yy + 7.8, note, fs=9, color=RED if k == "C" else BLUE, style="italic")
        else:
            text(col_x + 14, yy + 5, lab, fs=12, weight="bold")
yy = Y + 18 + 5 * 13.5 + 1
ax.plot([X + 5, X + W - 5], [yy - 2, yy - 2], color=GREY, lw=0.5)
key(X + 8, yy + 1, "Z", w=10, h=10, fs=15)
text(X + 22, yy + 6, "Annuler la dernière correction", fs=12, weight="bold")

text(PW / 2, 228, "✂  Découper le long des pointillés", fs=9, color=GREY, ha="center")

fig.savefig(OUT + ".pdf")

