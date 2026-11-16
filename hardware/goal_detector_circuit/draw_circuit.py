"""Draws the goal detector circuit for a Raspberry Pi alone (no Arduino): two IR leds in series with one range
potentiometer, switched at 38 kHz by a 2N3906, and two TSSP58038 receivers.

    python draw_circuit.py          writes doc/goal_detector_circuit.png and .pdf

Needs matplotlib.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

RED, BLUE, BLACK, GREY = "#c62828", "#1565c0", "#111111", "#555555"
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_xlim(0, 132); ax.set_ylim(-42, 109); ax.set_aspect("equal"); ax.axis("off")


def wire(pts, color=BLACK):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.6, solid_capstyle="round", solid_joinstyle="round")


def dot(x, y, color=BLACK):
    ax.add_patch(Circle((x, y), 0.7, color=color, zorder=5))


def resistor_v(x, y0, y1, label, value, pot=False):
    """Vertical resistor from y0 (top) to y1 (bottom)."""
    n, a = 6, 1.3
    ys = [y0, y0 - 1] + [y0 - 1 - (y0 - y1 - 2) * (k + 0.5) / n for k in range(n)] + [y1 + 1, y1]
    xs = [x, x] + [x + a * (1 if k % 2 == 0 else -1) for k in range(n)] + [x, x]
    ax.plot(xs, ys, color=BLACK, lw=1.6)
    if pot:
        ax.annotate("", (x + 3.6, (y0 + y1) / 2 + 3.4), (x - 3.6, (y0 + y1) / 2 - 3.4), arrowprops={"arrowstyle": "-|>", "color": BLACK, "lw": 1.4})
    ax.text(x + 5, (y0 + y1) / 2 + 1.4, label, fontsize=10, weight="bold", va="center")
    ax.text(x + 5, (y0 + y1) / 2 - 1.8, value, fontsize=9, va="center")


def resistor_h(x0, x1, y, label, value):
    n, a = 6, 1.3
    xs = [x0, x0 + 1] + [x0 + 1 + (x1 - x0 - 2) * (k + 0.5) / n for k in range(n)] + [x1 - 1, x1]
    ys = [y, y] + [y + a * (1 if k % 2 == 0 else -1) for k in range(n)] + [y, y]
    ax.plot(xs, ys, color=BLACK, lw=1.6)
    ax.text((x0 + x1) / 2, y + 3.8, label, fontsize=10, weight="bold", ha="center")
    ax.text((x0 + x1) / 2, y - 4.6, value, fontsize=9, ha="center")


def led_v(x, y0, y1, label, where):
    """Vertical led, anode on top (y0), cathode at the bottom (y1)."""
    m = (y0 + y1) / 2
    wire([(x, y0), (x, m + 2)]); wire([(x, m - 2), (x, y1)])
    ax.add_patch(Polygon([(x - 2.3, m + 2), (x + 2.3, m + 2), (x, m - 2)], closed=True, fc="white", ec=BLACK, lw=1.6))
    ax.plot([x - 2.3, x + 2.3], [m - 2, m - 2], color=BLACK, lw=1.8)
    for d in (0, 1.8):
        ax.annotate("", (x + 6.2, m + 1.6 - d), (x + 3.2, m + 3.0 - d), arrowprops={"arrowstyle": "->", "color": BLACK, "lw": 1.1})
    ax.text(x + 8, m + 1.4, label, fontsize=10, weight="bold", va="center")
    ax.text(x + 8, m - 1.8, where, fontsize=9, va="center")


def receiver(x, y, label, where):
    ax.add_patch(Rectangle((x, y), 12, 12, fc="white", ec=BLACK, lw=1.6))
    ax.add_patch(Circle((x + 6, y + 12), 2.2, fc="white", ec=BLACK, lw=1.4, zorder=0))
    ax.text(x + 0.8, y + 6, "OUT", fontsize=8, va="center")
    ax.text(x + 11.2, y + 9, "Vs", fontsize=8, va="center", ha="right")
    ax.text(x + 11.2, y + 3, "GND", fontsize=8, va="center", ha="right")
    ax.text(x + 26.5, y + 7.5, label, fontsize=10, weight="bold", va="center")
    ax.text(x + 26.5, y + 4.3, where, fontsize=9, va="center")


def indicator(x, y_top, y_out, rname, dname):
    """Indicator led between 3.3 V (y_top) and a receiver output (y_out): lit while the output is low."""
    n, a, y1 = 6, 1.3, y_top - 9
    ys = [y_top, y_top - 1] + [y_top - 1 - 7 * (k + 0.5) / n for k in range(n)] + [y1 + 1, y1]
    xs = [x, x] + [x + a * (1 if k % 2 == 0 else -1) for k in range(n)] + [x, x]
    ax.plot(xs, ys, color=BLACK, lw=1.6)
    ax.text(x + 3.2, y_top - 3.2, rname, fontsize=10, weight="bold", va="center")
    ax.text(x + 3.2, y_top - 6.2, "1 kΩ", fontsize=9, va="center")
    m = (y1 + y_out) / 2
    wire([(x, y1), (x, m + 2)]); wire([(x, m - 2), (x, y_out)])
    ax.add_patch(Polygon([(x - 2.3, m + 2), (x + 2.3, m + 2), (x, m - 2)], closed=True, fc="#ffe082", ec=BLACK, lw=1.6))
    ax.plot([x - 2.3, x + 2.3], [m - 2, m - 2], color=BLACK, lw=1.8)
    ax.text(x - 3.4, m, dname, fontsize=10, weight="bold", va="center", ha="right")
    dot(x, y_out, BLUE)


# ---- Raspberry Pi header ----------------------------------------------------------------------------------------
ax.add_patch(Rectangle((4, 6), 23, 88, fc="#f3f6f1", ec=BLACK, lw=1.8))
ax.text(15.5, 96.5, "Raspberry Pi", fontsize=11, weight="bold", ha="center")
ax.text(15.5, 4, "40-pin header\n(physical pin numbers)", fontsize=8, ha="center", va="top", color=GREY)
PINS = {"5V": (90, "pin 2", "5V"), "3V3": (82, "pin 1", "3V3"), "B": (63, "pin 8", "GPIO14"), "Y": (35, "pin 26", "GPIO7"),
        "PWM": (18, "pin 12", "GPIO18 (PWM)"), "GND": (12, "pin 6", "GND")}
for y, pin, name in PINS.values():
    ax.text(26, y + 1.3, pin, fontsize=9, weight="bold", ha="right", va="center")
    ax.text(26, y - 1.5, name, fontsize=8, ha="right", va="center", color=GREY)

# ---- led chain: 5 V -> RV1 -> R1 -> LED1 -> LED2 -> Q1 emitter ; Q1 collector -> GND ------------------------------
X = 100
wire([(27, 90), (X, 90), (X, 88)], RED)
ax.text(60, 91.5, "5 V", fontsize=9, color=RED, ha="center")
resistor_v(X, 88, 78, "RV1", "10 kΩ trimmer (range)", pot=True)
wire([(X, 78), (X, 76)])
resistor_v(X, 76, 66, "R1", "330 Ω")
led_v(X, 66, 56, "LED1", "IR led, goal 1")
led_v(X, 56, 46, "LED2", "IR led, goal 2")
QY = 18
wire([(X, 46), (X, QY + 5.5)])
# Q1: PNP, emitter on top, collector at the bottom
ax.add_patch(Circle((X - 2, QY), 6.2, fc="white", ec=BLACK, lw=1.4))
ax.plot([X - 4.5, X - 4.5], [QY - 4, QY + 4], color=BLACK, lw=2.6)
wire([(X - 10, QY), (X - 4.5, QY)])
wire([(X, QY + 5.5), (X - 4.5, QY + 1.8)]); wire([(X - 4.5, QY - 1.8), (X, QY - 5.5), (X, 4)])
ax.annotate("", (X - 3.9, QY + 2.3), (X - 0.9, QY + 4.75), arrowprops={"arrowstyle": "-|>", "color": BLACK, "lw": 1.4})
ax.text(X + 6, QY + 1.4, "Q1", fontsize=10, weight="bold", va="center")
ax.text(X + 6, QY - 1.8, "2N3906 (PNP)", fontsize=9, va="center")
ax.text(X + 0.9, QY + 6.6, "E", fontsize=8); ax.text(X + 0.9, QY - 8.0, "C", fontsize=8); ax.text(X - 9.4, QY + 1.0, "B", fontsize=8)
resistor_h(X - 22, X - 10, QY, "R2", "1 kΩ")
wire([(27, QY), (X - 22, QY)], BLUE)
ax.text(40, QY + 1.3, "38 kHz", fontsize=9, color=BLUE, ha="center")

# ---- receivers --------------------------------------------------------------------------------------------------
receiver(44, 57, "IR1  TSSP58038", "receiver, goal 1")
receiver(44, 29, "IR2  TSSP58038", "receiver, goal 2")
wire([(27, 63), (44, 63)], BLUE); wire([(27, 35), (44, 35)], BLUE)
wire([(27, 82), (67, 82), (67, 38), (56, 38)], RED); wire([(67, 66), (56, 66)], RED); dot(67, 66, RED)
ax.text(50, 83.5, "3.3 V", fontsize=9, color=RED, ha="center")
wire([(56, 60), (62, 60), (62, 4)]); wire([(56, 32), (62, 32)]); dot(62, 32)

# ---- indicator leds: lit while the receiver sees the 38 kHz light (output low) -----------------------------------
indicator(35, 82, 63, "R3", "D1")
dot(35, 82, RED)
indicator(35, 54, 35, "R4", "D2")
wire([(32.5, 54), (37.5, 54)], RED)
ax.text(35, 55.2, "3.3 V", fontsize=8, color=RED, ha="center")

# ---- ground -----------------------------------------------------------------------------------------------------
wire([(27, 12), (31, 12), (31, 4), (X, 4)]); dot(62, 4)
ax.text(46, 0.6, "GND", fontsize=9, ha="center")

# ---- title and notes --------------------------------------------------------------------------------------------
ax.text(66, 108, "Goal detectors on a Raspberry Pi alone: two IR leds in series, one range trimmer", fontsize=13, weight="bold", ha="center", va="top")
notes = [
    "RV1 sets the IR power, so the detection range, of both goals at once. Use the wiper and one end of the trimmer.",
    "   Light barrier (ir_barrier_reflective = False): turn RV1 to 0 Ω.   Reflective (True): raise RV1 until the opposite wall no longer triggers",
    "   a goal, then check that the ball still does.",
    "D1, D2: indicator leds, red or yellow (not white or blue: 3.3 V is too low for them). Lit while the receiver sees the IR light:",
    "   reflective = lit when the ball is in front of the detector;   light barrier = lit at rest, off while the ball cuts the beam.",
    "Q1 is wired as an emitter follower: leds on while GPIO18 is low (about 6 mA at most), fully off while it is high, even with a 3.3 V GPIO.",
    "The receivers are powered from 3.3 V so their output never exceeds what a GPIO accepts. Optional: 100 nF between Vs and GND at each receiver.",
    "Receiver pins 8 and 26 must match io_raspberry_pins in config.py (pin 8 = team black, pin 26 = team yellow in config.py.sample).",
    "Pinouts, legs down:  2N3906, flat face towards you: E  B  C.     TSSP58038, lens towards you: OUT  GND  Vs.     Leds: long leg = anode (+).",
    "To test without foos, the 38 kHz must still run. Start it with:",
    "   gpio mode 1 pwm && gpio pwm-ms && gpio pwmr 5 && gpio pwmc 101 && gpio pwm 1 2",
]
for k, t in enumerate(notes):
    ax.text(3, -6 - 3.3 * k, t, fontsize=8.6, va="top")
ax.plot([3, 129], [-3.6, -3.6], color=GREY, lw=0.6)

out = Path(__file__).resolve().parents[2] / "doc"
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
fig.savefig(str(out / "goal_detector_circuit.png"), dpi=150)
fig.savefig(str(out / "goal_detector_circuit.pdf"))
print("written to", out)
