"""Evaluate goal detection "by the camera alone": the ball vanishes inside a goal zone and does not come back.

Rule tested (goal zone = band near the left/right edge, at the level of the goal opening; constants in goal_service/rules.py):
  goal = last position of the ball inside the goal zone, then an absence >= MIN_ABSENCE frames until the end of the clip.
Approximate false alarm = same pattern (goal zone then absence >= MIN_ABSENCE) but the ball COMES BACK afterwards,
measured only in the part of the play that precedes the goal.
"""
from collections import Counter

from analyze_tracks import last_seen, load
from rules import FPS, MIN_ABSENCE, X_LEFT, X_RIGHT, in_goal_zone


def main():
    clips = [c for c in load() if c["kind"] == "short"]
    ok, bad = [], []
    for c in clips:
        L, x, y = last_seen(c)
        gap = len(c["det"]) - 1 - L
        (ok if in_goal_zone(x, y) and gap >= MIN_ABSENCE else bad).append((c, L, x, y, gap))
    print("Recall (goal found by the camera alone): %d/%d = %.1f %%" %
          (len(ok), len(clips), 100 * len(ok) / len(clips)))

    # Where do we fail?
    print("\nClips not found (%d): breakdown by rod / ball / session" % len(bad))
    print("  rod    :", dict(Counter(c["rod"] for c, *_ in bad)))
    print("  ball   :", dict(Counter(c["ball"] for c, *_ in bad)))
    print("  session:", dict(sorted(Counter(c["session"] for c, *_ in bad).items())))
    print("  total per rod    :", dict(Counter(c["rod"] for c in clips)))
    print("  total per session:", dict(sorted(Counter(c["session"] for c in clips).items())))
    why = Counter()
    for c, L, x, y, gap in bad:
        if gap < MIN_ABSENCE:
            why["ball still seen at the end of the clip"] += 1
        elif not (x < X_LEFT or x > X_RIGHT):
            why["vanishes far from the goals (middle of the field)"] += 1
        else:
            why["vanishes near an edge but outside the goal opening (y)"] += 1
    print("  causes:", dict(why))

    # Approximate false alarms in the part played before the goal
    events, frames = 0, 0
    clips_with = 0
    for c in clips:
        L, _, _ = last_seen(c)
        det, t = c["det"], c["track"]
        seg = det[: max(0, L - MIN_ABSENCE)]
        frames += len(seg)
        i, n, found = 0, len(seg), 0
        while i < n:
            if seg[i]:
                i += 1
                continue
            j = i
            while j < n and not seg[j]:
                j += 1
            # absence from i to j-1, the ball comes back at j (if j < n); last position seen = i-1
            if j < n and j - i >= MIN_ABSENCE and i > 0 and in_goal_zone(t[i - 1, 1], t[i - 1, 2]):
                found += 1
            i = j
        events += found
        clips_with += found > 0
    minutes = frames / FPS / 60
    print("\nApproximate false alarms (play before the goal, ball coming back afterwards):")
    print("  %d events in %d clips out of %d, for %.1f minutes of play analysed -> %.1f per minute" %
          (events, clips_with, len(clips), minutes, events / minutes))
    print("  (analysed play = 1 to 3 s before each goal, not a whole match: indicative value only)")


if __name__ == "__main__":
    main()
