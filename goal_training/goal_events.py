"""Improved goal rule: a goal = the ball is seen in a goal zone, then is not seen anywhere for >= MIN_ABSENCE frames.

Unlike goal_from_track.py, we do NOT require that the ball never comes back: in many clips a new ball is put in play
before the end of the clip (checked visually on the fail_*.jpg sheets). The rule itself lives in goal_service/rules.py
(`goal_events`), shared with the service; this script only reports statistics about it.
"""
from collections import Counter

import numpy as np

from analyze_tracks import load
from rules import goal_events


def main():
    clips = [c for c in load() if c["kind"] == "short"]
    n_events = Counter()
    first_frames, no_event = [], []
    ret_after = Counter()
    for c in clips:
        ev = goal_events(c)
        n_events[len(ev)] += 1
        if not ev:
            no_event.append(c)
        else:
            first_frames.append(ev[0][0])
            ret_after[ev[0][4]] += 1
    found = len(clips) - len(no_event)
    print("Recall with the improved rule: %d/%d = %.1f %%" % (found, len(clips), 100 * found / len(clips)))
    print("Number of 'vanishing into a goal' events per clip:", dict(sorted(n_events.items())))
    print("The first event is followed by the ball coming back in the clip:", dict(ret_after))
    print("Frame of the first event: median %d (10th-90th pc: %d-%d)" % (
        np.median(first_frames), np.percentile(first_frames, 10), np.percentile(first_frames, 90)))
    print("Clips without an event (%d): rod %s, sessions %s" % (
        len(no_event), dict(Counter(c["rod"] for c in no_event)),
        dict(sorted(Counter(c["session"] for c in no_event).items()))))
    multi = sum(v for k, v in n_events.items() if k >= 2)
    print("Clips with 2 events or more (ball lost near a goal, then a new disappearance): %d" % multi)


if __name__ == "__main__":
    main()
