"""Test the center-goal rule of plugins/goal_analysis.py with a fake bus (no service, no GPU needed).

    python test_center_cancel.py [path_to_foos_repo]

A center-rod goal does not count unless it is an own goal. Scenarios: automatic cancellation by the model, the own-goal
key giving the point back, a player declaring a center goal, low confidence, feature disabled, match just won, manual
removal of a goal.
"""
import sys
import tempfile
import time
import types
from pathlib import Path

repo = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2])
tmp = tempfile.mkdtemp()
cfg = types.ModuleType("foos.config")
cfg.replay_path = tmp
cfg.goal_clips_path = tmp
cfg.goal_service_url = None
cfg.center_goals_auto_cancel = True
cfg.center_cancel_confidence = 0.90
pkg = types.ModuleType("foos")
pkg.config = cfg
sys.modules["foos"], sys.modules["foos.config"] = pkg, cfg
sys.path.insert(0, repo)
from plugins import goal_analysis    # noqa: E402


class Bus:
    """Synchronous bus: handlers run immediately, like in a single thread (the plugin must cope with its own events)."""
    def __init__(self):
        self.handlers, self.events = {}, []

    def subscribe_map(self, fmap, thread=False):
        for k, f in fmap.items():
            self.handlers.setdefault(k, []).append(f)

    def notify(self, name, data=None):
        self.events.append((name, data))
        for f in self.handlers.get(name, []):
            f(data)

    def count(self, name):
        return sum(1 for n, _ in self.events if n == name)


def make(enabled=True):
    cfg.center_goals_auto_cancel = enabled
    bus = Bus()
    return goal_analysis.Plugin(bus), bus


def predict(p, team, type_, conf):
    goal = p._new_goal(team)
    p._apply(goal, {"type": type_, "confidence": conf}, time.time())
    return goal


# 1. the model says "center" with a high confidence: the point is removed once
p, bus = make()
g = predict(p, "yellow", "center", 0.99)
assert g["cancelled"] and bus.count("decrement_score") == 1 and bus.count("increment_score") == 0
assert bus.events[-1][1]["type"] == "center_void"
assert p.stats_lines("yellow")[0].endswith("0") and any("Center (void)" in l for l in p.stats_lines("yellow"))
assert len(p.match) == 1, "the automatic cancellation must not be treated as a manual removal"

# 2. the own-goal key gives the point back
p.on_key({"type": "own_goal"})
assert not g["cancelled"] and bus.count("increment_score") == 1
assert bus.events[-1][1]["type"] == "own_goal"
lines = p.stats_lines("yellow")
assert lines[0].endswith("1") and any("Own goals" in l for l in lines)
# ... and 'undo' puts the cancellation back
p.on_key({"type": "undo"})
assert g["cancelled"] and bus.count("decrement_score") == 2

# 3. any other type key also gives the point back (the model was wrong), a center key cancels it again
p.on_key({"type": "attack"})
assert not g["cancelled"] and bus.count("increment_score") == 2
p.on_key({"type": "center"})
assert g["cancelled"] and bus.count("decrement_score") == 3

# 4. the model says attack, a player says center: cancelled
p, bus = make()
g = predict(p, "black", "attack", 0.99)
assert bus.count("decrement_score") == 0
p.on_key({"type": "center"})
assert g["cancelled"] and bus.count("decrement_score") == 1

# 5. low confidence: nothing happens
p, bus = make()
g = predict(p, "black", "center", 0.5)
assert not g["cancelled"] and bus.count("decrement_score") == 0

# 6. feature disabled: nothing happens
p, bus = make(enabled=False)
g = predict(p, "black", "center", 0.99)
assert not g["cancelled"] and bus.count("decrement_score") == 0

# 7. the match has just been won: the score is not touched, the goal is flagged for review
p, bus = make()
g = p._new_goal("yellow")
g["created"] = time.time() - 5
p.last_win = time.time()
p._apply(g, {"type": "center", "confidence": 0.99}, time.time())
assert not g["cancelled"] and g["review"] and bus.count("decrement_score") == 0

# 8. a manual removal takes out the latest goal that is not cancelled
p, bus = make()
a = predict(p, "yellow", "attack", 0.99)
b = predict(p, "yellow", "center", 0.99)      # cancelled
p.on_decrement({"team": "yellow"})            # manual
assert b in p.match and a not in p.match

print("all center-goal scenarios OK")
