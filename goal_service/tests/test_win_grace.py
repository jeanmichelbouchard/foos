"""Test the grace period before a victory (plugins/game.py) with a fake bus.

    python test_win_grace.py [path_to_foos_repo]

The victory is declared win_check_delay seconds after the end of the replay, and not before the goal analysis announced by
plugins/goal_analysis.py is done (at most win_hold_max seconds): a center goal scored as the last point can still be removed.
"""
import sys
import time
import types
from pathlib import Path

repo = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2])
cfg = types.ModuleType("foos.config")
cfg.game_modes = [(None, None), (10, None)]
cfg.win_check_delay = 0.3
cfg.win_hold_max = 1.5
ui = types.ModuleType("foos.ui.ui")
ui.registerMenu = lambda f: None
ui_pkg = types.ModuleType("foos.ui")
ui_pkg.ui = ui
pkg = types.ModuleType("foos")
pkg.config, pkg.ui = cfg, ui_pkg
sys.modules.update({"foos": pkg, "foos.config": cfg, "foos.ui": ui_pkg, "foos.ui.ui": ui})
sys.path.insert(0, repo)
from plugins import game    # noqa: E402


class Ev:
    def __init__(self, name, data):
        self.name, self.data = name, data


class Bus:
    def __init__(self):
        self.subscribers, self.events = [], []

    def subscribe(self, f, thread=False, subscribed_events=None):
        self.subscribers.append(f)

    def notify(self, name, data=None):
        self.events.append(name)
        for f in self.subscribers:
            f(Ev(name, data))

    def won(self):
        return "win_game" in self.events


def new_game():
    bus = Bus()
    game.Plugin(bus)
    bus.notify("set_game_mode", {"mode": 10, "timeout": None})
    bus.events.clear()
    return bus


def score(bus, y, b):
    bus.notify("score_changed", {"yellow": y, "black": b, "last_goal": None})


# A. the last point is scored, the analysis is pending: no victory while it is pending, even after the replay
bus = new_game()
score(bus, 10, 3)
bus.notify("goal_analysis_pending", {"team": "yellow"})
bus.notify("replay_start", {"type": "goal"})
bus.notify("replay_end")
time.sleep(0.8)
assert not bus.won(), "the victory must wait for the goal analysis"
# B. the analysis cancels the point (center goal): back to 9, so no victory at all
score(bus, 9, 3)
bus.notify("goal_analysis_done", {})
time.sleep(1.0)
assert not bus.won(), "no victory after the point was removed"

# C. the analysis confirms the goal: the victory comes shortly after
bus = new_game()
score(bus, 10, 3)
bus.notify("goal_analysis_pending", {"team": "yellow"})
bus.notify("replay_start", {"type": "goal"})
bus.notify("replay_end")
time.sleep(0.5)
assert not bus.won()
bus.notify("goal_analysis_done", {})
time.sleep(0.6)
assert bus.won()

# D. no analysis (feature off): the victory comes win_check_delay after the replay, and a manual removal during the
#    grace period still cancels it
bus = new_game()
score(bus, 10, 3)
bus.notify("replay_start", {"type": "goal"})
bus.notify("replay_end")
time.sleep(0.15)
score(bus, 9, 3)                 # the minus key, pressed during the grace period
time.sleep(0.8)
assert not bus.won(), "a manual removal during the grace period must cancel the victory"
bus = new_game()
score(bus, 10, 3)
bus.notify("replay_start", {"type": "goal"})
bus.notify("replay_end")
time.sleep(0.8)
assert bus.won()

# E. the analysis never answers: the hold expires after win_hold_max
bus = new_game()
score(bus, 10, 3)
bus.notify("goal_analysis_pending", {"team": "yellow"})
bus.notify("replay_start", {"type": "goal"})
bus.notify("replay_end")
time.sleep(1.2)
assert not bus.won()
time.sleep(0.8)
assert bus.won(), "the hold must expire"

print("all grace-period scenarios OK")
