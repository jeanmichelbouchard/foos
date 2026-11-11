"""Goal-type analysis (service running on a PC) and player feedback through the keyboard.

When a goal is counted ('score_goal') and the PC is following the ball live (plugin goal_stream), the goal type is
requested immediately (POST /goal, about a second) and published on the bus ('goal_type'). Otherwise, at 'replay_start',
the short clip (replay_short.h264) is copied, sent to config.goal_service_url (goal_service/goal_service.py) and the type
found is published the same way. If the service does not answer in time nothing happens: foos-master behaves as before.

The type keys ('goal_type_key' events from io_evdev_keyboard) correct the LAST goal. Everything is written to an
append-only journal (one JSON line per event): goal, prediction, correction, undo, removed goals, end of match. The journal is used to
retrain the model and to find the goals "to review".

Configuration (config.py), everything is optional:
    goal_service_url = "http://192.168.1.50:8765"   # missing or None = analysis disabled (the journal stays active)
    goal_service_timeout = 20                      # seconds
    goal_clips_path = "./goals"                    # clip copies (on the SD card, NOT in replay_path which is in RAM)
    goal_type_labels = {"slice": "Pissette", ...}  # texts shown to the players (see DEFAULT_LABELS)
    center_goals_auto_cancel = False               # True = a center-rod goal does not count (see below)
    center_cancel_confidence = 0.90                # minimum model confidence to cancel automatically

Center goals: in many house rules a goal scored by the center (5-bar) rod does not count, unless it is an own goal.
With center_goals_auto_cancel the point is removed (decrement_score) when the model says "center" with a high
confidence, or when a player presses the center key. The goal is announced as "center_void" and kept in the journal.
The own-goal key (or any other type key) gives the point back (increment_score): the model cannot tell an own goal
from a center goal, so the players decide. A cancellation is skipped when the match has just been won.
"""
import collections
import json
import logging
import os
import shutil
import threading
import time
import urllib.request

import foos.config as config

logger = logging.getLogger(__name__)

# Order and texts of the end-of-match statistics screen (types with a zero count are not shown). The texts can be
# overridden in config.py with goal_type_labels (also used by the on-screen banner: "own_goal_one" is the singular form).
DEFAULT_LABELS = {"goals": "Goals", "goalie": "Goalie", "defense": "Defense", "center": "Center", "attack": "Attack",
                  "slice": "Slice", "wall_shot": "Wall shot", "chevreuil": "Chevreuil", "one_timer": "One-timer",
                  "own_goal": "Own goals", "own_goal_one": "Own goal", "unknown": "Unclassified",
                  "center_void": "Center (void)", "center_void_one": "Center goal void"}
STAT_TYPES = ["goalie", "defense", "center", "center_void", "attack", "slice", "wall_shot", "chevreuil", "one_timer", "own_goal", "unknown"]
LABELS = dict(DEFAULT_LABELS)
LABELS.update(getattr(config, "goal_type_labels", {}))
TYPE_LABELS = collections.OrderedDict((k, LABELS[k]) for k in STAT_TYPES)


class Plugin:
    def __init__(self, bus):
        self.bus = bus
        self.url = getattr(config, "goal_service_url", None)
        self.timeout = getattr(config, "goal_service_timeout", 20)
        self.clips_dir = getattr(config, "goal_clips_path", "./goals")
        self.journal = os.path.join(self.clips_dir, "journal.jsonl")
        os.makedirs(self.clips_dir, exist_ok=True)
        self.last = None        # dernier but: dict
        self.history = []       # corrections of the last goal, for 'undo'
        self.match = []         # goals of the current match (for the end-of-match statistics)
        self.pending = []       # running analyses (one per goal, each in its own thread)
        self.cancel_enabled = getattr(config, "center_goals_auto_cancel", False)
        self.cancel_conf = getattr(config, "center_cancel_confidence", 0.90)
        self.lock = threading.RLock()
        self.last_win = 0.0
        bus.subscribe_map({'replay_start': self.on_replay_start,
                           'score_goal': self.on_score_goal,
                           'goal_type_key': self.on_key,
                           'decrement_score': self.on_decrement,
                           'win_game': self.on_win,
                           'score_reset': lambda d: self.match.clear()}, thread=True)

    # -- journal --------------------------------------------------------------------------------------------------
    def log(self, **entry):
        entry["ts"] = round(time.time(), 2)
        with open(self.journal, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- goal detected ----------------------------------------------------------------------------------------------
    def _stream(self):
        """The goal_stream plugin, if it is sending fragments right now (live analysis)."""
        try:
            from plugins import goal_stream
            gs = goal_stream.INSTANCE
            return gs if gs is not None and gs.active() and self.url else None
        except Exception:
            return None

    def _new_goal(self, team, clip=None, info=None):
        gid = int(time.time())
        self.last = {"id": gid, "clip": clip, "label": None, "source": None, "predicted": None, "team": team,
                     "created": time.time(), "stream_seq": None, "stream_failed": False,
                     "confidence": None, "cancelled": False, "review": False}
        self.match.append(self.last)
        self.history = []
        self.log(event="goal", id=gid, clip=clip, info=info or {})
        return self.last

    def _start(self, target, goal):
        # own thread: the analysis can take seconds and must not block the keys or the end of the match
        t = threading.Thread(target=target, args=(goal,), daemon=True)
        self.pending.append(t)
        t.start()

    def on_score_goal(self, data):
        """Goal counted: if the PC follows the ball live, ask for the type right away (under a second)."""
        if self.url and self.cancel_enabled:
            # the game plugin will not declare a victory until the analysis is done (a center goal may be cancelled)
            self.bus.notify('goal_analysis_pending', {'team': data.get("team")})
        gs = self._stream()
        if gs is None:
            return          # otherwise wait for the replay and analyse the clip (on_replay_start)
        goal = self._new_goal(data.get("team"), info={k: v for k, v in data.items() if k != 'type'})
        goal["stream_seq"] = gs.last_seq()
        self._start(self.analyze_stream, goal)

    def on_replay_start(self, extra):
        if not extra or extra.get('type') != 'goal':
            return
        src = os.path.join(config.replay_path, "replay_short.h264")
        g = self.last
        streamed = g is not None and g["stream_seq"] is not None and g["clip"] is None and time.time() - g["created"] < 30
        clip = os.path.join(self.clips_dir, "%d.h264" % (g["id"] if streamed else int(time.time())))
        try:
            shutil.copyfile(src, clip)
        except OSError:
            logger.exception("Cannot copy replay clip")
            clip = None
        if streamed:
            g["clip"] = clip
            self.log(event="clip", id=g["id"], clip=clip)
            if g["stream_failed"] and self.url and clip:
                self._start(self.analyze, g)        # live tracking failed: fall back to analysing the clip
            return
        goal = self._new_goal(extra.get("team"), clip=clip, info={k: v for k, v in extra.items() if k != 'type'})
        if self.url and clip:
            self._start(self.analyze, goal)

    def analyze_stream(self, goal):
        t0 = time.time()
        try:
            req = urllib.request.Request(self.url.rstrip("/") + "/goal?last_seq=%d" % goal["stream_seq"], data=b"", method="POST")
            res = json.loads(urllib.request.urlopen(req, timeout=min(self.timeout, 8)).read().decode("utf-8"))
            if res.get("error"):
                raise RuntimeError(res["error"])
        except Exception as e:
            logger.warning("Goal stream analysis unavailable: %s", e)
            self.log(event="analysis_failed", id=goal["id"], error=str(e), mode="stream")
            goal["stream_failed"] = True
            if goal["clip"] and self.url:           # the clip is already there: fall back to analysing it
                self.analyze(goal)
            return
        self._apply(goal, res, t0)

    def analyze(self, goal):
        t0 = time.time()
        try:
            with open(goal["clip"], "rb") as f:
                fps = getattr(config, "video_fps", 49)   # the model is calibrated at 49 fps: the service resamples
                req = urllib.request.Request(self.url.rstrip("/") + "/analyze?fps=%s" % fps, data=f.read(), method="POST")
            res = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read().decode("utf-8"))   # json.load(bytes) fails on Python 3.5 (Pi)
        except Exception as e:
            logger.warning("Goal analysis unavailable: %s", e)
            self.log(event="analysis_failed", id=goal["id"], error=str(e))
            self._release_hold()
            return
        self._apply(goal, res, t0)

    def _release_hold(self):
        if self.url and self.cancel_enabled:
            self.bus.notify('goal_analysis_done', {})

    def _apply(self, goal, res, t0):
        try:
            self._apply_result(goal, res, t0)
        finally:
            self._release_hold()

    def _apply_result(self, goal, res, t0):
        goal["predicted"] = res.get("type", "unknown")
        self.log(event="prediction", id=goal["id"], result=res, round_trip=round(time.time() - t0, 2))
        if goal["source"] == "human":
            return          # a player has already decided: the prediction does not replace their choice
        goal["label"], goal["source"], goal["confidence"] = goal["predicted"], "model", res.get("confidence")
        with self.lock:
            self._sync_score(goal)
        self._announce(goal, confidence=res.get("confidence"), result=res)

    # -- center goals -------------------------------------------------------------------------------------------
    def _is_void(self, goal):
        """True if this goal must not count: a center goal, decided by a player or by the model with high confidence."""
        if not self.cancel_enabled or goal["label"] != "center":
            return False
        return goal["source"] == "human" or (goal["confidence"] or 0) >= self.cancel_conf

    def _sync_score(self, goal):
        """Make the score agree with the goal's current type: remove the point of a void goal, give it back otherwise."""
        want = self._is_void(goal)
        if want and not goal["cancelled"]:
            if time.time() - self.last_win < 30 and goal["created"] < self.last_win:
                goal["review"] = True       # the match was just won: do not touch the score, flag it for review
                self.log(event="center_cancel_skipped", id=goal["id"], team=goal["team"], review=True)
                return
            goal["cancelled"] = True
            self.bus.notify('decrement_score', {'team': goal["team"], 'source': 'center_cancel'})
            self.log(event="center_cancelled", id=goal["id"], team=goal["team"], confidence=goal["confidence"],
                     by=goal["source"])
        elif not want and goal["cancelled"]:
            goal["cancelled"] = False
            self.bus.notify('increment_score', {'team': goal["team"], 'source': 'center_restore'})
            self.log(event="center_restored", id=goal["id"], team=goal["team"], label=goal["label"], by=goal["source"])

    def _announce(self, goal, **extra):
        """Tell the rest of the game (banner, sounds) what this goal is."""
        data = {'type': "center_void" if goal["cancelled"] else goal["label"], 'source': goal["source"],
                'id': goal["id"], 'team': goal["team"]}
        data.update(extra)
        self.bus.notify('goal_type', data)

    # -- keyboard feedback --------------------------------------------------------------------------------------
    def on_key(self, data):
        goal, key = self.last, data['type']
        if goal is None:
            return
        if key == 'undo':
            if not self.history:
                return
            self.history.pop()
            goal["label"], goal["source"] = (self.history[-1], "human") if self.history else (goal["predicted"], "model" if goal["predicted"] else None)
            self.log(event="undo", id=goal["id"], label=goal["label"])
        else:
            self.history.append(key)
            goal["label"], goal["source"] = key, "human"
            self.log(event="correction", id=goal["id"], label=key, predicted=goal["predicted"])
        if goal["label"]:
            with self.lock:
                self._sync_score(goal)
            self._announce(goal)

    # -- statistiques de fin de match -----------------------------------------------------------------------------
    def on_decrement(self, data):
        """A goal removed by hand (false IR trigger? center goal?): take it out of the statistics and flag it for review."""
        if data.get('source') == 'center_cancel':
            return          # our own cancellation of a center goal, not a manual removal
        team = data.get('team')
        for g in reversed(self.match):
            if g["team"] == team and not g["cancelled"]:
                self.match.remove(g)
                self.log(event="goal_removed", id=g["id"], team=team, label=g["label"], review=True)
                if g is self.last:
                    self.last = None
                return

    def on_win(self, data):
        self.last_win = time.time()
        for t in list(self.pending):      # wait for the analyses still running so that the statistics are complete
            t.join(timeout=25)
        self.pending = [t for t in self.pending if t.is_alive()]
        lines = {t: self.stats_lines(t) for t in ('yellow', 'black')}
        self.log(event="match_end", team=data.get('team'), score={t: data.get(t) for t in ('yellow', 'black')},
                 goals=[{"id": g["id"], "team": g["team"], "label": g["label"], "source": g["source"],
                         "cancelled": g["cancelled"]} for g in self.match])
        self.bus.notify('match_stats', {'lines': lines, 'team': data.get('team'),
                                        'yellow': data.get('yellow'), 'black': data.get('black')})

    def stats_lines(self, team):
        goals = [g for g in self.match if g["team"] == team]
        counts = {}
        for g in goals:
            k = "center_void" if g["cancelled"] else g["label"] if g["label"] in TYPE_LABELS else "unknown"
            counts[k] = counts.get(k, 0) + 1
        lines = ["{:<16}{:>3}".format(LABELS["goals"], len([g for g in goals if not g["cancelled"]]))]
        lines += ["{:<16}{:>3}".format(TYPE_LABELS[k], counts[k]) for k in TYPE_LABELS if counts.get(k)]
        return lines
