"""Test the Pi plugins (goal_stream + goal_analysis) on the PC, with a fake bus and a recorded match.

    python test_pi_stream.py <match_folder> <foos_repo_folder> [first_goal] [number_of_goals] [port]

The fragments are fed at the camera's rhythm to goal_stream, the goals are signalled as the game would
(score_goal, then replay_start 1.2 s later) and the delay until the 'goal_type' message is measured.
"""
import json
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

match_dir, repo = Path(sys.argv[1]), sys.argv[2]
first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
count = int(sys.argv[4]) if len(sys.argv) > 4 else 2
port = int(sys.argv[5]) if len(sys.argv) > 5 else 8766

tmp = Path(tempfile.mkdtemp())
(tmp / "replay" / "fragments").mkdir(parents=True)
cfg = types.ModuleType("foos.config")
cfg.replay_path = str(tmp / "replay")
cfg.goal_service_url = "http://127.0.0.1:%d" % port
cfg.goal_clips_path = str(tmp / "goals")
cfg.video_fps = 49
pkg = types.ModuleType("foos")
pkg.config = cfg
sys.modules["foos"], sys.modules["foos.config"] = pkg, cfg
sys.path.insert(0, repo)


class Bus:
    def __init__(self):
        self.handlers = {}
        self.log = []

    def subscribe_map(self, fmap, thread=False):
        for k, f in fmap.items():
            self.handlers.setdefault(k, []).append(f)

    def notify(self, ev, data=None):
        self.log.append((time.time(), ev, data))
        for f in self.handlers.get(ev, []):
            threading.Thread(target=f, args=(data,), daemon=True).start()


from plugins import goal_analysis, goal_stream      # noqa: E402

bus = Bus()
gs = goal_stream.Plugin(bus)
ga = goal_analysis.Plugin(bus)


def read_jsonl(p):
    out = []
    for l in open(p, encoding="utf-8-sig", errors="replace"):
        l = l.strip().lstrip("\x00")
        if l:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


chunks, events = read_jsonl(match_dir / "chunks.jsonl"), read_jsonl(match_dir / "events.jsonl")
sizes = [c["bytes"] for c in chunks]
off = [0]
for s in sizes:
    off.append(off[-1] + s)
goals = [e for e in events if e["event"] == "score_goal"][first:first + count]
t_from, t_to = goals[0]["ts"] - 12, goals[-1]["ts"] + 8
idx = [i for i, c in enumerate(chunks) if t_from <= c["ts"] <= t_to and not c.get("lead")]
i0, i1 = idx[0], idx[-1]
base = chunks[i0]["mtime"]
ref = json.load(open(match_dir.parent.parent / "goals_analysis.json"))
journal = read_jsonl(match_dir.parent.parent / "goals" / "journal.jsonl")
gid = lambda ts: next((e["id"] for e in journal if e["event"] == "goal" and ts - 1 <= e["ts"] <= ts + 12), None)

t0 = time.time() + 1.0
sched = []
for ev in goals:
    sched.append((t0 + ev["ts"] - base, "goal", ev))
    sched.append((t0 + ev["ts"] - base + 1.2, "replay", ev))
sched.sort(key=lambda x: x[0])
sent_goal = {}
f = open(match_dir / "match.h264", "rb")
si = 0
bus.notify("movement_detected", {})            # people are moving: streaming becomes active
for i in range(i0, i1 + 1):
    due = t0 + chunks[i]["mtime"] - base
    while si < len(sched) and sched[si][0] <= due:
        when, kind, ev = sched[si]
        time.sleep(max(0, when - time.time()))
        if kind == "goal":
            sent_goal[ev["ts"]] = time.time()
            bus.notify("score_goal", {"team": ev["data"]["team"] if "data" in ev else "yellow"})
        else:
            gidx = gid(ev["ts"])
            src = match_dir.parent.parent / "goals" / ("%d.h264" % gidx) if gidx else None
            if src and src.exists():
                shutil.copyfile(src, tmp / "replay" / "replay_short.h264")
            bus.notify("replay_start", {"type": "goal", "team": "yellow"})
        si += 1
    time.sleep(max(0, due - time.time()))
    f.seek(off[i])
    p = tmp / "replay" / "fragments" / ("out%04d.h264" % (i % 100))
    p.write_bytes(f.read(sizes[i]))
    gs.feed(str(p))
time.sleep(6)
print("fragments sent: %d, dropped: %d, error: %s" % (gs.sent, gs.dropped, gs.last_error))
types_ = [(t, d) for t, ev, d in bus.log if ev == "goal_type"]
for ev in goals:
    g = gid(ev["ts"])
    t_goal = sent_goal.get(ev["ts"])
    # the first goal_type after this goal
    got = next(((t, d) for t, d in types_ if t_goal and t >= t_goal), None)
    print("goal %s: type=%s delay=%.2f s (clip: %s)  mode=%s" % (
        time.strftime("%H:%M:%S", time.localtime(ev["ts"])), got[1]["type"] if got else None,
        (got[0] - t_goal) if got else -1, ref.get("%d.h264" % g, {}).get("type") if g else None,
        got[1]["result"].get("mode") if got else None))
    if got:
        types_.remove(got)
j = [json.loads(l)["event"] for l in open(tmp / "goals" / "journal.jsonl", encoding="utf-8")]
print("Pi journal:", j)
