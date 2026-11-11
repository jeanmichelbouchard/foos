"""Shared constants and utilities for the ball-annotation and training tools.

Paths are taken from environment variables so nothing depends on one person's disk layout:

    FOOS_VIDEOS   folder with the recorded goal clips, organised by rod (see README.md)       default ~/foos-videos
    FOOS_ML_DATA  working folder for generated data (frames, annotations, tracks, models)    default ~/foos-ml
    FOOS_LABELS   folder with the clip tag lists (autogoal.txt, ...)                         default ./labels

Put the generated data outside any cloud-synchronised folder (it can be several GB).
"""
import os
import sys
from pathlib import Path

# The goal rules and the tracking code are shared with the service (goal_service/): one copy only.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "goal_service"))

VIDEO_ROOT = Path(os.environ.get("FOOS_VIDEOS", Path.home() / "foos-videos"))
LABELS_DIR = Path(os.environ.get("FOOS_LABELS", Path(__file__).resolve().parent / "labels"))
DATA_ROOT = Path(os.environ.get("FOOS_ML_DATA", Path.home() / "foos-ml"))
FRAMES_DIR = DATA_ROOT / "frames"
ANNOT_CSV = DATA_ROOT / "ball_annotations.csv"
SAMPLES_CSV = DATA_ROOT / "samples.csv"

# Folder -> rod class
ROD_BY_FOLDER = {"1_rod": "goalie", "2_rod": "defense", "3_rod": "attack", "5_rod": "midfield"}


def read_tag_file(name):
    """Read labels/<name>.txt: one entry per line, 'file' or 'file : label'."""
    path = LABELS_DIR / name
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().lstrip("\ufeff")
            if line:
                out.add(line.split(":")[0].strip())
    return out


def list_clips():
    """Return the list of clips (dicts) with rod class, sub-type, session and ball condition."""
    corkball = read_tag_file("corkball.txt")
    redball = read_tag_file("redball.txt")
    autogoal = read_tag_file("autogoal.txt")
    clips = []
    for path in sorted(VIDEO_ROOT.rglob("*.h264")):
        rel = path.relative_to(VIDEO_ROOT).parts
        rod = ROD_BY_FOLDER.get(rel[0], rel[0])
        subtype = rel[1] if len(rel) > 2 else ""
        ts = int(path.name.split("_")[0])
        ball = "cork" if path.name in corkball else "red" if path.name in redball else "white"
        clips.append({
            "path": str(path), "name": path.name, "ts": ts, "rod": rod, "subtype": subtype,
            "kind": "long" if "_long" in path.name else "short",
            "autogoal": int(path.name in autogoal), "ball": ball,
        })
    # Session = series of clips less than 2 h apart
    prev, sess = None, -1
    for c in sorted(clips, key=lambda c: c["ts"]):
        if prev is None or c["ts"] - prev > 7200:
            sess += 1
        c["session"] = sess
        prev = c["ts"]
    return clips
