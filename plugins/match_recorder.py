"""Enregistrement d'une partie complete sur la carte SD (pour tester l'analyse des buts sur de vrais matchs).

La camera ecrit des fragments h264 (out%04d.h264) dans <replay_path>/fragments, en RAM, avec un anneau de 100 fichiers.
Ce plugin copie chaque fragment termine (inotify) a la suite d'un seul fichier <dossier>/<date>/match.h264.
Il note aussi, avec l'heure, chaque fragment (chunks.jsonl) et chaque evenement du jeu (events.jsonl): signal IR brut
(goal_event, meme s'il est ignore), but compte, ajout/retrait manuel, fin de match. Cela permet de retrouver dans la
video les buts, les faux declenchements et les buts manques.

Demarrage: au premier signal IR / but / ajustement de la partie. Arret: fin de match (win_game), remise a zero,
inactivite, duree maximale ou manque d'espace disque.

Configuration (config.py), tout est optionnel; activer avec plugins.add('match_recorder'):
    match_record_path = "./matches"
    match_record_max_minutes = 40        # arret de securite
    match_record_idle_minutes = 6        # arret apres ce temps sans evenement
    match_record_min_free_gb = 3         # arret si l'espace libre passe sous ce seuil
    match_record_lead_chunks = 15        # fragments deja en memoire copies au demarrage
"""
import json
import logging
import os
import shutil
import threading
import time

import foos.config as config

logger = logging.getLogger(__name__)

EVENTS = ('goal_event', 'score_goal', 'increment_score', 'decrement_score', 'score_reset', 'win_game')
START_EVENTS = ('goal_event', 'score_goal', 'increment_score', 'decrement_score')


class Plugin:
    def __init__(self, bus):
        self.bus = bus
        self.root = getattr(config, "match_record_path", "./matches")
        self.max_s = 60 * getattr(config, "match_record_max_minutes", 40)
        self.idle_s = 60 * getattr(config, "match_record_idle_minutes", 6)
        self.min_free = getattr(config, "match_record_min_free_gb", 3) * 1024 ** 3
        self.lead = getattr(config, "match_record_lead_chunks", 15)
        self.frag_dir = os.path.join(config.replay_path, "fragments")
        self.lock = threading.RLock()
        self.dir = None
        self.video = None
        self.t0 = self.last_event = 0
        os.makedirs(self.root, exist_ok=True)
        bus.subscribe_map({ev: (lambda d, ev=ev: self.on_event(ev, d)) for ev in EVENTS}, thread=True)
        threading.Thread(target=self.watch, daemon=True).start()

    # -- evenements du jeu ----------------------------------------------------------------------------------------
    def on_event(self, name, data):
        with self.lock:
            if self.video is None and name in START_EVENTS:
                self.start()
            if self.video is None:
                return
            self.last_event = time.time()
            self.append("events.jsonl", {"event": name, "data": {k: v for k, v in (data or {}).items()
                                                                if isinstance(v, (int, float, str, bool, type(None)))}})
            if name in ('win_game', 'score_reset'):
                self.stop(name)

    def start(self):
        free = shutil.disk_usage(self.root).free
        if free < self.min_free:
            logger.warning("Match recorder: not enough free space (%.1f GB)", free / 1024 ** 3)
            return
        self.dir = os.path.join(self.root, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.video = open(os.path.join(self.dir, "match.h264"), "ab")
        self.t0 = self.last_event = time.time()
        logger.info("Match recorder: recording to %s", self.dir)
        try:    # fragments deja termines (les dernieres secondes avant le premier evenement)
            frags = sorted((f for f in os.listdir(self.frag_dir) if f.startswith("out") and f.endswith(".h264")),
                           key=lambda f: os.path.getmtime(os.path.join(self.frag_dir, f)))
            for f in frags[-(self.lead + 1):-1]:     # le dernier est peut-etre en cours d'ecriture
                self.copy(f, lead=True)
        except OSError:
            logger.exception("Match recorder: cannot copy lead fragments")

    def stop(self, why):
        if self.video is None:
            return
        self.append("events.jsonl", {"event": "recording_stopped", "why": why})
        self.video.close()
        self.video = None
        logger.info("Match recorder: stopped (%s)", why)

    # -- fragments ------------------------------------------------------------------------------------------------
    def copy(self, name, lead=False):
        src = os.path.join(self.frag_dir, name)
        with open(src, "rb") as f:
            data = f.read()
        self.video.write(data)
        self.video.flush()
        self.append("chunks.jsonl", {"chunk": name, "bytes": len(data), "lead": lead,
                                     "mtime": round(os.path.getmtime(src), 2)})

    def append(self, fname, entry):
        entry["ts"] = round(time.time(), 2)
        with open(os.path.join(self.dir, fname), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def watch(self):
        try:
            from inotify_simple import INotify, flags
            while not os.path.exists(self.frag_dir):
                time.sleep(10)
            inotify = INotify()
            inotify.add_watch(self.frag_dir, flags.CLOSE_WRITE)
            while True:
                for ev in inotify.read():
                    if not (ev.name.startswith("out") and ev.name.endswith(".h264")):
                        continue
                    with self.lock:
                        if self.video is None:
                            continue
                        try:
                            self.copy(ev.name)
                        except OSError:
                            logger.exception("Match recorder: copy failed")
                        self.check_limits()
        except Exception:
            logger.exception("Match recorder stopped")

    def check_limits(self):
        now = time.time()
        if now - self.t0 > self.max_s:
            self.stop("max_duration")
        elif now - self.last_event > self.idle_s:
            self.stop("idle")
        elif shutil.disk_usage(self.root).free < self.min_free:
            self.stop("disk_full")
