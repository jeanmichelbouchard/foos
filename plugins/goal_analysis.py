"""Analyse du type de but (service sur le portable) et rétroaction des joueurs par le clavier.

Au 'replay_start' d'un but, le clip court (replay_short.h264) est copié, envoyé à config.goal_service_url
(tools/ball/goal_service.py) et le type trouvé est publié sur le bus ('goal_type'). Si le service ne répond pas
à temps, on ne fait rien: foos-master se comporte comme avant.

Les touches de type (événements 'goal_type_key' de io_evdev_keyboard) corrigent le DERNIER but. Tout est écrit dans
un journal append-only (une ligne JSON par événement): but, prédiction, correction, annulation. Ce journal sert
au réentraînement et à repérer les buts « à revoir » (ambiguous).

Configuration (config.py), tout est optionnel:
    goal_service_url = "http://192.168.1.50:8765"   # absent ou None = analyse désactivée (le journal reste actif)
    goal_service_timeout = 20                      # secondes
    goal_clips_path = "./goals"                   # copies des clips (sur la carte SD, PAS dans replay_path qui est en RAM)
"""
import json
import logging
import os
import shutil
import threading
import time
import urllib.request

import foos.config as config

logger = logging.getLogger(__name__)

# Ordre et libellés de l'écran de statistiques (les types à zéro ne sont pas affichés)
TYPE_LABELS = {"goalie": "Gardien", "defense": "Défense", "center": "Centre", "attack": "Attaque",
               "slice": "Pissette", "wall_shot": "Main courante", "chevreuil": "Chevreuil",
               "one_timer": "One-timer", "own_goal": "Auto-buts", "unknown": "Non classés"}


class Plugin:
    def __init__(self, bus):
        self.bus = bus
        self.url = getattr(config, "goal_service_url", None)
        self.timeout = getattr(config, "goal_service_timeout", 20)
        self.clips_dir = getattr(config, "goal_clips_path", "./goals")
        self.journal = os.path.join(self.clips_dir, "journal.jsonl")
        os.makedirs(self.clips_dir, exist_ok=True)
        self.last = None        # dernier but: dict
        self.history = []       # corrections du dernier but, pour 'undo'
        self.match = []         # buts du match en cours (pour les statistiques de fin de match)
        self.pending = []       # analyses en cours (une par but, chacune dans son propre thread)
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

    # -- but détecté ----------------------------------------------------------------------------------------------
    def _stream(self):
        """Le plugin goal_stream, s'il envoie des fragments en ce moment (analyse en continu)."""
        try:
            from plugins import goal_stream
            gs = goal_stream.INSTANCE
            return gs if gs is not None and gs.active() and self.url else None
        except Exception:
            return None

    def _new_goal(self, team, clip=None, info=None):
        gid = int(time.time())
        self.last = {"id": gid, "clip": clip, "label": None, "source": None, "predicted": None, "team": team,
                     "created": time.time(), "stream_seq": None, "stream_failed": False}
        self.match.append(self.last)
        self.history = []
        self.log(event="goal", id=gid, clip=clip, info=info or {})
        return self.last

    def _start(self, target, goal):
        # son propre thread: l'analyse dure plusieurs secondes et ne doit pas bloquer les touches ni la fin de match
        t = threading.Thread(target=target, args=(goal,), daemon=True)
        self.pending.append(t)
        t.start()

    def on_score_goal(self, data):
        """But compté: si le portable suit la balle en continu, on demande le type tout de suite (moins d'une seconde)."""
        gs = self._stream()
        if gs is None:
            return          # sinon on attend le replay et l'analyse du clip (on_replay_start)
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
                self._start(self.analyze, g)        # le suivi en continu a echoue: analyse du clip en secours
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
            if goal["clip"] and self.url:           # le clip est deja la: analyse de secours
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
            return
        self._apply(goal, res, t0)

    def _apply(self, goal, res, t0):
        goal["predicted"] = res.get("type", "unknown")
        self.log(event="prediction", id=goal["id"], result=res, round_trip=round(time.time() - t0, 2))
        if goal["source"] == "human":
            return          # un joueur a déjà tranché: la prédiction ne remplace pas son choix
        goal["label"], goal["source"] = goal["predicted"], "model"
        self.bus.notify('goal_type', {'type': goal["label"], 'source': 'model', 'id': goal["id"],
                                      'team': goal["team"], 'confidence': res.get("confidence"), 'result': res})

    # -- rétroaction clavier --------------------------------------------------------------------------------------
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
            self.bus.notify('goal_type', {'type': goal["label"], 'source': goal["source"], 'id': goal["id"],
                                          'team': goal["team"]})

    # -- statistiques de fin de match -----------------------------------------------------------------------------
    def on_decrement(self, data):
        """Un but retiré à la main (faux déclenchement IR?): on le sort des statistiques et on le signale pour revue."""
        team = data.get('team')
        for g in reversed(self.match):
            if g["team"] == team:
                self.match.remove(g)
                self.log(event="goal_removed", id=g["id"], team=team, label=g["label"], review=True)
                if g is self.last:
                    self.last = None
                return

    def on_win(self, data):
        for t in list(self.pending):      # attendre les analyses encore en cours pour que les statistiques soient completes
            t.join(timeout=25)
        self.pending = [t for t in self.pending if t.is_alive()]
        lines = {t: self.stats_lines(t) for t in ('yellow', 'black')}
        self.log(event="match_end", team=data.get('team'), score={t: data.get(t) for t in ('yellow', 'black')},
                 goals=[{"id": g["id"], "team": g["team"], "label": g["label"], "source": g["source"]} for g in self.match])
        self.bus.notify('match_stats', {'lines': lines, 'team': data.get('team'),
                                        'yellow': data.get('yellow'), 'black': data.get('black')})

    def stats_lines(self, team):
        goals = [g for g in self.match if g["team"] == team]
        counts = {}
        for g in goals:
            k = g["label"] if g["label"] in TYPE_LABELS else "unknown"
            counts[k] = counts.get(k, 0) + 1
        lines = ["{:<16}{:>3}".format("Buts", len(goals))]
        lines += ["{:<16}{:>3}".format(TYPE_LABELS[k], counts[k]) for k in TYPE_LABELS if counts.get(k)]
        return lines
