"""Envoie au portable chaque fragment video termine, pour que le type de but soit calcule PENDANT le jeu.

La camera ecrit des fragments h264 de 0,2 s (10 images) dans <replay_path>/fragments (en RAM). Chaque fragment est
autonome. Ce plugin les envoie un par un a goal_service.py (POST /fragment?seq=N) avec un compteur croissant; au moment
d'un but, goal_analysis demande le type (POST /goal?last_seq=N) et la reponse arrive moins d'une seconde plus tard.

Pour ne pas occuper la carte graphique du portable pendant des heures, l'envoi n'est actif que pendant l'activite:
mouvement pres de la table, signal IR, ajustement du pointage. Il s'arrete apres goal_stream_idle_minutes sans activite.
A l'activation, les derniers fragments encore en memoire (goal_stream_lead) sont envoyes d'un coup pour que le premier but
ait son contexte.

Si le portable ne repond pas, rien ne change pour le jeu: les fragments sont abandonnes (jamais d'attente).

Configuration (config.py): utilise goal_service_url de goal_analysis; tout le reste est optionnel.
    goal_stream = True              # False = ne pas envoyer de fragments
    goal_stream_idle_minutes = 5
    goal_stream_lead = 20           # fragments (4 s) envoyes a l'activation
Activer avec plugins.add('goal_stream') (a cote de plugins.add('goal_analysis')).
"""
import collections
import http.client
import logging
import os
import threading
import time
import urllib.parse

import foos.config as config

logger = logging.getLogger(__name__)

INSTANCE = None          # lu par goal_analysis

ACTIVITY = ('movement_detected', 'goal_event', 'score_goal', 'increment_score', 'decrement_score', 'button_event')


class Plugin:
    def __init__(self, bus):
        global INSTANCE
        url = getattr(config, "goal_service_url", None)
        self.enabled = bool(url) and getattr(config, "goal_stream", True)
        self.frag_dir = os.path.join(config.replay_path, "fragments")
        self.idle_s = 60 * getattr(config, "goal_stream_idle_minutes", 5)
        self.lead = getattr(config, "goal_stream_lead", 20)
        # dernier fragment termine; compteur a nous (pas le nom du fichier), initialise sur l'heure pour ne jamais repartir de 0
        # apres un redemarrage de foos-master (le portable detecte aussi un recul et repart de zero)
        self.seq = int(time.time() * 5) - 1
        self.active_until = 0.0
        self.q = collections.deque(maxlen=40)
        self.cv = threading.Condition()
        self.sent = self.dropped = 0
        self.last_error = None
        INSTANCE = self
        if not self.enabled:
            logger.info("goal_stream disabled")
            return
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        bus.subscribe_map({ev: self.on_activity for ev in ACTIVITY}, thread=True)
        threading.Thread(target=self.watch, daemon=True).start()
        threading.Thread(target=self.sender, daemon=True).start()

    # -- etat -----------------------------------------------------------------------------------------------------
    def active(self):
        return self.enabled and time.time() < self.active_until

    def last_seq(self):
        return self.seq

    def on_activity(self, data=None):
        was = self.active()
        self.active_until = time.time() + self.idle_s
        if not was:
            logger.info("goal_stream: active")
            self.send_lead()

    def send_lead(self):
        """Fragments deja termines (les plus recents) a envoyer d'un coup au reveil."""
        try:
            with self.cv:
                recent = list(self.history)[-self.lead:]
                for seq, data in recent:
                    self.q.append((seq, data))
                self.cv.notify()
        except Exception:
            logger.exception("goal_stream: lead")

    # -- fragments ------------------------------------------------------------------------------------------------
    history = collections.deque(maxlen=25)      # derniers fragments lus, pour le reveil (25 x 0,4 Mo en memoire)

    def feed(self, path):
        """Un fragment est termine: le lire, lui donner son numero, l'envoyer si actif."""
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return
        with self.cv:
            self.seq += 1
            seq = self.seq
            self.history.append((seq, data))
            if self.active():
                if len(self.q) == self.q.maxlen:
                    self.dropped += 1
                self.q.append((seq, data))
                self.cv.notify()

    def watch(self):
        try:
            from inotify_simple import INotify, flags
            while not os.path.exists(self.frag_dir):
                time.sleep(10)
            inotify = INotify()
            inotify.add_watch(self.frag_dir, flags.CLOSE_WRITE)
            while True:
                for ev in inotify.read():
                    if ev.name.startswith("out") and ev.name.endswith(".h264"):
                        self.feed(os.path.join(self.frag_dir, ev.name))
        except Exception:
            logger.exception("goal_stream: watcher stopped")

    def sender(self):
        conn = None
        while True:
            with self.cv:
                while not self.q:
                    self.cv.wait()
                seq, data = self.q.popleft()
            try:
                if conn is None:
                    conn = http.client.HTTPConnection(self.host, self.port, timeout=3)
                conn.request("POST", "/fragment?seq=%d" % seq, body=data)
                conn.getresponse().read()
                self.sent += 1
                self.last_error = None
            except Exception as e:           # portable absent ou occupe: on abandonne ce fragment, sans jamais attendre
                if self.last_error is None:
                    logger.warning("goal_stream: send failed: %s", e)
                self.last_error = str(e)
                self.dropped += 1
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                time.sleep(0.5)
                with self.cv:
                    self.q.clear()           # on repart du present a la reconnexion
