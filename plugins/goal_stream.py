"""Send every finished video fragment to the PC, so the goal type is computed WHILE the game is played.

The camera writes 0.2 s h264 fragments (10 frames) into <replay_path>/fragments (in RAM). Every fragment is
self-contained. This plugin sends them one by one to goal_service.py (POST /fragment?seq=N) with an increasing counter;
when a goal is scored, goal_analysis asks for the type (POST /goal?last_seq=N) and the answer arrives less than a second later.

To avoid keeping the PC's GPU busy for hours, sending is only active during activity: movement near the table, infrared
signal, score adjustment. It stops after goal_stream_idle_minutes without activity. When it wakes up, the latest fragments
still in memory (goal_stream_lead) are sent at once so the first goal has its context.

If the PC does not answer nothing changes for the game: fragments are dropped (never any waiting).

Configuration (config.py): uses goal_service_url from goal_analysis; everything else is optional.
    goal_stream = True              # False = do not send fragments
    goal_stream_idle_minutes = 5
    goal_stream_lead = 20           # fragments (4 s) sent on wake-up
Enable with plugins.add('goal_stream') (next to plugins.add('goal_analysis')).
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

INSTANCE = None          # read by goal_analysis

ACTIVITY = ('movement_detected', 'goal_event', 'score_goal', 'increment_score', 'decrement_score', 'button_event')


class Plugin:
    def __init__(self, bus):
        global INSTANCE
        url = getattr(config, "goal_service_url", None)
        self.enabled = bool(url) and getattr(config, "goal_stream", True)
        self.frag_dir = os.path.join(config.replay_path, "fragments")
        self.idle_s = 60 * getattr(config, "goal_stream_idle_minutes", 5)
        self.lead = getattr(config, "goal_stream_lead", 20)
        # last finished fragment; our own counter (not the file name), initialised from the clock so it never restarts at 0
        # after a foos-master restart (the PC also detects a step back and starts over)
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
        """Already finished fragments (the most recent ones) to send at once on wake-up."""
        try:
            with self.cv:
                recent = list(self.history)[-self.lead:]
                for seq, data in recent:
                    self.q.append((seq, data))
                self.cv.notify()
        except Exception:
            logger.exception("goal_stream: lead")

    # -- fragments ------------------------------------------------------------------------------------------------
    history = collections.deque(maxlen=25)      # last fragments read, for the wake-up (25 x 0.4 MB in memory)

    def feed(self, path):
        """A fragment is finished: read it, give it its number, send it if active."""
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
            except Exception as e:           # PC absent or busy: drop this fragment, never wait
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
                    self.q.clear()           # start again from the present on reconnection
