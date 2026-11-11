"""Analyse en continu: le Pi envoie chaque fragment vidéo (10 images, 0,2 s) dès qu'il est terminé; le portable suit la
balle pendant que l'action se déroule. Quand un but est signalé, il reste seulement à classer les dernières images.

    s = StreamAnalyzer(analyzer, gpu_lock)
    s.add_fragment(seq, octets_h264)          # appelé à chaque fragment
    s.analyze_goal(last_seq)                  # -> même dictionnaire que Analyzer.analyze + "mode": "stream"

`seq` est un compteur croissant fourni par le Pi (les noms de fichiers de la caméra tournent sur 100 valeurs).
`last_seq` = le dernier fragment TERMINÉ au moment du signal IR; le but est dans le fragment suivant (en cours d'écriture),
on attend donc quelques fragments de plus pour voir la balle absente (>= 15 images) après sa disparition.

Chaque fragment est autonome (en-têtes + image clé): un fragment perdu ne gâche que ses 10 images.
Les réglages sont calibrés à 49 images/s (config.video_fps du Pi).
"""
import collections
import logging
import queue
import threading
import time

import av
import numpy as np
import torch
import torchvision

logger = logging.getLogger("stream")

W, H = 1280, 720
PAD_H = 736          # 720 arrondi au multiple de 32, comme le letterbox de ultralytics
CONF = 0.10
IOU = 0.7
TOP_K = 5
WINDOW = 200         # images gardées pour classer un but: la même longueur que les clips d'entraînement (200 images, ~4 s)
EXTRA = 3            # fragments à attendre après celui du signal IR


class FastDetector:
    """Détection de la balle en préparant les images sur la carte graphique (le prétraitement de ultralytics, sur le
    processeur, coûtait environ 5 ms par image). Même résultat que model.predict à 1280 px (vérifié par le banc d'essai)."""

    def __init__(self, yolo):
        self.yolo = yolo
        self.net = yolo.model.to("cuda").eval().float()
        try:
            self.net = self.net.fuse(verbose=False)
        except Exception:
            logger.warning("fuse indisponible")
        self.pad = None

    def detect(self, frames):
        """frames: liste d'images BGR uint8 de 720 x 1280 -> tableau B x TOP_K x 4 (détectée, x, y, confiance)."""
        b = len(frames)
        out = np.zeros((b, TOP_K, 4), np.float32)
        if any(f.shape[:2] != (H, W) for f in frames):      # autre format: chemin général (plus lent)
            for n, f in enumerate(frames):
                r = self.yolo.predict(f, imgsz=1280, conf=CONF, verbose=False, device=0)[0]
                if len(r.boxes):
                    order = r.boxes.conf.argsort(descending=True)[:TOP_K]
                    for j, i in enumerate(order.tolist()):
                        x, y = r.boxes.xywh[i][:2].tolist()
                        out[n, j] = (1, x, y, float(r.boxes.conf[i]))
            return out
        x = torch.from_numpy(np.stack(frames)).to("cuda", non_blocking=True)
        x = x[..., [2, 1, 0]].permute(0, 3, 1, 2).float().div_(255.0)
        pad = torch.full((b, 3, PAD_H, W), 114 / 255.0, device="cuda")
        pad[:, :, (PAD_H - H) // 2:(PAD_H - H) // 2 + H, :] = x
        with torch.no_grad():
            y = self.net(pad)[0]                              # B x 5 x N : cx, cy, w, h, confiance
        off = (PAD_H - H) // 2
        d = y.permute(0, 2, 1)                                # B x N x 5
        mask = d[..., 4] >= CONF
        fb = mask.nonzero()[:, 0]                             # image de chaque candidat
        d = d[mask]
        if len(d) == 0:
            return out
        boxes = torch.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2,
                             d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2], 1)
        keep = torchvision.ops.batched_nms(boxes, d[:, 4], fb, IOU)      # triés par confiance décroissante
        k, fk = d[keep].cpu().numpy(), fb[keep].cpu().numpy()            # un seul transfert
        n = np.zeros(b, np.int32)
        for row, i in zip(k, fk):
            if n[i] < TOP_K:
                out[i, n[i]] = (1, row[0], row[1] - off, row[4])
                n[i] += 1
        return out


class StreamAnalyzer:
    def __init__(self, analyzer, gpu_lock, queue_max=40):
        self.analyzer = analyzer
        self.lock = gpu_lock
        self.det = FastDetector(analyzer.yolo)
        self.det.detect([np.zeros((H, W, 3), np.uint8)] * 2)        # préchauffage
        self.in_q = collections.deque()
        self.in_cv = threading.Condition()
        self.queue_max = queue_max
        self.frames_q = queue.Queue(maxsize=3)
        self.raw = collections.deque(maxlen=800)                    # (seq, tableau K x 4) par image
        self.cond = threading.Condition()
        self.done_seq = -1
        self.stats = {"fragments": 0, "dropped": 0, "frames": 0, "detect_ms": 0.0, "last_ms": 0.0}
        self.ctx = None
        self.arrived = {}          # seq -> heure de réception
        self.session = 0           # change quand le Pi redémarre (ses numéros de fragment repartent de zéro)
        self.last_added = None
        self.lags = collections.deque(maxlen=2000)   # secondes entre réception et fin de détection
        self.lag_parts = collections.deque(maxlen=2000)
        threading.Thread(target=self._decode_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    # -- entrée ---------------------------------------------------------------------------------------------------
    def add_fragment(self, seq, data):
        with self.in_cv:
            if self.last_added is not None and seq < self.last_added - 5:
                # numéro en recul: le Pi a redémarré. On repart de zéro, sinon les anciennes images se mêleraient aux nouvelles.
                logger.warning("nouvelle session: fragment %s après %s", seq, self.last_added)
                self.session += 1
                self.in_q.clear()
                self.arrived.clear()
                with self.cond:
                    self.raw.clear()
                    self.done_seq = -1
            self.last_added = seq
            self.arrived[seq] = time.time()
            self.in_q.append((seq, data, self.session))
            while len(self.in_q) > self.queue_max:                  # en retard de plus de ~8 s: on abandonne les plus vieux
                self.in_q.popleft()
                self.stats["dropped"] += 1
            self.in_cv.notify()

    def _decode_loop(self):
        while True:
            with self.in_cv:
                while not self.in_q:
                    self.in_cv.wait()
                seq, data, sess = self.in_q.popleft()
            try:
                if self.ctx is None:
                    self.ctx = av.CodecContext.create("h264", "r")
                frames = []
                for p in self.ctx.parse(data):
                    frames += self.ctx.decode(p)
                for p in self.ctx.parse(None):
                    frames += self.ctx.decode(p)
                imgs = [f.to_ndarray(format="bgr24") for f in frames]
            except Exception:
                logger.exception("décodage du fragment %s", seq)
                self.ctx = None
                imgs = []
            self.frames_q.put((seq, imgs, time.time(), sess))

    def _detect_loop(self):
        while True:
            seq, imgs, t_dec, sess = self.frames_q.get()
            if sess != self.session:
                continue                       # fragment d'une session précédente
            t = time.time()
            res = np.zeros((0, TOP_K, 4), np.float32)
            if imgs:
                try:
                    with self.lock:
                        res = self.det.detect(imgs)
                except Exception:
                    logger.exception("détection du fragment %s", seq)
            dt = 1000 * (time.time() - t)
            with self.cond:
                if sess != self.session:
                    continue
                for r in res:
                    self.raw.append((seq, r))
                self.done_seq = max(self.done_seq, seq)
                self.stats["fragments"] += 1
                self.stats["frames"] += len(res)
                self.stats["detect_ms"] += dt
                self.stats["last_ms"] = dt
                t_arr = self.arrived.pop(seq, t)
                self.lags.append(time.time() - t_arr)
                self.lag_parts.append((t_dec - t_arr, t - t_dec, dt / 1000))
                self.cond.notify_all()

    # -- but ------------------------------------------------------------------------------------------------------
    def analyze_goal(self, last_seq, extra=EXTRA, timeout=4.0):
        t0 = time.time()
        target = last_seq + extra
        with self.cond:
            while self.done_seq < target and time.time() - t0 < timeout:
                self.cond.wait(timeout=timeout - (time.time() - t0))
            waited = time.time() - t0
            rows = []
            prev = None
            for s_, r in reversed(self.raw):         # de la fin vers le début, en s'arrêtant à une coupure (> 3 fragments manquants)
                if s_ > target:
                    continue
                if prev is not None and prev - s_ > 3:
                    break
                prev = s_
                rows.append(r)
                if len(rows) >= WINDOW:
                    break
            rows.reverse()
            reached = self.done_seq
        if len(rows) < 30:
            return {"type": "unknown", "rod": None, "confidence": 0.0, "goal_frame": None, "goal_side": None,
                    "mode": "stream", "error": "pas assez d'images (%d)" % len(rows)}
        t1 = time.time()
        res = self.analyzer.classify_raw(np.stack(rows))
        res.update(mode="stream", waited=round(waited, 2), classify=round(time.time() - t1, 2), frames=len(rows),
                   target_seq=target, reached_seq=reached)
        return res
