"""Sortie fiable de foos-master vers la ligne de commande.

L'ancien 'quit' passait par le bus puis DISPLAY.stop() (pi3d) dans le thread du bus: quand l'arret de l'affichage
bloquait, tout le bus se figeait et le programme ne se terminait jamais. Ici la sortie ne depend ni du bus ni de l'UI:
un chien de garde tue le processus quoi qu'il arrive.

Sequence: fichier .quit (main.sh ne relance pas foos), demande d'arret propre a l'UI, sauvegarde de l'etat (atexit),
arret de la camera et des processus enfants, puis os._exit apres au plus WATCHDOG_S secondes.
"""
import atexit
import logging
import multiprocessing
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

WATCHDOG_S = 3
_started = False


def force_quit(bus=None):
    global _started
    if _started:
        return
    _started = True
    logger.warning("Quit requested")
    try:
        open(".quit", "w").close()       # lu par main.sh: ne pas relancer foos.py
    except OSError:
        pass

    def worker():
        if bus is not None:
            try:
                bus.notify('quit')       # arret propre de l'affichage si possible
            except Exception:
                pass
        time.sleep(1)
        try:
            atexit._run_exitfuncs()      # sauvegarde de l'etat des plugins
        except Exception:
            logger.exception("exit functions failed")
        _kill_all()

    def watchdog():
        time.sleep(WATCHDOG_S)
        _kill_all()

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()


def _kill_all():
    for p in multiprocessing.active_children():
        try:
            p.terminate()
        except Exception:
            pass
    try:
        subprocess.call(["pkill", "raspivid"], timeout=2)
    except Exception:
        pass
    os._exit(0)
