"""Reliable exit from foos-master back to the command line.

The old 'quit' went through the bus and then DISPLAY.stop() (pi3d) in the bus thread: when stopping the display
blocked, the whole bus froze and the program never ended. Here the exit depends neither on the bus nor on the UI:
a watchdog kills the process whatever happens.

Sequence: .quit file (main.sh does not restart foos), request for a clean UI shutdown, state saving (atexit),
camera and child processes stopped, then os._exit after at most WATCHDOG_S seconds.
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
        open(".quit", "w").close()       # read by main.sh: do not restart foos.py
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
            atexit._run_exitfuncs()      # save the plugins' state
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
        subprocess.call(["pkill", "player"], timeout=2)     # replay player (video layer over the console)
    except Exception:
        pass
    os._exit(0)
