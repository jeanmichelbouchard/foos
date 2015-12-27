#!/usr/bin/env python
import time
import sys
import threading
import queue
import collections


class Pattern:
    def __init__(self, time, leds=[]):
        self.time = time
        self.leds = leds


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable):
            for sub in flatten(el):
                yield sub
        else:
            yield el


class LedController:
    def __init__(self, writers):
        self.queue = queue.Queue()
        self.writers = writers
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while True:
            loop, m = self.queue.get()
            first = True
            while first or loop:
                first = False
                for p in flatten(m):
                    if self.__canRun():
                        self.setLeds(p.leds)
                        self.__safeSleep(p.time)
                    else:
                        loop = False
                        break

            #reset leds
            self.setLeds()

    def __safeSleep(self, t):
        start = time.time()
        while (time.time() < start + t) and self.__canRun():
            time.sleep(0.05)

    def __canRun(self):
        return self.queue.empty()

    def setLeds(self, leds=[]):
        for w in self.writers:
            w.write_data(leds)

    def setMode(self, mode, loop=False):
        self.stop = True
        self.queue.put((loop, mode))


pat_reset = 3 * [Pattern(0.2, ["BI", "BD", "YI", "YD"]),
                 Pattern(0.1),
                 Pattern(0.2, ["BI", "BD", "YI", "YD"]),
                 Pattern(1)]

pat_poweredoff = [Pattern(0.5, ["OK"]),
                  Pattern(1)]

pat_goal = [[Pattern(0.1, ["BD", "YD"]),
             Pattern(0.1, ["OK"]),
             Pattern(0.1, ["BI", "YI"])],
            3 * [Pattern(0.1),
                 Pattern(0.1, ["BI", "BD", "OK", "YI", "YD"])]]

pat_ok = [Pattern(0.3, ["OK"])]

pat_error = 2 * [Pattern(0.3, ["YD", "BD"]),
                 Pattern(0.3)]

pat_demo = [Pattern(1, ["BD"]),
            Pattern(1, ["BI"]),
            Pattern(1, ["YD"]),
            Pattern(1, ["YI"]),
            Pattern(1, ["OK"])]


if __name__ == "__main__":
    class DemoWriter():
        def write_data(self, leds):
            print("\r", end="")
            for led in ["BD", "BI", "OK", "YI", "YD"]:
                print("0" if led in leds else " ", end=" ")
            sys.stdout.flush()

    controller = LedController([DemoWriter()])
    controller.setMode(pat_poweredoff, loop=True)
    time.sleep(5)
    controller.setMode(pat_goal)
    time.sleep(5)
    controller.setMode(pat_upload)
    time.sleep(5)