from .io_base import IOBase
import evdev
from select import select
import logging
import time
logger = logging.getLogger(__name__)

import foos.config as config

# Score: top-row digits, one block of three keys per team, grouped on the side of the
# keyboard closest to that team (keyboard sits at the end of the table).
#   left block : 1 = minus, 2 = goal, 3 = plus      right block: 8 = minus, 9 = goal, 0 = plus
#   5 = ok
# config.keyboard_left_team (default "black" = the "rouges") says which team the left block drives.
LEFT_TEAM = getattr(config, "keyboard_left_team", "black")
RIGHT_TEAM = "yellow" if LEFT_TEAM == "black" else "black"

# Goal-type feedback: one key per type (letters, freed from the old score layout).
# Emitted as 'goal_type_key' events; a separate plugin applies them to the last goal.
TYPE_KEYS = {
    'KEY_G': 'goalie',      # 1_rod
    'KEY_D': 'defense',     # 2_rod
    'KEY_C': 'center',      # 5_rod (centre)
    'KEY_T': 'attack',      # 3_rod, none of the special shots below
    'KEY_P': 'slice',       # pissette
    'KEY_M': 'wall_shot',   # main courante
    'KEY_H': 'chevreuil',
    'KEY_O': 'one_timer',
    'KEY_A': 'own_goal',    # auto-but
    'KEY_X': 'ambiguous',
    'KEY_Z': 'undo',        # cancel the last correction
}


class Plugin(IOBase):
    key_map = {
        'KEY_1': LEFT_TEAM + '_minus',
        'KEY_3': LEFT_TEAM + '_plus',
        'KEY_8': RIGHT_TEAM + '_minus',
        'KEY_0': RIGHT_TEAM + '_plus',
        'KEY_5': 'ok',
    }

    goal_map = {
        'KEY_2': LEFT_TEAM,
        'KEY_9': RIGHT_TEAM,
    }

    def __init__(self, bus):
        self.devices = self.list_devices()

        if len(self.devices) == 0:
            logger.warn("Can't find any keyboards to read from - maybe you need permissions")
        else:
            logger.info("Reading key events from: {}".format(self.devices))

        super().__init__(bus)
        
    def list_devices(self):
        devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
        def hasAKey(device):
            caps = device.capabilities(verbose=True)
            events = caps.get(('EV_KEY', 1), [])
            a_keys = [e for e in events if e[0] == 'KEY_A']
            return len(a_keys) > 0
        
        return list(filter(hasAKey, devices))


    def handle_key(self, code, keystate):
        if keystate == evdev.events.KeyEvent.key_hold:
            return
        
        state = "down" if keystate == evdev.events.KeyEvent.key_down else "up"
        
        if code in self.key_map:
            btn = self.key_map[code]

            event_data = {'source': 'keyboard', 'btn': btn, 'state': state}
            self.bus.notify('button_event', event_data)

        if code in self.goal_map and state == "down":
            team = self.goal_map[code]
            self.bus.notify('goal_event', {'source': 'keyboard', 'team': team})
            
        if code in TYPE_KEYS and state == "down":
            self.bus.notify('goal_type_key', {'source': 'keyboard', 'type': TYPE_KEYS[code]})

        if code == 'KEY_DOT' and state == "down":  # PERIOD: three presses within 2 s, to avoid quitting by accident
            now = time.time()
            self.dot_presses = [t for t in getattr(self, 'dot_presses', []) if now - t < 2.0] + [now]
            logger.info("Quit key: press %d/3", len(self.dot_presses))
            if len(self.dot_presses) >= 3:
                from foos.quit import force_quit
                force_quit(self.bus)

    def reader_thread(self):
        # A mapping of file descriptors (integers) to InputDevice instances.
        devices = {dev.fd: dev for dev in self.devices}

        while True:
            r, w, x = select(devices, [], [])
            for fd in r:
                for event in devices[fd].read():
                    ce = evdev.categorize(event)
                    if isinstance(ce, evdev.KeyEvent):
                        self.handle_key(ce.keycode, ce.keystate)

        return
    
    def writer_thread(self):
        while True:
            self.write_queue.get()
            pass
