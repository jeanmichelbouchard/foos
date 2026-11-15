import time
import logging
import ctypes
import time
import RPi.GPIO as GPIO
from .io_base import IOBase
import foos.config as config
import foos.process as process

#logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger(__name__)

#The button should be connected to GND
class Button:
    def __init__(self, bus, pin_number, name, debounce_time_ms = 10):
        self.pin = pin_number
        self.name = name
        self.bus = bus
        if self.pin:
            GPIO.setup(self.pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
            GPIO.add_event_detect(self.pin, GPIO.FALLING, callback = self.button_changed, bouncetime = debounce_time_ms)
            self.button_state = GPIO.input(self.pin)
            logger.info("{0} button initial state is {1}".format(self.name, self.button_state))
        else:
            logger.warn("Cannot init button {0}, pin not specified".format(self.name))
        
    def button_changed(self, channel):
        input = GPIO.input(self.pin)
        if input == self.button_state:
#            logger.info("{0} button is still in state {1}".format(self.name, self.button_state))
            return;
        self.button_state = input
        logger.info("{0} button changed to {1}!".format(self.name, input));
        event_data = {'source': 'rpi', 'btn': self.name, 'state': 'up' if input else 'down'}
        if event_data:
            self.bus.notify('button_event', event_data)

    def __del__(self):
        if self.pin:
            GPIO.remove_event_detect(self.pin)

class GoalDetector:
    """Goal detector made of a 38 kHz IR led and a TSSP58038 receiver (output LOW while it sees the 38 kHz light).

    Two ways to mount them, selected with ir_barrier_reflective in config.py:

    False (default): light barrier. Led and receiver face each other across the ball channel. The receiver sees the
        light all the time (LOW); the ball cuts it (HIGH) and the goal is the falling edge, when the light comes back
        behind the ball.
    True: reflective. Led and receiver sit side by side on the same wall, shielded from each other. The receiver sees
        nothing at rest (HIGH); the goal is the falling edge, when the ball sends the light back to the receiver.

    The edge is the same in both modes. The reflection of a rolling ball flickers, so the reflective mode ignores further
    edges for a longer time and checks that the receiver still sees the light a moment later (no goal on a glitch).
    """
    BARRIER_BOUNCE_MS = 10
    REFLECTIVE_BOUNCE_MS = 300
    REFLECTIVE_CONFIRM_S = 0.002

    def __init__(self, bus, pin_number, team):
        self.bus = bus
        self.pin = pin_number
        self.team = team
        self.reflective = bool(getattr(config, "ir_barrier_reflective", False))
        if self.pin:
            #GPIO.setup(self.pin, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
            bounce = self.REFLECTIVE_BOUNCE_MS if self.reflective else self.BARRIER_BOUNCE_MS
            GPIO.add_event_detect(self.pin, GPIO.FALLING, callback=self.on_goal, bouncetime=bounce)
            logger.info("Goal detector {0}: {1} mode".format(self.team, "reflective" if self.reflective else "light barrier"))
        else:
            logger.warn("Cannot init GoalDetector {0}, pin not specified".format(self.team))
    def __del__(self):
        if self.pin:
            GPIO.remove_event_detect(self.pin)

    def on_goal(self, channel):
        if self.reflective:
            time.sleep(self.REFLECTIVE_CONFIRM_S)
            if GPIO.input(self.pin):
                logger.info("Goal detector {}: reflection too short, ignored".format(self.team))
                return
        logger.info("Goal {}!".format(self.team))
        self.bus.notify('goal_event', {'source': 'rpi', 'team': self.team})

class IRBarrierPwmGenerator:
    def __init__(self):
        
        # https://sites.google.com/site/semilleroadt/raspberry-pi-tutorials/gpio:
        # "According to it, configure GPIO18 (WiringPi Pin 1)..."

        # Raspberry base PWM frequency: 19,200,000 Hz
        # Resulted frequency: base freq / 101 / 5 = 38.019 kHz
        # Signal duty cycle = 3/5 = ~60%
        process.call_and_log("gpio mode 1 pwm && gpio pwm-ms && gpio pwmr 5 && gpio pwmc 101 && gpio pwm 1 2", shell=True)
        
    def __del__(self):
        process.call_and_log("gpio pwm 1 0", shell=True)
  
class Plugin(IOBase):
    def __init__(self, bus):
        GPIO.setmode(GPIO.BOARD)
        
        self.goal_pin_black = config.io_raspberry_pins["irbarrier_team_black"]
        self.goal_pin_yellow = config.io_raspberry_pins["irbarrier_team_yellow"]

        self.ok_button_pin = config.io_raspberry_pins["ok_button"]
 
        self.yellow_plus_pin = config.io_raspberry_pins["yellow_plus"]
        self.yellow_minus_pin = config.io_raspberry_pins["yellow_minus"]

        self.black_plus_pin = config.io_raspberry_pins["black_plus"]
        self.black_minus_pin = config.io_raspberry_pins["black_minus"]
        
        self.ir_barrier_pwm = IRBarrierPwmGenerator()

        time.sleep(0.5)   # let the PWM really start before starting detectors
            
        self.goal_detector_black = GoalDetector(bus, self.goal_pin_black, "black")
        self.goal_detector_yellow = GoalDetector(bus, self.goal_pin_yellow, "yellow")

        self.yellow_plus_button = Button(bus, self.yellow_plus_pin, 'yellow_plus')
        self.yellow_minus_button = Button(bus, self.yellow_minus_pin, 'yellow_minus')
        
        self.black_plus_button = Button(bus, self.black_plus_pin, 'black_plus')
        self.black_minus_button = Button(bus, self.black_minus_pin, 'black_minus')

        self.ok_button = Button(bus, self.ok_button_pin, 'ok')
            
        super().__init__(bus)

    def reader_thread(self):
        while True:
            #Do nothing for now
            time.sleep(1)
            
    def writer_thread(self):
        while True:
            line = self.write_queue.get()
            #Do nothing for now
            time.sleep(1)

        
