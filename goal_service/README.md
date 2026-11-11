# goal_service: foosball goal-type recognition (runs on a PC/laptop)

The Raspberry Pi keeps doing what it always did (camera, infrared goal detectors, score, replay). This service runs on
a more powerful machine with an NVIDIA GPU and tells the Pi **what kind of goal it was** (goalie, defense, center,
attack, slice, wall shot, chevreuil, one-timer), normally less than one second after the goal.

```
Pi camera --H.264 fragments (0.2 s)--> plugins/goal_stream.py --POST /fragment--> goal_service.py (GPU: ball tracking)
infrared goal ---> plugins/goal_analysis.py --POST /goal----------------------> goal_service.py (classification)
                                         <---------------- {"type": "slice", ...} (~0.8 s)
```

How it works: a YOLO11n detector finds the ball in every frame, a continuity tracker links the detections into a
trajectory, the frame where the ball disappears into a goal is found, and two small classifiers (rod, attack sub-type)
decide the type from the trajectory before that frame. See `../doc/GoalStream.md` for the protocol and measurements.

## Install (PC)
```
pip install -r requirements.txt     # install the CUDA build of PyTorch first for GPU use
```
Put `ball.pt`, `rod.joblib` and `attack.joblib` in `models/` (see `models/README.md`).

## Run
```
python goal_service.py [port]       # default 8765
```
Then on the Pi, in `config.py`: `goal_service_url = "http://<pc address>:8765"` and `plugins.add('goal_analysis')`,
`plugins.add('goal_stream')`. A wired network link between the Pi and the PC is strongly recommended (the live stream
is about 17 Mbit/s). Allow the port in the PC firewall.

## HTTP API
* `GET /health`: `{"ok": true, "stream": {...}}` (stream statistics: lag, dropped fragments...).
* `POST /fragment?seq=N`: body = one H.264 fragment (10 frames); `N` increases with every fragment.
* `POST /goal?last_seq=N`: type of the goal that happened in the fragments after `N` (waits ~0.6 s for the end of the shot).
* `POST /analyze?fps=49`: body = a whole clip (fallback, 3 to 5 s on a GTX 1650 Ti).

## Files
* `goal_service.py`, `stream_analyzer.py`, `analyzer.py`: server, live mode, model chain.
* `tracking.py`: frame reading, multi-candidate detection, continuity tracking. `rules.py`: goal zones and trajectory features.
* `tests/replay_match_test.py` replays a recording made by `plugins/match_recorder.py` against a running service;
  `tests/test_pi_stream.py` does the same through the Pi plugins with a fake bus.

## Limits
* Calibrated for **49 frames/s** (`video_fps = 49` on the Pi), a 1280x720 image, one table and one camera framing.
  Goal zones in `rules.py` are pixel coordinates.
* On a GTX 1650 Ti, detection takes about 160 ms per 0.2 s fragment (real time with 20 % margin). FP16 does not work
  on that card.
* About 40 % of goals are currently classified `unknown` (ball not tracked into the goal), and rod/sub-type accuracy
  measured on held-out sessions is about 92 % / 82 %.
