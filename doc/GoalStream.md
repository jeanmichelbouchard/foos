# Live goal-type recognition (goal_stream)

## Goal
Show the type of a goal (goalie, defense, center, attack, slice, wall shot, chevreuil, one-timer) **during the replay**,
less than a second after the infrared signal, instead of 15 to 20 s when the clip is sent after the fact.

## Principle
The model runs on a PC (GPU). Instead of waiting for the goal to send it a clip, the Pi sends it the video **as it is
recorded**; the PC tracks the ball continuously. When a goal is scored only the last frames remain to be classified.

```
camera --(0.2 s h264 fragments, in RAM)--> plugin goal_stream --POST /fragment?seq=N--> PC
                                                                                        decodes + detects the ball
score_goal --> plugin goal_analysis --POST /goal?last_seq=N-----------------------------> waits for 3 fragments, classifies
            <------------------------------- {"type": "slice", ...} (0.7 to 0.9 s) ----
```

* **Fragments**: the camera writes `out%04d.h264` files of 10 frames (0.2 s), each self-contained (headers + key frame).
  The names wrap around after 100 values; `goal_stream` numbers them with its own increasing counter `seq`
  (initialised from the clock so it never restarts at zero after a restart of foos-master; the PC also detects a
  counter that goes backwards and starts a new session).
* **`/fragment?seq=N`** (Pi to PC): body = one fragment. Persistent connection, ~430 KB, 5 per second (~17 Mbit/s):
  **a wired network link is recommended** (the Wi-Fi of a Pi 3B will not keep up).
* **`/goal?last_seq=N`** (Pi to PC): `N` = last fragment finished when the goal was counted. The goal is in the next
  fragment (still being written); the PC waits for 3 more fragments (~0.6 s) to see the ball absent for >= 15 frames,
  then classifies the last 200 frames (the same length as the training clips). The window stops at a break of more
  than 3 missing fragments, so frames from another period are never mixed in.
* **Activity**: sending is only active during play (movement, IR, score adjustments) and stops after 5 minutes without
  activity, so the GPU does not run all evening. On wake-up the 20 latest fragments are sent at once.
* **Fallback**: if live tracking fails (PC absent, error), `goal_analysis` falls back to analysing the replay clip
  (`POST /analyze`), then to nothing: the game never waits.

## Display and keys
* The goal type found (or corrected on the keyboard) is shown at the top of the screen, in the middle, during the replay
  and for 3 s after it (`foos/ui/ui.py`: `BANNER_LABELS`, `_goal_type`). Size and position: `font_size=150` and
  `self._ftop(130)` in `__setup_sprites`. An unknown type shows nothing.
* The texts shown to the players (banner and end-of-match statistics) default to English and can be changed with
  `goal_type_labels` in `config.py` (see `config.py.sample`).
* Keys (plugin `io_evdev_keyboard`): `g` goalie, `d` defense, `c` center, `t` attack, `p` slice, `m` wall shot,
  `h` chevreuil, `o` one-timer, `a` own goal, `x` ambiguous, `z` undo. Score keys: `1 2 3` and `8 9 0`
  (`keyboard_left_team` chooses which team drives the left block).

## Measurements (match of 2026-11-08, 9 goals, replayed at the camera's pace)
* Detection: 162 ms per fragment for a budget of 200 ms (GTX 1650 Ti) = real time with a 20 % margin
  (needs the "High performance" power mode on a laptop).
* Processing lag: median 0.2 s, maximum 0.5 s.
* Delay between the goal and the type: **0.74 to 0.90 s**; same types as the clip analysis in 9 cases out of 9.
* FP16 is unusable on that card (slower, invalid results); detection at 1280 px is the ceiling.

## Known limits
* The Pi must send fragments at a regular pace: a copy to the SD card at the same time (goal clip, `match_recorder`)
  produced delays of up to 6 s. `goal_stream` reads from RAM and writes nothing to the card.
* The classification window assumes 49 frames/s (`video_fps = 49`).
* The camera framing must stay the one used for training (goal zones are pixel coordinates in `goal_service/rules.py`).
* Not yet verified on the Pi itself: the `inotify` watch on the fragment folder and Python 3.5 (the code avoids newer syntax).

## Files
* `plugins/goal_stream.py` (Pi), `plugins/goal_analysis.py` (Pi, calls `/goal`), `foos/ui/ui.py` (banner, statistics),
* `goal_service/` (PC): `goal_service.py`, `stream_analyzer.py`, `analyzer.py`, `tracking.py`, `rules.py`,
* test benches: `goal_service/tests/replay_match_test.py` (service alone), `goal_service/tests/test_pi_stream.py` (Pi plugins).

## Center-rod goals
In the author's house rules a goal scored by the center (5-bar) rod does not count, unless it is an own goal. With
`center_goals_auto_cancel = True` (and `center_cancel_confidence`, default 0.90) `plugins/goal_analysis.py` removes the
point (`decrement_score`) when the model says "center" with a high confidence, or when a player presses the center key (`c`).
The banner shows "center goal void" and the statistics list it under "Center (void)". The own-goal key (`a`), or any other
type key, gives the point back (`increment_score`); `z` undoes the last correction. The model cannot tell an own goal from
a center goal (51.6 % on own goals), so the players decide. Cancellation is skipped, and the goal flagged for review in the
journal, when the match has just been won (the score has already been reset). Tests: `goal_service/tests/test_center_cancel.py`.

## Grace period before a victory
If the last point is a center goal, the match must not end before the players (or the model) can remove it.
* `plugins/game.py` already waits for the end of the replay before declaring a victory; the delay after the replay is now
  configurable (`win_check_delay`, default 2 s; 5 s in the author's `config.py`). During the replay and this delay the
  players can remove the point with the minus key of the team, or press the center key (`c`).
* `plugins/goal_analysis.py` announces `goal_analysis_pending` when a goal is counted and `goal_analysis_done` when the
  analysis (and a possible cancellation) is finished; the game waits for it, at most `win_hold_max` seconds (default 10).
  This covers the slow fallback (clip analysis, 15 s or more), where the answer would otherwise arrive after the victory.
* Tests: `goal_service/tests/test_win_grace.py`.
