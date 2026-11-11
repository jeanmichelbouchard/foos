# goal_training: build the models used by goal_service

These scripts take a collection of recorded goal clips and produce the three model files that `../goal_service` needs
(`ball.pt`, `rod.joblib`, `attack.joblib`). They run on a PC with an NVIDIA GPU. Nothing here is needed on the Pi.

The goal rules, the tracking code and the trajectory features are **not** duplicated: they live in
`../goal_service/` (`rules.py`, `tracking.py`) and are imported from there, so training and inference cannot diverge.

## Folders (set with environment variables)
| variable | meaning | default |
|---|---|---|
| `FOOS_VIDEOS` | recorded clips, organised by rod (see below) | `~/foos-videos` |
| `FOOS_ML_DATA` | generated data: frames, annotations, dataset, tracks, runs, models. Keep it outside cloud-synchronised folders (several GB) | `~/foos-ml` |
| `FOOS_LABELS` | clip tag lists (`autogoal.txt`, `corkball.txt`, `redball.txt`: one clip file name per line) | `./labels` |

### Clip layout
```
$FOOS_VIDEOS/1_rod/  2_rod/  3_rod/  5_rod/      one folder per rod that scored (goalie, defense, attack, center)
$FOOS_VIDEOS/3_rod/Slice/  Wall Shot/  Chevreuil/  One-Timer/     attack sub-types (clips of a normal attack goal sit in 3_rod/)
<unix timestamp>_short.h264                      the clip saved by the replay (about 3-4 s before the goal is detected)
```
Clips are H.264 files recorded at 49 frames/s, 1280x720 (the files `replay_short.h264` that foos-master writes).
The `labels/` files shipped here are the tag lists of the author's own clips: replace them with yours.

## Pipeline, in order
1. `extract_frames.py [n]`: pick a varied sample of frames from the clips (`samples.csv`).
2. `annotate.py`: click the ball in each frame (OpenCV window; candidates are proposed by `detect.py`).
3. `prepare_dataset.py`: build the YOLO dataset, split by session (not by frame).
4. `train_ball.py`: train the ball detector (`runs/ball/weights/best.pt`). `eval_ball.py` evaluates it.
5. `track_clips.py` then `track_multi.py [n] [weights.pt]`: run the detector on every frame of every clip.
6. `continuity.py`: link the detections into trajectories (`tracks_cont/`).
7. `rod_classify.py` and `attack_subtype.py`: honest accuracy of the two classifiers (cross-validation, one session at a time);
   `calibrate.py` shows accuracy versus confidence threshold (to choose the thresholds in `goal_service/analyzer.py`).
8. `train_final.py`: train the two classifiers on all the goals and write `models/rod.joblib` and `models/attack.joblib`.
9. Copy `ball.pt` (from `runs/.../weights/best.pt`), `rod.joblib` and `attack.joblib` into `../goal_service/models/`.

Improving the detector: `harvest_hard.py` collects hard examples (false positives after a goal, blurred balls),
`annotate.py hard` lets you correct them, and `prepare_dataset.py` adds them to the dataset. `compare_models.py` compares
two detectors on held-out sessions. `goal_events.py`, `goal_from_track.py` and `analyze_tracks.py` print statistics about
the goal rule; `fail_sheets.py` and `make_review_videos.py` produce images/videos of the goals the rule misses.

## Results obtained so far (author's table, 499 goals, held-out sessions)
Rod accuracy 92.2 % (white ball, without own goals); attack sub-type accuracy 81.8 % (always answering "normal": 60.1 %);
own goals 51.6 % (not handled yet); goalie goals are unlearnable with 8 examples.

## Licences
The ball detector is a fine-tuned YOLO11 model: `ultralytics` and its pretrained weights are AGPL-3.0. Trained weights
derived from them should be shared under AGPL-3.0. The repository is GPL-3.0. The models are not stored in git.
