# Models

`goal_service` needs three files in this folder (or in the folder named by the `FOOS_MODELS` environment variable):

| file | what it is |
|---|---|
| `ball.pt` | YOLO11n ball detector, trained at 1280 px on frames of one foosball table (white ball, camera above the table) |
| `rod.joblib` | gradient boosting classifier: which rod scored (goalie, defense, center, attack) |
| `attack.joblib` | gradient boosting classifier: attack sub-type (normal, slice, wall shot, chevreuil, one-timer) |

They are not stored in git. The models were trained on one table and one camera position, so on another table the
goal zones in `rules.py` (pixels) and the models will need to be adapted and retrained.
