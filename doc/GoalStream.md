# Analyse du type de but en continu (goal_stream)

## But
Afficher le type de but (gardien, défense, centre, attaque, pissette, main courante, chevreuil, one-timer) **pendant la
reprise**, moins d'une seconde après le signal infrarouge, au lieu de 15 à 20 s avec l'envoi du clip après coup.

## Principe
Le modèle tourne sur le portable (GPU). Plutôt que d'attendre le but pour lui envoyer un clip, le Pi lui envoie la vidéo
**au fur et à mesure**; le portable suit la balle en continu. Au but, il ne reste qu'à classer les dernières images.

```
caméra --(fragments h264 de 0,2 s, en RAM)--> plugin goal_stream --POST /fragment?seq=N--> portable
                                                                                            décode + détecte la balle
score_goal --> plugin goal_analysis --POST /goal?last_seq=N-----------------------------> attend 3 fragments, classe
            <------------------------------- {"type": "slice", ...} (0,7 à 0,9 s) --------
```

* **Fragments**: la caméra écrit des fichiers `out%04d.h264` de 10 images (0,2 s), chacun autonome (en-têtes + image clé).
  Les noms tournent sur 100 valeurs; `goal_stream` les numérote avec son propre compteur croissant `seq`.
* **`/fragment?seq=N`** (Pi → portable): corps = un fragment. Connexion persistante, ~430 Ko, 5 par seconde (~17 Mbit/s):
  **câble réseau recommandé** (le Wi-Fi 2,4 GHz du Pi 3B ne tiendra pas).
* **`/goal?last_seq=N`** (Pi → portable): `N` = dernier fragment terminé au moment du but. Le but est dans le fragment
  suivant (en cours d'écriture); le portable attend 3 fragments de plus (~0,6 s) pour voir la balle absente ≥ 15 images,
  puis classe les 200 dernières images (même longueur que les clips d'entraînement).
* **Activité**: l'envoi n'est actif que pendant le jeu (mouvement, IR, pointage) et s'arrête après 5 min sans activité,
  pour ne pas faire tourner le GPU toute la soirée. À l'activation, les 20 derniers fragments sont envoyés d'un coup.
* **Secours**: si le suivi en continu échoue (portable absent, erreur), `goal_analysis` retombe sur l'analyse du clip du
  replay (`POST /analyze`), puis sur rien: le jeu n'attend jamais.

## Mesures (partie du 2026-11-08, 9 buts, rejouée au rythme de la caméra)
* Détection: 162 ms par fragment pour un budget de 200 ms (GTX 1650 Ti) → temps réel avec 20 % de marge.
* Retard de traitement: médiane 0,2 s, maximum 0,5 s.
* Délai entre le but et le type: **0,74 à 0,90 s**; types identiques à l'analyse du clip dans 9 cas sur 9.
* Le FP16 est inutilisable sur cette carte (plus lent, résultats invalides); la détection à 1280 px est le plafond.

## Limites connues
* Le Pi doit envoyer les fragments à cadence régulière: une copie sur la carte SD en même temps (clip de but,
  `match_recorder`) a produit des retards jusqu'à 6 s. goal_stream lit en RAM et n'écrit rien sur la carte.
* La fenêtre de classement suppose 49 images/s (`video_fps = 49`).
* Le cadrage de la caméra doit rester celui de l'entraînement (zones de but en pixels dans `goal_from_track.py`).
* Non essayé sur le Pi: la surveillance `inotify` du dossier de fragments, le Python 3.5, le réseau filaire.

## Fichiers
* `plugins/goal_stream.py` (Pi), `plugins/goal_analysis.py` (Pi, appelle `/goal`),
* `tools/ball/stream_analyzer.py`, `tools/ball/goal_service.py` (portable),
* bancs d'essai: `tools/ball/replay_match_test.py` (service seul), `tools/ball/test_pi_stream.py` (plugins du Pi).

## Affichage
Le type de but trouve (ou corrige au clavier) s'affiche en haut de l'ecran, au centre, pendant la reprise puis 3 s apres (`ui.py`: `BANNER_LABELS`, `_goal_type`). Taille et position: `font_size=150` et `self._ftop(130)` dans `__setup_sprites`. Un type inconnu n'affiche rien.
