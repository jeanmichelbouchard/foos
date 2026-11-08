#!/bin/sh

# Relance foos.py s'il plante. Une sortie voulue (touche . trois fois) cree le fichier .quit: on s'arrete alors.
rm -f .quit
while true; do
  sleep 10
  python3 foos.py
  if [ -f .quit ]; then
    rm -f .quit
    break
  fi
done
