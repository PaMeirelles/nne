# NNE

The minimal NEAT experiment trains player one to shoot a stationary target in
the center of the board. Each genome is evaluated from north, south, east, and
west so fitness reflects learning the observation-to-action mapping rather than
getting one constant shot right by chance.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python neat_training.py --generations 30
```

Add `--replay` to open the existing pygame replay viewer for the winner:

```bash
.venv/bin/python neat_training.py --generations 30 --replay
```

The network receives five normalized values: target delta X/Y, player X/Y,
and whether its bullet is active. Its nine outputs are interpreted by argmax as
idle, four movement directions, or four shooting directions.
