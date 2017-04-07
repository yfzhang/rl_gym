#!/bin/bash

./run.py \
  --mode train \
  --model dqn_atari \
  --agent dqn \
  --game BreakoutDeterministic-v3 \
  --base-dir exp \
  --save-weight-interval 250000 \
  --save-log-interval 100 \
  --max-steps 10000000 \
  --memory-limit 1000000 \
  --memory-window-length 4 \
  --visualize-train false \
  $@
