#!/bin/bash

./run.py \
  --mode train \
  --model 3dense \
  --agent dqn \
  --game CartPole-v1 \
  --base-dir exp \
  --save-weight-interval 1000 \
  --save-log-interval 10 \
  --max-steps 5000 \
  $@


