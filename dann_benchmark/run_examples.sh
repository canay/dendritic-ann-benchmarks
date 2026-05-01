#!/usr/bin/env bash
set -e

python benchmark.py \
  --dataset fashionmnist \
  --subset-fraction 0.10 \
  --epochs 3 \
  --seeds 0 \
  --models dann_lrf naive_branch vann_same mlp_param

python benchmark.py \
  --dataset kmnist \
  --subset-fraction 0.10 \
  --epochs 3 \
  --seeds 0 \
  --models dann_lrf naive_branch vann_same mlp_param
