#!/bin/bash

cd ~/dann_benchmark
source venv/bin/activate

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

echo "[2/6] KMNIST full"
python benchmark.py --dataset kmnist --output-dir runs_kmnist_full --epochs 30 --batch-size 256 --num-workers 4 --seeds 0 1 2 3 4 5 6 7 8 9 --models dann_random dann_lrf dann_grf naive_branch vann_same mlp_param --device cpu

echo "[3/6] FashionMNIST low-data 0.2"
python benchmark.py --dataset fashionmnist --output-dir runs_fashion_low02 --subset-fraction 0.2 --epochs 30 --batch-size 256 --num-workers 4 --seeds 0 1 2 3 4 5 6 7 8 9 --models dann_lrf naive_branch mlp_param --device cpu

echo "[4/6] FashionMNIST low-data 0.1"
python benchmark.py --dataset fashionmnist --output-dir runs_fashion_low01 --subset-fraction 0.1 --epochs 30 --batch-size 256 --num-workers 4 --seeds 0 1 2 3 4 5 6 7 8 9 --models dann_lrf naive_branch mlp_param --device cpu

echo "[5/6] CIFAR-10 full"
python benchmark.py --dataset cifar10 --output-dir runs_cifar_full --epochs 30 --batch-size 256 --num-workers 4 --seeds 0 1 2 3 4 5 6 7 8 9 --models dann_lrf naive_branch mlp_param vann_same --device cpu

echo "[6/6] CIFAR-10 low-data 0.2"
python benchmark.py --dataset cifar10 --output-dir runs_cifar_low02 --subset-fraction 0.2 --epochs 30 --batch-size 256 --num-workers 4 --seeds 0 1 2 3 4 5 6 7 8 9 --models dann_lrf naive_branch mlp_param vann_same --device cpu

echo "ALL DONE"
