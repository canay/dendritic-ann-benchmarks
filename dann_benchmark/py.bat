pip install -r requirements.txt
python benchmark.py \
  --dataset fashionmnist \
  --subset-fraction 0.1 \
  --epochs 3 \
  --seeds 0 \
  --models dann_lrf naive_branch vann_same mlp_param

pause