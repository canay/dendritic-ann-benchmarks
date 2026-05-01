import gzip
import os
import struct
import torch

RAW = "data/KMNIST/raw"
PROCESSED = "data/KMNIST/processed"
os.makedirs(PROCESSED, exist_ok=True)

def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
        return data.view(n, rows, cols)

def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
        return data.long()

train_images = read_idx_images(f"{RAW}/train-images-idx3-ubyte.gz")
train_labels = read_idx_labels(f"{RAW}/train-labels-idx1-ubyte.gz")
test_images = read_idx_images(f"{RAW}/t10k-images-idx3-ubyte.gz")
test_labels = read_idx_labels(f"{RAW}/t10k-labels-idx1-ubyte.gz")

torch.save((train_images, train_labels), f"{PROCESSED}/training.pt")
torch.save((test_images, test_labels), f"{PROCESSED}/test.pt")

print("KMNIST processed hazır:")
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)