from datasets import load_dataset
import numpy as np
import gzip
import os

ds = load_dataset("tanganke/kmnist")

os.makedirs("data/KMNIST/raw", exist_ok=True)

def save_images(path, images):
    images = np.array(images, dtype=np.uint8)
    with gzip.open(path, "wb") as f:
        f.write(b'\x00\x00\x08\x03')
        f.write((len(images)).to_bytes(4, 'big'))
        f.write((28).to_bytes(4, 'big'))
        f.write((28).to_bytes(4, 'big'))
        f.write(images.tobytes())

def save_labels(path, labels):
    labels = np.array(labels, dtype=np.uint8)
    with gzip.open(path, "wb") as f:
        f.write(b'\x00\x00\x08\x01')
        f.write((len(labels)).to_bytes(4, 'big'))
        f.write(labels.tobytes())

save_images("data/KMNIST/raw/train-images-idx3-ubyte.gz", ds['train']['image'])
save_labels("data/KMNIST/raw/train-labels-idx1-ubyte.gz", ds['train']['label'])

save_images("data/KMNIST/raw/t10k-images-idx3-ubyte.gz", ds['test']['image'])
save_labels("data/KMNIST/raw/t10k-labels-idx1-ubyte.gz", ds['test']['label'])

print("KMNIST hazır")