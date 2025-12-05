import os, json, random
random.seed(42)

REAL_DIR = "data/frames/real"
FAKE_DIR = "data/frames/fake"
OUT_JSON = "data/splits/paths.json"

real_images = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith(".jpg")]
fake_images = [os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith(".jpg")]

print(f"Real frames: {len(real_images)} | Fake frames: {len(fake_images)}")

random.shuffle(real_images)
random.shuffle(fake_images)

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]

real_train, real_val, real_test = split_data(real_images)
fake_train, fake_val, fake_test = split_data(fake_images)

min_train = min(len(real_train), len(fake_train))
real_train = real_train[:min_train]
fake_train = fake_train[:min_train]

splits = {
    "train": {"real": real_train, "fake": fake_train},
    "val": {"real": real_val, "fake": fake_val},
    "test": {"real": real_test, "fake": fake_test}
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Split file created: {OUT_JSON}")
print(f"Train → {len(real_train)} real + {len(fake_train)} fake")
print(f"Val → {len(real_val)} real + {len(fake_val)} fake")
print(f"Test → {len(real_test)} real + {len(fake_test)} fake")
