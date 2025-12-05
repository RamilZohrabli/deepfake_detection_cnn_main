import matplotlib.pyplot as plt
import cv2, random
import os
def load_images(dir_path, n=4):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".jpg")]
    if len(files) == 0:
        raise Exception(f"No images in {dir_path}")
    if len(files) < n:
        n = len(files)
    imgs = random.sample(files, n)
    return [cv2.imread(i)[..., ::-1] for i in imgs]

real_dir = "data/faces/train/real"
fake_dir = "data/faces/train/fake"

real_imgs = load_images(real_dir)
fake_imgs = load_images(fake_dir)

# Real
plt.figure(figsize=(5,5))
for i, img in enumerate(real_imgs):
    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.axis("off")
plt.suptitle("Real Samples")
plt.savefig("real_samples.png", dpi=300)

# Fake
plt.figure(figsize=(5,5))
for i, img in enumerate(fake_imgs):
    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.axis("off")
plt.suptitle("Fake Samples")
plt.savefig("fake_samples.png", dpi=300)
