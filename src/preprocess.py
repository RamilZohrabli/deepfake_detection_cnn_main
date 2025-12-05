import os
import json
import random
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

random.seed(42)

def list_videos(path):
    if not os.path.isdir(path):
        return []
    return sorted(str(p) for p in Path(path).glob("*.mp4"))

def sample_indices(total_frames, n=8):
    if total_frames <= n:
        return list(range(total_frames))
    return [int(i * (total_frames - 1) / (n - 1)) for i in range(n)]

def crop_face(img, min_size=80, margin=0.25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy = x + w / 2, y + h / 2
    size = int(max(w, h) * (1.0 + margin))

    x0 = max(0, int(cx - size / 2))
    y0 = max(0, int(cy - size / 2))
    x1 = min(img.shape[1], x0 + size)
    y1 = min(img.shape[0], y0 + size)

    crop = img[y0:y1, x0:x1]
    return crop if crop.size else None

def process_video(video, out_dir, frames_per_video=8):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = sample_indices(total, frames_per_video)
    saved = []
    base = Path(video).stem

    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        face = crop_face(frame)
        if face is None:
            continue

        face = cv2.resize(face, (224, 224))
        out_path = os.path.join(out_dir, f"{base}_f{i}.jpg")
        cv2.imwrite(out_path, face)
        saved.append(out_path)

    cap.release()
    return saved

def split_fully_balanced(real_videos, fake_videos):
    n = min(len(real_videos), len(fake_videos))
    real_videos = real_videos[:n]
    fake_videos = fake_videos[:n]

    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    def split(arr):
        N = len(arr)
        n_train = int(0.7 * N)
        n_val = int(0.15 * N)
        return arr[:n_train], arr[n_train:n_train+n_val], arr[n_train+n_val:]

    r_tr, r_va, r_te = split(real_videos)
    f_tr, f_va, f_te = split(fake_videos)

    n_train = min(len(r_tr), len(f_tr))
    n_val = min(len(r_va), len(f_va))
    n_test = min(len(r_te), len(f_te))

    return {
        "train": {"real": r_tr[:n_train], "fake": f_tr[:n_train]},
        "val":   {"real": r_va[:n_val],   "fake": f_va[:n_val]},
        "test":  {"real": r_te[:n_test],  "fake": f_te[:n_test]},
    }

def build_and_preprocess():
    real_dir = "data/DFD_original_sequences"
    fake_dir = "data/DFD_manipulated_sequences"
    out_root = "data/faces"
    json_out = "data/splits/paths.json"

    os.makedirs(out_root, exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

    real = list_videos(real_dir)
    fake = list_videos(fake_dir)

    print("Real videos:", len(real))
    print("Fake videos:", len(fake))

    splits = split_fully_balanced(real, fake)

    # DEBUG: check video counts per split
    for s in ["train", "val", "test"]:
        print(f"{s}_videos: real={len(splits[s]['real'])}, fake={len(splits[s]['fake'])}")

    paths = {"train": {"real": [], "fake": []},
             "val": {"real": [], "fake": []},
             "test": {"real": [], "fake": []}}

    with ProcessPoolExecutor(max_workers=4) as ex:
        tasks = []
        for split in ["train", "val", "test"]:
            for cls in ["real", "fake"]:
                out_dir = os.path.join(out_root, split, cls)
                for vid in splits[split][cls]:
                    fut = ex.submit(process_video, vid, out_dir)
                    tasks.append((split, cls, fut))

        for split, cls, fut in tasks:
            paths[split][cls].extend(fut.result())

    with open(json_out, "w") as f:
        json.dump(paths, f, indent=2)

    for s in ["train", "val", "test"]:
        print(f"{s}_frames: real={len(paths[s]['real'])}, fake={len(paths[s]['fake'])}")

    print("DONE.")

if __name__ == "__main__":
    build_and_preprocess()
