import os
import json

def build_celebdf_paths():
    root = "data/Celeb_V2"
    out_json = "data/splits/paths_celebdf.json"

    splits = ["train", "val", "test"]
    classes = ["real", "fake"]

    paths = {s: {c: [] for c in classes} for s in splits}

    for split in splits:
        for cls in classes:
            dir_path = os.path.join(root, split, cls)
            if not os.path.isdir(dir_path):
                raise Exception(f"Missing folder: {dir_path}")

            files = [
                f for f in os.listdir(dir_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            files.sort()

            for f in files:
                full = os.path.join(dir_path, f)
                # JSON üçün slashi normallaşdıraq
                paths[split][cls].append(full.replace("\\", "/"))

            print(f"{split}-{cls}: {len(files)} images")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(paths, f, indent=2)

    print("Saved:", out_json)


if __name__ == "__main__":
    build_celebdf_paths()
