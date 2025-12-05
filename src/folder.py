import os

def count_files(path):
    """Count all files inside a folder recursively."""
    if not os.path.exists(path):
        return 0
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


def scan_dataset(base_path):
    """Scan dataset structure and count real/fake files separately."""
    sections = ["train", "val", "test"]
    results = {}

    total_real = 0
    total_fake = 0

    for sec in sections:
        sec_path = os.path.join(base_path, sec)

        real_path = os.path.join(sec_path, "real")
        fake_path = os.path.join(sec_path, "fake")

        real_count = count_files(real_path)
        fake_count = count_files(fake_path)

        results[f"{sec}_real"] = real_count
        results[f"{sec}_fake"] = fake_count

        total_real += real_count
        total_fake += fake_count

    results["total_real"] = total_real
    results["total_fake"] = total_fake

    return results


# ====================
# Dataset paths
# ====================
celeb_path = "data/Celeb_V2"
dfd_path   = "data/faces"


# ==== Celeb_V2 ====
print("=== Celeb_V2 ===")
celeb_results = scan_dataset(celeb_path)
for k, v in celeb_results.items():
    print(f"{k}: {v}")

print("\n=== DFD ===")
dfd_results = scan_dataset(dfd_path)
for k, v in dfd_results.items():
    print(f"{k}: {v}")
