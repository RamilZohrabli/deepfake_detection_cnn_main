import os
import time
import torch
from torch.utils.data import DataLoader

import train_celebdf as train
from dataset_list import FrameListDataset

JSON = "data/splits/paths_celebdf.json"
# keep consistent with train_celebdf.py
MODELS_DIR = "models_celeb"
device = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark_model(model_name: str, num_batches: int = 50, batch_size: int = 32):
    """
    Celeb-DF üçün modeli yükləyib test setində latency / FPS ölçür.
    analyze_celebdf_advanced.py buradan MƏHZ BU FUNKSİYANI import edir.
    """
    print(f"\n[CELEB-DF] Benchmarking {model_name} on {device.upper()}...")

    # Dataset
    test_ds = FrameListDataset(JSON, "test")
    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    # Model + checkpoint
    model = train.build_model(model_name)
    ckpt_path = os.path.join(MODELS_DIR, f"celeb_{model_name}_best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Warm-up
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= 3:
                break
            x = x.to(device)
            _ = model(x)

    total_time = 0.0
    total_frames = 0

    # Measurement
    with torch.no_grad():
        it = iter(loader)
        for i in range(num_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                break

            x = x.to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(x)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_time += (t1 - t0)
            total_frames += x.size(0)

    if total_frames == 0 or total_time == 0:
        print("[CELEB-DF] Benchmark: no frames processed.")
        return {
            "model": model_name,
            "avg_latency_ms": 0.0,
            "fps": 0.0,
        }

    avg_latency = (total_time / total_frames) * 1000.0  # ms/frame
    fps = total_frames / total_time

    print(f"[CELEB-DF] Average latency: {avg_latency:.2f} ms/frame")
    print(f"[CELEB-DF] Throughput:      {fps:.2f} FPS")

    return {
        "model": model_name,
        "avg_latency_ms": float(avg_latency),
        "fps": float(fps),
    }


if __name__ == "__main__":
    results = {}
    for m in ["mobilenetv3", "efficientnet_b0", "resnet18"]:
        results[m] = benchmark_model(m)

    print("\n[CELEB-DF] === Benchmark Summary ===")
    for k, v in results.items():
        print(f"{k}: {v['avg_latency_ms']:.2f} ms/frame, {v['fps']:.2f} FPS")
