import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    mobilenet_v3_small, efficientnet_b0, resnet18,
    MobileNet_V3_Small_Weights, EfficientNet_B0_Weights, ResNet18_Weights
)
from dataset_list import FrameListDataset

JSON = "data/splits/paths.json"
MODELS_DIR = "models"
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(name):
    if name == "mobilenetv3":
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif name == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(device)


def benchmark_model(model_name, num_batches=50, batch_size=32):
    print(f"\nBenchmarking {model_name} on {device.upper()}...")

    # Load subset of test set
    test_ds = FrameListDataset(JSON, "test")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = build_model(model_name)
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
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

    with torch.no_grad():
        it = iter(loader)
        for i in range(num_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                break
            x = x.to(device)

            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize() if device == "cuda" else None
            t1 = time.perf_counter()

            elapsed = t1 - t0
            total_time += elapsed
            total_frames += x.size(0)

    if total_frames == 0:
        print("No frames processed. Check dataset.")
        return

    avg_latency = (total_time / total_frames) * 1000.0  # ms per frame
    fps = total_frames / total_time if total_time > 0 else 0.0

    print(f"Average latency: {avg_latency:.2f} ms/frame")
    print(f"Throughput:      {fps:.2f} FPS")

    return {
        "model": model_name,
        "avg_latency_ms": avg_latency,
        "fps": fps,
    }


if __name__ == "__main__":
    results = {}
    for m in ["mobilenetv3", "efficientnet_b0", "resnet18"]:
        results[m] = benchmark_model(m)

    print("\n=== Real-time Benchmark Summary ===")
    for k, v in results.items():
        if v is None:
            continue
        print(f"{k}: {v['avg_latency_ms']:.2f} ms/frame, {v['fps']:.2f} FPS")
