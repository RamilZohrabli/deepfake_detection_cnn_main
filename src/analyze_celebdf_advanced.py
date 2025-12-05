import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# src-dən import üçün
sys.path.append("src")

import train_celebdf as train
import eval_celebdf as eval_mod
from benchmark_rt_celebdf import benchmark_model
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MODELS = ["mobilenetv3", "efficientnet_b0", "resnet18"]
MODEL_LABELS = {
    "mobilenetv3": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
}

LOG_DIR = "logs_celeb"                  # <<<
RESULTS_DIR = "analysis_results_celeb"  # ayrı folder
os.makedirs(RESULTS_DIR, exist_ok=True)


# 1. Parametr statistikas
def count_params_for(name: str):
    model = train.build_model(name)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_ratio": float(ratio),
    }


# 2. TensorBoard əyriləri
def load_curve(model_name: str, tag: str):
    # train_celebdf SummaryWriter: logs_celeb / celeb_{model}
    event_dir = os.path.join(LOG_DIR, f"celeb_{model_name}")
    if not os.path.isdir(event_dir):
        print(f"[WARN] No log dir for {model_name}: {event_dir}")
        return [], []

    ea = EventAccumulator(event_dir)
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    if tag not in scalar_tags:
        print(f"[WARN] Tag '{tag}' not found for {model_name}")
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def summarize_curve(steps, values, threshold=0.9):
    if not values:
        return {}

    max_val = max(values)
    idx_max = values.index(max_val)
    step_max = steps[idx_max]
    mean_val = float(sum(values) / len(values))

    first_thr = None
    for s, v in zip(steps, values):
        if v >= threshold:
            first_thr = s
            break

    return {
        "max": float(max_val),
        "step_at_max": int(step_max),
        "mean": mean_val,
        "first_step_ge_threshold": None if first_thr is None else int(first_thr),
        "threshold": float(threshold),
    }


# 3. Vizualizasiya helper-lər
def plot_accuracy_curves(curves):
    plt.figure(figsize=(10, 4))

    # Train
    plt.subplot(1, 2, 1)
    for name, data in curves.items():
        steps_tr, vals_tr = data["train"]
        if not steps_tr:
            continue
        plt.plot(steps_tr, vals_tr, label=MODEL_LABELS[name])
    plt.title("Celeb-DF – Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.0)
    plt.grid(True)
    plt.legend()

    # Val
    plt.subplot(1, 2, 2)
    for name, data in curves.items():
        steps_va, vals_va = data["val"]
        if not steps_va:
            continue
        plt.plot(steps_va, vals_va, label=MODEL_LABELS[name])
    plt.title("Celeb-DF – Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.0)
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(RESULTS_DIR, "celebdf_train_val_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved accuracy curves → {out_path}")


def plot_metric_bars(summary):
    metrics = ["acc", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    models = list(summary.keys())
    x = np.arange(len(metrics))
    width = 0.22

    plt.figure(figsize=(8, 4))
    for i, m in enumerate(models):
        vals = [summary[m]["test"][k] for k in metrics]
        plt.bar(x + i * width, vals, width, label=MODEL_LABELS[m])

    plt.xticks(x + width, metric_labels)
    plt.ylim(0.7, 1.0)
    plt.ylabel("Score")
    plt.title("Celeb-DF – Test Metrics per Model")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = os.path.join(RESULTS_DIR, "celebdf_metrics_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved metrics bar chart → {out_path}")


def plot_param_bars(summary):
    models = list(summary.keys())
    totals = [summary[m]["params"]["total"] for m in models]
    trains = [summary[m]["params"]["trainable"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, np.log10(totals), width, label="Total (log10)")
    plt.bar(x + width / 2, np.log10(trains), width, label="Trainable (log10)")

    labels = [MODEL_LABELS[m] for m in models]
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("log10(#parameters)")
    plt.title("Celeb-DF – Parameter Counts (Total vs Trainable)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = os.path.join(RESULTS_DIR, "celebdf_params_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved parameter bar chart → {out_path}")


def plot_runtime(summary):
    models = list(summary.keys())
    latencies = [summary[m]["runtime"]["avg_latency_ms"] for m in models]
    fps_vals = [summary[m]["runtime"]["fps"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    # Latency
    plt.figure(figsize=(8, 4))
    plt.bar(x, latencies)
    labels = [MODEL_LABELS[m] for m in models]
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Latency (ms / frame)")
    plt.title("Celeb-DF – Average Inference Latency (Test Set)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = os.path.join(RESULTS_DIR, "celebdf_latency_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved latency bar chart → {out_path}")

    # FPS
    plt.figure(figsize=(8, 4))
    plt.bar(x, fps_vals)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Frames per Second (FPS)")
    plt.title("Celeb-DF – Throughput on Test Set")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = os.path.join(RESULTS_DIR, "celebdf_fps_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved FPS bar chart → {out_path}")

    # Acc vs latency
    plt.figure(figsize=(6, 5))
    for m in models:
        acc = summary[m]["test"]["acc"]
        lat = summary[m]["runtime"]["avg_latency_ms"]
        plt.scatter(lat, acc, s=60)
        plt.text(lat + 0.01, acc, MODEL_LABELS[m], fontsize=8)

    plt.xlabel("Latency (ms / frame)")
    plt.ylabel("Accuracy")
    plt.title("Celeb-DF – Accuracy vs Latency Trade-off")
    plt.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(RESULTS_DIR, "celebdf_acc_vs_latency.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved accuracy vs latency scatter → {out_path}")


def plot_confusion_matrices(summary):
    import matplotlib.ticker as ticker

    for m, res in summary.items():
        errors = res["test"]["errors"]
        tn, fp, fn, tp = errors["tn"], errors["fp"], errors["fn"], errors["tp"]

        cm = np.array([[tn, fp],
                       [fn, tp]], dtype=float)

        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm_norm, cmap="Blues")

        ax.set_title(f"Celeb-DF – Confusion Matrix – {MODEL_LABELS[m]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticklabels(["Real", "Fake"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.0f}\n({cm_norm[i, j]*100:.1f}%)",
                        ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     format=ticker.PercentFormatter(xmax=1))

        out_path = os.path.join(RESULTS_DIR, f"celebdf_confusion_{m}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FIG] Saved confusion matrix → {out_path}")


# 4. Full pipeline
def analyze_model(name: str):
    print(f"\n=== [CELEB-DF] Analyzing {name} ({MODEL_LABELS[name]}) ===")

    param_stats = count_params_for(name)

    tr_steps, tr_vals = load_curve(name, "acc/train")
    va_steps, va_vals = load_curve(name, "acc/val")

    train_curve = summarize_curve(tr_steps, tr_vals)
    val_curve = summarize_curve(va_steps, va_vals)

    if train_curve and val_curve:
        gap = train_curve["max"] - val_curve["max"]
    else:
        gap = None

    test_metrics = eval_mod.evaluate_model(name)
    cm = test_metrics.get("cm")
    errors = {}
    if cm is not None:
        cm = np.array(cm)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        errors = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    test_summary = {
        "loss": float(test_metrics["loss"]),
        "acc": float(test_metrics["acc"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "errors": errors,
    }

    rt_metrics = benchmark_model(name)

    return {
        "params": param_stats,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "overfitting_gap": None if gap is None else float(gap),
        "test": test_summary,
        "runtime": rt_metrics,
        "raw_curves": {
            "train": (tr_steps, tr_vals),
            "val": (va_steps, va_vals),
        },
    }


def main():
    summary = {}
    for m in MODELS:
        summary[m] = analyze_model(m)

    out_json = os.path.join(RESULTS_DIR, "celebdf_analysis_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[JSON] Saved summary → {out_json}")

    curves = {
        m: {
            "train": summary[m]["raw_curves"]["train"],
            "val": summary[m]["raw_curves"]["val"],
        }
        for m in MODELS
    }

    plot_accuracy_curves(curves)
    plot_metric_bars(summary)
    plot_param_bars(summary)
    plot_runtime(summary)
    plot_confusion_matrices(summary)

    print("\n=== SHORT OVERVIEW (CELEB-DF) ===")
    for m in MODELS:
        res = summary[m]
        acc = res["test"]["acc"]
        f1 = res["test"]["f1"]
        fps = res["runtime"]["fps"]
        gap = res["overfitting_gap"]
        print(f"{MODEL_LABELS[m]:18s} acc={acc:.4f}, f1={f1:.4f}, fps={fps:.1f}, gap={gap:.4f}")


if __name__ == "__main__":
    main()
