import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import train_celebdf as train
from dataset_list import FrameListDataset

JSON = "data/splits/paths_celebdf.json"
MODELS_DIR = "models_celeb"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str):
    """Build the model architecture and load the saved Celeb-DF checkpoint."""
    model = train.build_model(name)
    ckpt_path = os.path.join(MODELS_DIR, f"celeb_{name}_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found for '{name}': {ckpt_path}. "
            "Train the model first or place the checkpoint in the models_celeb folder."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def evaluate_model(name: str, batch_size: int = 64):
    """Evaluate a Celeb-DF model on the test split."""
    print(f"\n[CELEB-DF] Evaluating {name} on {device.upper()} ...")

    model = load_model(name)
    ds = FrameListDataset(JSON, "test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    y_true, y_pred, y_prob = [], [], []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            bs = y.size(0)
            total_samples += bs
            total_loss += loss.item() * bs

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    test_loss = total_loss / max(total_samples, 1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nTEST METRICS")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nDetailed Report")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    return {
        "name": name,
        "loss": float(test_loss),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "cm": cm.tolist(),  # JSON-serializable for downstream consumers
    }


if __name__ == "__main__":
    results = {}
    for m in ["mobilenetv3", "efficientnet_b0", "resnet18"]:
        results[m] = evaluate_model(m)

    print("\nSUMMARY")
    for k, v in results.items():
        print(f"\n{k.upper()}: acc={v['acc']:.3f}, f1={v['f1']:.3f}")
