import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from torchvision.models import (
    mobilenet_v3_small, efficientnet_b0, resnet18,
    MobileNet_V3_Small_Weights, EfficientNet_B0_Weights, ResNet18_Weights
)
from dataset_list import FrameListDataset
import numpy as np

JSON = "data/splits/paths.json"
MODELS_DIR = "models"
device = "cuda" if torch.cuda.is_available() else "cpu"


# Model loader
def load_model(name):
    if name == "mobilenetv3":
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)

    elif name == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)

    else:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)

    ckpt_path = os.path.join(MODELS_DIR, f"{name}_best.pt")
    state = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(state)
    return m.to(device).eval()


# Evaluation
def evaluate_model(name, batch_size=64):
    print(f"\n Evaluating {name} ...")

    model = load_model(name)
    ds = FrameListDataset(JSON, "test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    y_true = []
    y_pred = []
    y_prob = []
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    test_loss = total_loss / total_samples
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
        "loss": test_loss,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
    }


# MAIN
if __name__ == "__main__":
    results = {}
    for m in ["mobilenetv3", "efficientnet_b0", "resnet18"]:
        results[m] = evaluate_model(m)

    print("\nSUMMARY")
    for k, v in results.items():
        print(f"\n{k.upper()}: acc={v['acc']:.3f}, f1={v['f1']:.3f}")
