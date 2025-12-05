import os, torch, torch.nn as nn, torch.optim as optim
import sys
sys.path.append("src")
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v3_small, efficientnet_b0, resnet18
from dataset_list import FrameListDataset
from tqdm import tqdm
import warnings

# warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#json for Celebdf
JSON = "data/splits/paths_celebdf.json"

OUT  = "models_celeb"; os.makedirs(OUT, exist_ok=True)
LOGS = "logs_celeb";   os.makedirs(LOGS, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# model builder
def build_model(name):
    if name == "mobilenetv3":
        m = mobilenet_v3_small(weights="IMAGENET1K_V1")
        for p in m.features.parameters():
            p.requires_grad = False
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)

    elif name == "efficientnet_b0":
        m = efficientnet_b0(weights="IMAGENET1K_V1")
        for p in m.features.parameters():
            p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)

    else:  # resnet18
        m = resnet18(weights="IMAGENET1K_V1")
        for layer in [m.layer1, m.layer2, m.layer3]:
            for p in layer.parameters():
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, 2)

    return m.to(device)


#Train loop
def train_one(model_name, epochs=30, lr=3e-4, batch=64):

    print(f"\n[CELEB-DF] Training {model_name} on {device.upper()}")

    # dataset-lər
    train_dataset = FrameListDataset(JSON, "train")
    val_dataset   = FrameListDataset(JSON, "val")

    # Class weights
    real_count = len([1 for _, lbl in train_dataset.items if lbl == 0])
    fake_count = len([1 for _, lbl in train_dataset.items if lbl == 1])

    total  = real_count + fake_count
    w_real = total / (2 * real_count)
    w_fake = total / (2 * fake_count)

    sample_weights = [w_real if lbl == 0 else w_fake for _, lbl in train_dataset.items]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    tr = DataLoader(train_dataset, batch_size=batch, sampler=sampler,
                    num_workers=4, pin_memory=True)
    va = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                    num_workers=4, pin_memory=True)

    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(model_name)
    opt = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(os.path.join(LOGS, f"celeb_{model_name}"))

    best, wait, patience = 0.0, 0, 6

    for ep in range(1, epochs + 1):

        #Train
        model.train()
        correct = total_loss = total_samples = 0
        pbar = tqdm(tr, desc=f"[CELEB-DF][{model_name}] Epoch {ep}/{epochs}",
                    leave=False)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            total_samples += bs
            total_loss    += loss.item() * bs
            correct       += (logits.argmax(1) == y).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct/total_samples:.3f}")

        train_acc  = correct / total_samples
        train_loss = total_loss / total_samples

        # Validation
        model.eval()
        correct = total_loss = total_samples = 0

        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss   = crit(logits, y)

                bs = y.size(0)
                total_samples += bs
                total_loss    += loss.item() * bs
                correct       += (logits.argmax(1) == y).sum().item()

        val_acc  = correct / total_samples
        val_loss = total_loss / total_samples

        # TensorBoard
        writer.add_scalar("acc/train", train_acc, ep)
        writer.add_scalar("acc/val",   val_acc,  ep)
        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalar("loss/val",   val_loss,  ep)

        print(f"[CELEB-DF][{model_name}] Ep {ep} → "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        # checkpoint
        ckpt_path = os.path.join(OUT, f"celeb_{model_name}_best.pt")
        if val_acc > best:
            best = val_acc
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early Stopping triggered.")
                break

    writer.close()
    print(f"[CELEB-DF] {model_name} — BEST VAL ACC = {best:.3f}\n")
    return best


if __name__ == "__main__":
    for m in ["mobilenetv3", "efficientnet_b0", "resnet18"]:
        best = train_one(m)
        print(f"[CELEB-DF] {m} → best accuracy: {best:.3f}")
