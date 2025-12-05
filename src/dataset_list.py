import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FrameListDataset(Dataset):
    """Dataset that loads image file paths from a JSON split file.

    JSON structure expected (example in data/splits/paths.json):
    {
      "train": {"real": ["path1.jpg", ...], "fake": ["path2.jpg", ...]},
      "val":   {"real": [...], "fake": [...]},
      "test":  {"real": [...], "fake": [...]}
    }
    """

    def __init__(self, json_path: str, split: str):
        # Load JSON
        with open(json_path, "r") as f:
            sp = json.load(f)
        self.items = [(p, 0) for p in sp[split]["real"]] + [(p, 1) for p in sp[split]["fake"]]
        self.aug = (split == "train")

        # Transforms
        # augmentations for training
        self.t_train = T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),

            T.RandomHorizontalFlip(p=0.5),

            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.02
            ),

            T.RandomRotation(degrees=8),

            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

        self.t_eval = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        img = Image.open(path).convert("RGB")
        x = self.t_train(img) if self.aug else self.t_eval(img)
        return x, label


__all__ = ["FrameListDataset"]
