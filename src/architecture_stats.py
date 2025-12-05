import torch
from torchvision.models import (
    mobilenet_v3_small, efficientnet_b0, resnet18,
    MobileNet_V3_Small_Weights, EfficientNet_B0_Weights, ResNet18_Weights
)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total(model):
    return sum(p.numel() for p in model.parameters())

# Build models exactly like in your training code
def build_mobilenet():
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 2)
    return m

def build_efficientnet():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    return m

def build_resnet():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Freeze layer1–3 exactly like in your code
    for layer in [m.layer1, m.layer2, m.layer3]:
        for p in layer.parameters():
            p.requires_grad = False
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m

models = {
    "MobileNetV3": build_mobilenet(),
    "EfficientNetB0": build_efficientnet(),
    "ResNet18": build_resnet(),
}

for name, model in models.items():
    print("\n==============================")
    print(name)
    print("Total parameters:    ", count_total(model))
    print("Trainable parameters:", count_trainable(model))
    print("==============================")
