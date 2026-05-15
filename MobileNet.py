import torch
import torchvision.models as models
import torch.nn as nn

#print(f"PyTorch version: {torch.__version__}")
#print(f"Is CUDA (GPU) available? {torch.cuda.is_available()}")

def mobilenet_traffic_sign_model(sign_classes=43):
    # 1. Load Pre-trained MobileNetV3-Large
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # 2. Freeze backbone (Transfer Learning)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Replace the head
    # MobileNetV3 classifier is a Sequential block; we replace the last Linear layer [3]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, sign_classes)

    return model

if __name__ == "__main__":
    # Quick test to see if it loads
    test_model = mobilenet_traffic_sign_model(43)
    print("MobileNetV3 loaded successfully with custom head.")