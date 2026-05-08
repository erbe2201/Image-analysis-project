import torch
import torchvision.models as models
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA (GPU) available? {torch.cuda.is_available()}")

def resnet_traffic_sign_model(sign_classes=43):
    #Pre-trained model weights
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Freeze backbone and prevent pre-trained weights from being destroyed
    for param in model.parameters():
        param.requires_grad = False

    # Replace head
    model.fc = nn.Linear(model.fc.in_features, sign_classes)

    return model

if __name__ == "__main__":
    # Quick test to see if it loads
    test_model = resnet_traffic_sign_model(43)
    print("ResNet50 loaded successfully with custom head.")