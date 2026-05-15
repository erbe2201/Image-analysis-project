import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from ResNet import resnet_traffic_sign_model
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor, RandomRotation
from MobileNet import mobilenet_traffic_sign_model
from main import ReadDataset

def model_training(model, loader, criterion, optimizer):
    #Starts model training and keeps a list of all model predictions and true labels
    model.train()
    all_preds, all_labels = [], []
    running_loss = 0.0

    start_time = time.time()

    for inputs, labels in loader:
        #All data is moved to the same device as the model
        inputs, labels = inputs.to(device), labels.to(device)

        #Clear all previous gradients to only look at current relecant images
        optimizer.zero_grad()
        #Calculate predictions on input data. Forwards pass
        outputs = model(inputs)
        #Calculate how wrong the model was. Loss
        loss = criterion(outputs, labels)
        #Calculate the weight adjustments. Backwards pass
        loss.backward()
        #Update model weights
        optimizer.step()

        #Stat tracker for tracking every model prediction misstake and the confidence score for each class per image guess.
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    epoch_duration = end_time - start_time
    #f1 score calculation
    score = f1_score(all_labels, all_preds, average='weighted')
    return running_loss / len(loader), score, epoch_duration

def measure_inference_speed(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up (important for GPU)
    for _ in range(10):
        _ = model(dummy_input)

    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    end = time.time()

    avg_time_ms = ((end - start) / 100) * 1000
    print(f"--- Performance Data ---")
    print(f"Average Inference Time: {avg_time_ms:.2f} ms per image")
    print(f"Throughput: {1000 / avg_time_ms:.2f} images per second")

if __name__ == "__main__":
    # Forcing CPU usage instead of GPU if set to true.
    use_CPU = False
    device = torch.device("cpu" if use_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training using: {device}")
    # Num of clases defined by dataset
    sign_classes = 43

    print(f"Starting smoke test on {device}...")

    """" 1. Setup Model, Loss, Optimizer
    model = resnet_traffic_sign_model(sign_classes=sign_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Fake set of images to test the model if it is worrking
    fake_images = torch.randn(64, 3, 224, 224)
    fake_labels = torch.randint(0, sign_classes, (64,))

    fake_dataset = TensorDataset(fake_images, fake_labels)
    fake_loader = DataLoader(fake_dataset, batch_size=8)"""

    DATASET_PATH = r"C:\Users\Erik\Desktop\Image Analysis Dataset"
    train_df = pd.read_csv(f"{DATASET_PATH}/Train.csv")
    train_df = train_df.sample(frac=1.0).reset_index(drop=True)
    test_df = pd.read_csv(f"{DATASET_PATH}/Test.csv")
    test_df = test_df.sample(frac=1.0).reset_index(drop=True)

    # Define transformations (Matched to 224 for ResNet)
    transform = Compose([
        Resize((224, 224)),  # Changed from 32 to 224 to fit ResNet requirements
        RandomRotation(degrees=15),
        ToTensor()
    ])

    # Initialize the real dataset
    train_dataset = ReadDataset(train_df, DATASET_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f"Dataset loaded with {len(train_dataset)} training images.")

    # --- 2. MODEL SETUP ---
    model = mobilenet_traffic_sign_model(sign_classes=sign_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimization Tip: Since this is real data, optimize ALL parameters,
    # not just the last layer (model.fc), unless you are doing strict Transfer Learning.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_num = 10

    # Test run for one time epoch
    for epoch in range(epoch_num):
        try:
            clock_time = time.time()
            loss, f1, duration = model_training(model, train_loader, criterion, optimizer)
            clock_end = time.time()
            clock_total = clock_end - clock_time
            print(f"Epoch [{epoch + 1}/{epoch_num}] - Loss: {loss:.4f}, F1: {f1:.4f}, Time: {duration:.2f}s")
            print(f"  - Total (Wall) Time: {clock_total:.2f}s")
        except Exception as e:
            print(f"---Test Failed --- \nError: {e}")
            break

    print("\nRunning Speed Test...")
    measure_inference_speed(model, device)

