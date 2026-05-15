import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, RandomRotation
from PIL import Image
import os

# 2. Create custom Dataset class
class ReadDataset(Dataset):
    def __init__(self, dataframe, base_path, transform=None):
        self.dataframe = dataframe
        self.base_path = base_path
        self.transform = transform or Compose([
            Resize((32, 32)),
            RandomRotation(degrees=15),
            ToTensor()
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.base_path, row['Path'])
        image = Image.open(img_path).convert('RGB')
        label = row['ClassId']

        if self.transform:
            image = self.transform(image)

        return image, label


# 4. Define CNN Model
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    # Configuration
    DATASET_PATH = r"C:\Users\Erik\Desktop\Image Analysis Dataset"
    NUM_CLASSES = 43
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "gtsrb_model.pth"

    # 1. Load metadata and annotations
    meta_df = pd.read_csv(f"{DATASET_PATH}/Meta.csv")
    train_df = pd.read_csv(f"{DATASET_PATH}/Train.csv")
    test_df = pd.read_csv(f"{DATASET_PATH}/Test.csv")

    print(f"Number of classes: {len(meta_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")



    # 3. Create datasets and dataloaders
    train_dataset = ReadDataset(train_df, DATASET_PATH)
    test_dataset = ReadDataset(test_df, DATASET_PATH)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Verify data loading
    for X, y in train_loader:
        print(f"Batch shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        break


    # 5. Initialize model, loss, optimizer
    model = TrafficSignCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training function
    def train_epoch(model, train_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    # 7. Validation function
    def validate(model, test_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    # 8. Train the model
    print(f"\nTraining on device: {DEVICE}\n")
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Valid - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Model saved! (Best accuracy: {best_accuracy:.2f}%)")
        print()

    print(f"Training complete! Best model saved to '{MODEL_PATH}'")
    print(f"Best test accuracy: {best_accuracy:.2f}%")