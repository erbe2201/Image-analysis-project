import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from ResNet import resnet_traffic_sign_model

#Forcing CPU usage instead of GPU if set to true.
use_CPU = True
device = torch.device("cpu" if use_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Training using: {device}")
#Num of clases defined by dataset
sign_classes = 43

def model_training(model, loader, criterion, optimizer):
    #Starts model training and keeps a list of all model predictions and true labels
    model.train()
    all_preds, all_labels = [], []
    running_loss = 0.0

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
    #f1 score calculation
    score = f1_score(all_labels, all_preds, average='weighted')
    return running_loss / len(loader), score


if __name__ == "__main__":
    print(f"Starting smoke test on {device}...")

    # 1. Setup Model, Loss, Optimizer
    model = resnet_traffic_sign_model(sign_classes=sign_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Fake set of images to test the model if it is worrking
    fake_images = torch.randn(64, 3, 224, 224)
    fake_labels = torch.randint(0, sign_classes, (64,))

    fake_dataset = TensorDataset(fake_images, fake_labels)
    fake_loader = DataLoader(fake_dataset, batch_size=8)
    epoch_num = 10
    # Test run for one time epoch
    for epoch in range(epoch_num):
        try:
            loss, f1 = model_training(model, fake_loader, criterion, optimizer)
            print(f"Epoch [{epoch + 1}/{epoch_num}] - Loss: {loss:.4f}, F1 Score: {f1:.4f}")
        except Exception as e:
            print(f"---Test Failed --- \nError: {e}")