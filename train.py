import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data_loader import UTKFaceDataset
from model import EthnicityClassifier
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = UTKFaceDataset("data/UTKFace", transform=transform)
print(f"Total images: {len(dataset)}")


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


labels = [label for _, label in dataset]
counts = Counter(labels)
print("Class counts:", counts)

num_classes = 5
weights = torch.tensor(
    [1.0 / counts.get(i, 1) for i in range(num_classes)],
    dtype=torch.float
).to(device)
print("Class weights:", weights)


model = EthnicityClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 20
best_val_acc = 0.0

for epoch in range(epochs):
    
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

 
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {total_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

    
    if val_acc > best_val_acc:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/ethnicity_model_best.pth")
        best_val_acc = val_acc
        print(f"✅ Saved new best model (Val Acc: {val_acc:.2f}%)")

print("Training finished. Best Val Acc:", best_val_acc)
