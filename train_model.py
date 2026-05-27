# train_model.py — FIXED VERSION for Windows
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ── CONFIG ──────────────────────────────────────────────
DATASET_DIR = "dataset"
MODEL_SAVE  = "fruit_cnn.pth"
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 0.001
IMG_SIZE    = 224
# ────────────────────────────────────────────────────────

# ── TRANSFORMS ──────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# ────────────────────────────────────────────────────────


# ── PUT ALL EXECUTION CODE INSIDE THIS BLOCK ────────────
if __name__ == '__main__':

    # Dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, "train"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, "val"),
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    # Model
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    best_val_acc = 0.0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        train_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✅ Model saved (best val acc: {best_val_acc:.1f}%)")

    print(f"\nTraining complete. Best accuracy: {best_val_acc:.1f}%")
    print(f"Weights saved to: {MODEL_SAVE}")