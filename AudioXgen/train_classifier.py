import os

import torch
import torch.nn as nn
import torch.optim as optim
from audio_transformer import AudioTransformer
from torch.utils.data import DataLoader, TensorDataset, random_split

import utils

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0001
MODEL_SAVE_PATH = "models/transformer_classifier.pth"


def train():
    print(f"--- Starting Training on {utils.DEVICE} ---")

    (X_train_full, y_train_full), (_, _) = utils.get_data_splits(test_size=0.2)

    # Internal Split for Validation (Train 80% / Val 20% of the TRAINING set)
    dataset_size = len(X_train_full)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    full_dataset = TensorDataset(X_train_full, y_train_full)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Internal Training Split: {len(train_dataset)} Train / {len(val_dataset)} Val"
    )

    # Initialize Model
    model = AudioTransformer(feature_size=128, seq_length=375, num_classes=2).to(
        utils.DEVICE
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(utils.DEVICE), labels.to(utils.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(utils.DEVICE), labels.to(utils.DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
